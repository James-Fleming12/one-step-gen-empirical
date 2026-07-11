import argparse
import os
import time

import torch

from modules import metrics, trainer, utils
from modules.datasets import ALL_DATASETS, REAL_REGISTRY, SYNTHETIC_REGISTRY, get_dataset, sample_synthetic
from modules.models import MODEL_REGISTRY, ONE_STEP_METHODS, get_model

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(REPO_ROOT, "logs")
WEIGHT_DIR = os.path.join(REPO_ROOT, "weights")

def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--dataset", type=str, default="checkerboard", choices=ALL_DATASETS, help="Which dataset to train/evaluate on.")
    p.add_argument("--method", type=str, default="all", help="Comma-separated list of methods to (re)run, or 'all'. Choices: {list(MODEL_REGISTRY)}")
    p.add_argument("--force", action="store_true", help="Retrain from scratch even if a checkpoint already exists for a method. Without this flag, existing checkpoints are loaded and only re-evaluated.")

    p.add_argument("--epochs", type=int, default=None, help="Training epochs. Default: 150 for synthetic datasets, 20 for real datasets.")
    p.add_argument("--batch_size", type=int, default=None, help="Default: 256 for synthetic datasets, 128 for real datasets.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256, help="MLP hidden width (synthetic datasets only).")
    p.add_argument("--depth", type=int, default=4, help="MLP depth (synthetic datasets only).")

    p.add_argument("--n_samples", type=int, default=20000, help="Synthetic dataset size.")
    p.add_argument("--subset_size", type=int, default=8000, help="Real dataset subset size.")
    p.add_argument("--image_size", type=int, default=28, help="Real dataset image resolution.")
    p.add_argument("--data_root", type=str, default=os.path.join(REPO_ROOT, "data"))

    p.add_argument("--eval_samples", type=int, default=2000, help="Number of samples drawn for evaluation.")
    p.add_argument("--sample_steps", type=int, default=50, help="Euler steps for the flow_matching baseline.")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None, help="'cpu' or 'cuda'. Default: auto-detect.")
    p.add_argument("--smoke_test", action="store_true", help="Override sizes/epochs with tiny values, for a fast end-to-end sanity check.")

    return p

def resolve_config(args):
    is_synthetic = args.dataset in SYNTHETIC_REGISTRY
    if args.epochs is None:
        args.epochs = 150 if is_synthetic else 20
    if args.batch_size is None:
        args.batch_size = 256 if is_synthetic else 128
    if args.device is None:
        args.device = utils.get_device()

    if args.smoke_test:
        args.epochs = 2
        args.n_samples = 500
        args.subset_size = 300
        args.batch_size = 64
        args.eval_samples = 128
        args.hidden = 32
        args.depth = 2
        args.sample_steps = 5

    if args.method == "all":
        args.methods = list(MODEL_REGISTRY)
    else:
        args.methods = [m.strip() for m in args.method.split(",") if m.strip()]
        for m in args.methods:
            if m not in MODEL_REGISTRY:
                raise ValueError(f"Unknown method '{m}'. Choices: {list(MODEL_REGISTRY)}")
    return args

def get_eval_step_candidates(method_name, cfg):
    if method_name in ONE_STEP_METHODS:
        return [1, 2, 4]
    return sorted({1, cfg.sample_steps})

def get_native_steps(method_name, cfg):
    return cfg.sample_steps if method_name not in ONE_STEP_METHODS else 1

def run_evaluation(model, method_name, data_info, ref_samples, cfg, logger, dataset_name):
    results = {}
    step_candidates = get_eval_step_candidates(method_name, cfg)

    encoder = None
    feat_ref = None
    if data_info["type"] == "image":
        encoder = metrics.RandomFeatureEncoder(channels=data_info["channels"]).to(model.device)
        feat_ref = encoder(ref_samples.to(model.device))

    for steps in step_candidates:
        t0 = time.time()
        with torch.no_grad():
            samples = model.sample(cfg.eval_samples, steps=steps)
        elapsed = time.time() - t0
        nfe = model.nfe_for_sampling(steps)
        entry = {"steps": steps, "nfe": nfe, "sample_time_sec": round(elapsed, 4)}

        if data_info["type"] == "synthetic":
            entry["mmd"] = metrics.rbf_mmd(samples.cpu(), ref_samples)
            entry["swd"] = metrics.sliced_wasserstein(samples.cpu(), ref_samples)
        else:
            feat_gen = encoder(samples.to(model.device))
            entry["frechet_distance_proxy"] = metrics.frechet_distance(feat_gen, feat_ref)
            n_cmp = min(500, samples.shape[0], ref_samples.shape[0])
            entry["pixel_mmd"] = metrics.rbf_mmd(samples.cpu()[:n_cmp], ref_samples.cpu()[:n_cmp])
            grid_path = os.path.join(LOG_DIR, f"{dataset_name}_{method_name}_steps{steps}_grid.png")
            metrics.save_image_grid(samples[:64], grid_path)

        logger.info(f"[{dataset_name}/{method_name}] eval steps={steps} nfe={nfe} -> { {k: v for k, v in entry.items() if k not in ('steps', 'nfe')} }")
        results[f"steps_{steps}"] = entry

    if data_info["type"] == "synthetic":
        native_steps = get_native_steps(method_name, cfg)
        with torch.no_grad():
            plot_samples = model.sample(min(cfg.eval_samples, 2000), steps=native_steps)
        plot_path = os.path.join(LOG_DIR, f"{dataset_name}_{method_name}_scatter.png")
        metrics.save_scatter(plot_samples.cpu(), ref_samples[:2000], plot_path, title=f"{method_name} ({dataset_name}, {native_steps}-step)")

    return results

def run_one_method(method_name, dataset, data_info, cfg, ref_samples):
    log_path = os.path.join(LOG_DIR, f"{cfg.dataset}_{method_name}.log")
    logger = utils.get_logger(f"{cfg.dataset}_{method_name}", log_path)
    weight_path = os.path.join(WEIGHT_DIR, f"{cfg.dataset}_{method_name}.pt")

    logger.info(f"=== {method_name} on {cfg.dataset} ({data_info['type']}) ===")
    logger.info(f"config: epochs={cfg.epochs} batch_size={cfg.batch_size} lr={cfg.lr} device={cfg.device} seed={cfg.seed}")

    model = get_model(method_name, data_info, device=cfg.device, hidden=cfg.hidden, depth=cfg.depth, sample_steps=cfg.sample_steps)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"model params: {n_params:,}")

    history, train_time = None, None
    if os.path.exists(weight_path) and not cfg.force:
        logger.info(f"found existing checkpoint at {weight_path}, loading (use --force to retrain)")
        model.load_state_dict(torch.load(weight_path, map_location=cfg.device))
    else:
        history, train_time = trainer.train_model(model, dataset, method_name, cfg.dataset, cfg, logger)
        os.makedirs(WEIGHT_DIR, exist_ok=True)
        torch.save(model.state_dict(), weight_path)
        logger.info(f"saved weights to {weight_path}")

    eval_results = run_evaluation(model, method_name, data_info, ref_samples, cfg, logger, cfg.dataset)

    metrics_out = {
        "dataset": cfg.dataset,
        "method": method_name,
        "n_params": n_params,
        "train_time_sec": train_time,
        "loss_history": history,
        "eval": eval_results,
    }
    metrics_path = os.path.join(LOG_DIR, f"{cfg.dataset}_{method_name}_metrics.json")
    utils.save_json(metrics_out, metrics_path)
    logger.info(f"saved metrics to {metrics_path}")
    return metrics_out

def print_summary_table(all_results, data_info):
    print("\n" + "=" * 78)
    print(f"SUMMARY: dataset={all_results[0]['dataset']}")
    print("=" * 78)
    primary_metric = "mmd" if data_info["type"] == "synthetic" else "frechet_distance_proxy"
    header = f"{'method':<22}{'NFE':>6}{'sample_time_s':>16}{primary_metric:>16}{'n_params':>14}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        for step_key, entry in r["eval"].items():
            label = r["method"] if step_key == list(r["eval"].keys())[0] else ""
            print(f"{r['method']+'/'+step_key:<22}{entry['nfe']:>6}{entry['sample_time_sec']:>16.4f}{entry.get(primary_metric, float('nan')):>16.5f}{r['n_params']:>14,}")

    print("=" * 78 + "\n")

def main():
    args = build_arg_parser().parse_args()
    args.dataset_orig = args.dataset
    cfg = resolve_config(args)

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(WEIGHT_DIR, exist_ok=True)

    utils.set_seed(cfg.seed)
    print(f"Device: {cfg.device} | Dataset: {cfg.dataset} | Methods: {cfg.methods}")

    dataset, data_info = get_dataset(cfg.dataset, n_samples=cfg.n_samples, seed=cfg.seed, data_root=cfg.data_root, subset_size=cfg.subset_size, image_size=cfg.image_size)

    if data_info["type"] == "synthetic":
        ref_samples = sample_synthetic(cfg.dataset, cfg.eval_samples, seed=cfg.seed + 12345)
    else:
        from modules.datasets.real import RealImageDataset
        ref_ds = RealImageDataset(cfg.dataset, root=cfg.data_root, train=False, subset_size=min(cfg.eval_samples, 2000), image_size=cfg.image_size, seed=cfg.seed + 12345)
        ref_samples = ref_ds.images

    all_results = []
    for method_name in cfg.methods:
        try:
            result = run_one_method(method_name, dataset, data_info, cfg, ref_samples)
            all_results.append(result)
        except Exception as e:
            print(f"[ERROR] method '{method_name}' failed: {e}")
            import traceback
            traceback.print_exc()

    if all_results:
        print_summary_table(all_results, data_info)
        utils.save_json(all_results, os.path.join(LOG_DIR, f"{cfg.dataset}_comparison.json"))

if __name__ == "__main__":
    main()