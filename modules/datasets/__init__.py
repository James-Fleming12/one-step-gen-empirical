from .real import REAL_REGISTRY, RealImageDataset
from .synthetic import SYNTHETIC_REGISTRY, SyntheticDataset, sample_synthetic

ALL_DATASETS = sorted(SYNTHETIC_REGISTRY) + sorted(REAL_REGISTRY)

def get_dataset(name: str, n_samples: int = 20000, seed: int = 0, data_root: str = "./data", subset_size: int = 8000, image_size: int = 28):
    """Returns (dataset, data_info). data_info describes the data modality so
    models/metrics know which network backbone and metrics to use."""
    if name in SYNTHETIC_REGISTRY:
        ds = SyntheticDataset(name, n_samples=n_samples, seed=seed)
        return ds, {"type": "synthetic", "dim": ds.dim}
    elif name in REAL_REGISTRY:
        ds = RealImageDataset(name, root=data_root, train=True, subset_size=subset_size, image_size=image_size, seed=seed)
        return ds, {"type": "image", "channels": ds.channels, "image_size": ds.image_size}
    else:
        raise ValueError(f"Unknown dataset '{name}'. Choices: {ALL_DATASETS}")