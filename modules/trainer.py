"""Generic training loop, shared across all four model types. Each model
only needs to implement `training_step(batch) -> (loss, log_dict)` and
`parameters()`; everything else (optimizer, gradient clipping, epoch
logging, loss history) lives here."""
import time

import torch
from torch.utils.data import DataLoader

def train_model(model, dataset, method_name, dataset_name, cfg, logger):
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history = []
    log_every = max(1, cfg.epochs // 10)
    t_start = time.time()
    for epoch in range(cfg.epochs):
        epoch_losses = []
        for batch in loader:
            opt.zero_grad()
            loss, logs = model.training_step(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_losses.append(logs["loss"])

        mean_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        history.append({"epoch": epoch, "loss": mean_loss})
        if (epoch + 1) % log_every == 0 or epoch == 0 or epoch == cfg.epochs - 1:
            logger.info(f"[{dataset_name}/{method_name}] epoch {epoch + 1}/{cfg.epochs} loss={mean_loss:.5f}")

    train_time = time.time() - t_start
    logger.info(f"[{dataset_name}/{method_name}] training finished in {train_time:.1f}s ({len(dataset)} examples, {cfg.epochs} epochs)")
    return history, train_time