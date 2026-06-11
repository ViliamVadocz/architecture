from pathlib import Path
from typing import Any, NamedTuple

import torch as tch
import wandb
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe

from data import Data, TakDataset
from models.baseline.model import Baseline
from models.baseline.representation import game_to_tensor, policy_to_tensors


def loss_fn(
    policy: Tensor,
    policy_target: Tensor,
    policy_mask: Tensor,
    value: Tensor,
    value_target: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    policy_loss = nn.functional.cross_entropy(policy[policy_mask], policy_target[policy_mask])
    value_loss = nn.functional.mse_loss(value, value_target)
    total_loss = value_loss + policy_loss
    return policy_loss, value_loss, total_loss


class Checkpoint(NamedTuple):
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    scheduler_state: dict[str, Any]
    epoch: int
    best_validation_loss: float


class Resume(NamedTuple):
    id: str
    checkpoint_path: Path


class Config(NamedTuple):
    learning_rate: float
    epochs: int
    checkpoint_interval: int
    checkpoint_dir: Path
    batch_size: int
    resume: Resume | None


def save_checkpoint(  # noqa: PLR0913
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_validation_loss: float,
) -> None:
    path.parent.mkdir(exist_ok=True)
    checkpoint = Checkpoint(
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict(),
        epoch=epoch,
        best_validation_loss=best_validation_loss,
    )
    tch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
) -> tuple[int, float]:
    if path is None:
        return 0, float("inf")
    checkpoint: Checkpoint = tch.load(path)
    model.load_state_dict(checkpoint.model_state)
    optimizer.load_state_dict(checkpoint.optimizer_state)
    scheduler.load_state_dict(checkpoint.scheduler_state)
    return checkpoint.epoch, checkpoint.best_validation_loss


def validate(model: nn.Module, loader: DataLoader, device: str, batches: int = 1024) -> float:
    model.eval()
    total_loss = 0
    count = 0
    with tch.no_grad():
        batch: Data
        for batch in loader:
            observation = batch.observation.to(device)
            value_target = batch.value.to(device, dtype=tch.float32)
            policy_mask = batch.mask.to(device)
            policy_target = batch.policy.to(device)

            policy, value = model(observation)
            _, _, loss = loss_fn(policy, policy_target, policy_mask, value, value_target)
            total_loss += loss.sum()
            count += 1
            if count >= batches:
                break
    avg_val_loss = total_loss / count
    assert isinstance(avg_val_loss, Tensor)
    return avg_val_loss.item()


if __name__ == "__main__":
    assert tch.cuda.is_available()
    assert tch.cuda.device_count() == 1
    device = "cuda"

    config = Config(
        learning_rate=1e-3,
        epochs=100,
        checkpoint_interval=1,
        checkpoint_dir=Path("./checkpoints").absolute(),
        batch_size=256,
        resume=None,
    )
    config.checkpoint_dir.mkdir(exist_ok=True)

    current_dir = Path.cwd()
    selfplay_path = current_dir / "target-selfplay-reversed.txt"
    reanalyze_path = current_dir / "target-reanalyze-reversed.txt"

    train_dataset = TakDataset(selfplay_path, game_to_tensor, policy_to_tensors)
    train_shuffled_dataset = ShufflerIterDataPipe(train_dataset)
    train_loader = DataLoader(train_shuffled_dataset, batch_size=config.batch_size, num_workers=16)

    validation_dataset = TakDataset(reanalyze_path, game_to_tensor, policy_to_tensors)
    validation_shuffled_dataset = ShufflerIterDataPipe(validation_dataset)
    validation_loader = DataLoader(validation_shuffled_dataset, batch_size=config.batch_size, num_workers=16)

    model: nn.Module = Baseline()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1_000_000, gamma=0.5)

    # Load checkpoint
    start_epoch, best_validation_loss = 0, float("inf")
    resume_id = None
    if config.resume is not None:
        start_epoch, best_validation_loss = load_checkpoint(config.resume.checkpoint_path, model, optimizer, scheduler)
        resume_id = config.resume.id

    run = wandb.init(
        entity="viliam-vadocz-team",
        project="equivariant-tak",
        config=config._asdict(),
        dir=Path("./.wandb").absolute(),
        id=resume_id,
        resume="allow",
    )

    # Training loop
    step = 0
    for epoch in range(start_epoch, config.epochs):
        model.train()
        epoch_policy_loss = tch.zeros((), device=device)
        epoch_value_loss = tch.zeros((), device=device)
        epoch_total_loss = tch.zeros((), device=device)
        batch_count = 0

        batch: Data
        for batch in train_loader:
            observation = batch.observation.to(device)
            value_target = batch.value.to(device, dtype=tch.float32)
            policy_mask = batch.mask.to(device)
            policy_target = batch.policy.to(device)

            optimizer.zero_grad()
            policy, value = model(observation)
            policy_loss, value_loss, total_loss = loss_fn(policy, policy_target, policy_mask, value, value_target)
            total_loss.backward()
            optimizer.step()

            epoch_policy_loss += policy_loss.detach()
            epoch_value_loss += value_loss.detach()
            epoch_total_loss += total_loss.detach()
            batch_count += 1

            step += 1
            if step % 100 == 0:  # every 100 batches
                wandb.log(
                    {
                        "policy_loss": policy_loss,
                        "value_loss": value_loss,
                        "total_loss": total_loss,
                    },
                    step,
                )

        validation_loss = validate(model, validation_loader, device)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            save_checkpoint(
                config.checkpoint_dir / run.id / "best.pt", model, optimizer, scheduler, epoch, best_validation_loss
            )

        if (epoch + 1) % config.checkpoint_interval == 0:
            path = config.checkpoint_dir / run.id / f"epoch-{epoch:0>4}"
            save_checkpoint(path, model, optimizer, scheduler, epoch, best_validation_loss)

        wandb.log(
            {
                "epoch": epoch,
                "epoch_policy_loss": epoch_policy_loss / batch_count,
                "epoch_value_loss": epoch_value_loss / batch_count,
                "epoch_total_loss": epoch_total_loss / batch_count,
                "validation_loss": validation_loss.item(),
            },
            step,
        )

        scheduler.step()

    run.finish()
