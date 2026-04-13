"""Training script for SO-100 action-chunking imitation learning.

Imports a model from hw3.model and trains it on
state -> action-chunk prediction using the processed zarr dataset.

Usage:
    python scripts/train.py --zarr datasets/processed/single_cube/processed_ee_xyz.zarr \
        --state-keys ... \
        --action-keys ...
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import fmean

import torch
import zarr as zarr_lib
from hw3.dataset import (
    Normalizer,
    SO100ChunkDataset,
    load_and_merge_zarrs,
    load_zarr,
)
from hw3.model import BasePolicy, build_policy

from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

EPOCHS = 500
BATCH_SIZE = 256
LR = 3e-3
VAL_SPLIT = 0.05
LOSS_START = "mse"
LOSS_END = "lp"
LOSS_SWITCH_THRESHOLD = 1e-3
LOSS_SWITCH_WINDOW = 30
LOSS_SWITCH_METRIC = "val"
MIN_EPOCHS_BEFORE_SWITCH = 200
EARLY_STOP_THRESHOLD = 1e-4
EARLY_STOP_WINDOW = 20
LP_P = 1.0
SMOOTH_L1_BETA = 0.5


def compute_regression_loss(
    model: BasePolicy,
    states: torch.Tensor,
    action_chunks: torch.Tensor,
    *,
    loss_name: str,
    lp_p: float,
    smooth_l1_beta: float,
) -> torch.Tensor:
    pred = model.sample_actions(states)

    if loss_name == "mse":
        return F.mse_loss(pred, action_chunks)
    if loss_name == "l1":
        return F.l1_loss(pred, action_chunks)
    if loss_name == "smooth_l1":
        return F.smooth_l1_loss(pred, action_chunks, beta=smooth_l1_beta)
    if loss_name == "lp":
        if lp_p <= 0:
            raise ValueError(f"lp_p must be > 0, got {lp_p}")
        return (pred - action_chunks).abs().pow(lp_p).mean()
    raise ValueError(f"Unknown loss_name: {loss_name}")


def train_one_epoch(
    model: BasePolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    loss_name: str,
    lp_p: float,
    smooth_l1_beta: float,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        states, action_chunks = batch
        states = states.to(device)
        action_chunks = action_chunks.to(device)

        optimizer.zero_grad(set_to_none=True)
        loss = compute_regression_loss(
            model,
            states,
            action_chunks,
            loss_name=loss_name,
            lp_p=lp_p,
            smooth_l1_beta=smooth_l1_beta,
        )
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: BasePolicy,
    loader: DataLoader,
    device: torch.device,
    *,
    loss_name: str,
    lp_p: float,
    smooth_l1_beta: float,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        states, action_chunks = batch
        states = states.to(device)
        action_chunks = action_chunks.to(device)
        loss = compute_regression_loss(
            model,
            states,
            action_chunks,
            loss_name=loss_name,
            lp_p=lp_p,
            smooth_l1_beta=smooth_l1_beta,
        )
        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


def should_switch_loss(
    metric_history: list[float],
    *,
    window: int,
    threshold: float,
) -> tuple[bool, float | None]:
    if len(metric_history) < window + 1:
        return False, None

    recent = metric_history[-(window + 1) :]
    deltas = [abs(curr - prev) for prev, curr in zip(recent[:-1], recent[1:], strict=True)]
    avg_delta = fmean(deltas)
    return avg_delta < threshold, avg_delta


def resolve_data_keys(
    zarr_paths: list[Path],
    requested_keys: list[str] | None,
    *,
    attr_name: str,
    default_key: str,
) -> list[str]:
    if requested_keys is not None:
        return requested_keys

    resolved_keys: list[str] | None = None
    for zarr_path in zarr_paths:
        root = zarr_lib.open_group(str(zarr_path), mode="r")
        keys = [root.attrs.get(attr_name, default_key)]
        if resolved_keys is None:
            resolved_keys = keys
        elif resolved_keys != keys:
            raise ValueError(
                f"Mismatched default {attr_name} across zarr stores: "
                f"{resolved_keys} vs {keys} from {zarr_path}"
            )

    if resolved_keys is None:
        raise ValueError("No zarr paths were provided.")
    return resolved_keys


def main() -> None:
    parser = argparse.ArgumentParser(description="Train action-chunking policy.")
    parser.add_argument(
        "--zarr", type=Path, required=True, help="Path to processed .zarr store."
    )
    parser.add_argument(
        "--extra-zarr",
        type=Path,
        nargs="*",
        default=None,
        help="Optional additional processed .zarr stores to merge into training.",
    )
    parser.add_argument(
        "--policy",
        choices=["obstacle", "multitask"],
        default="obstacle",
        help="Policy type: 'obstacle' for single-cube obstacle scene, 'multitask' for multicube (default: obstacle).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16,
        help="Action chunk horizon H (default: 16).",
    )
    parser.add_argument(
        "--state-keys",
        nargs="+",
        default=None,
        help='State array key specs to concatenate, e.g. state_ee_xyz state_gripper "state_cube[:3]". '
        "Supports column slicing with [:N], [M:], [M:N]. "
        "If omitted, uses the state_key attribute from the zarr metadata.",
    )
    parser.add_argument(
        "--action-keys",
        nargs="+",
        default=None,
        help="Action array key specs to concatenate, e.g. action_ee_xyz action_gripper. "
        "Supports column slicing with [:N], [M:], [M:N]. "
        "If omitted, uses the action_key attribute from the zarr metadata.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--d-model", type=int, default=350, help="Hidden width of the MLP policy.")
    parser.add_argument("--depth", type=int, default=8, help="Number of hidden layers in the MLP policy.")
    parser.add_argument(
        "--loss-schedule",
        choices=["fixed", "plateau_switch"],
        default="fixed",
        help="Use fixed MSE training or enable the optional plateau-based loss switch.",
    )
    parser.add_argument(
        "--loss-end",
        choices=["l1", "smooth_l1", "lp"],
        default=LOSS_END,
        help="Loss to switch to when --loss-schedule plateau_switch is enabled.",
    )
    parser.add_argument(
        "--switch-window",
        type=int,
        default=LOSS_SWITCH_WINDOW,
        help="Rolling window size used by the optional plateau-based loss switch.",
    )
    parser.add_argument(
        "--switch-threshold",
        type=float,
        default=LOSS_SWITCH_THRESHOLD,
        help="Switch threshold for the optional plateau-based loss switch.",
    )
    parser.add_argument(
        "--min-epochs-before-switch",
        type=int,
        default=MIN_EPOCHS_BEFORE_SWITCH,
        help="Minimum number of epochs to complete before allowing a loss switch.",
    )
    parser.add_argument(
        "--lp-p",
        type=float,
        default=LP_P,
        help="Exponent p for the generalized Lp loss when using --loss-end lp.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load data ─────────────────────────────────────────────────────
    zarr_paths = [args.zarr]
    if args.extra_zarr:
        zarr_paths.extend(args.extra_zarr)
    effective_state_keys = resolve_data_keys(
        zarr_paths,
        args.state_keys,
        attr_name="state_key",
        default_key="state",
    )
    effective_action_keys = resolve_data_keys(
        zarr_paths,
        args.action_keys,
        attr_name="action_key",
        default_key="action",
    )

    if len(zarr_paths) == 1:
        states, actions, ep_ends = load_zarr(
            args.zarr,
            state_keys=effective_state_keys,
            action_keys=effective_action_keys,
        )
    else:
        print(f"Merging {len(zarr_paths)} zarr stores: {[str(p) for p in zarr_paths]}")
        states, actions, ep_ends = load_and_merge_zarrs(
            zarr_paths,
            state_keys=effective_state_keys,
            action_keys=effective_action_keys,
        )
    normalizer = Normalizer.from_data(states, actions)

    dataset = SO100ChunkDataset(
        states,
        actions,
        ep_ends,
        chunk_size=args.chunk_size,
        normalizer=normalizer,
    )
    print(f"Dataset: {len(dataset)} samples, chunk_size={args.chunk_size}")
    print(f"  state_dim={states.shape[1]}, action_dim={actions.shape[1]}")

    # ── train / val split ─────────────────────────────────────────────
    n_val = max(1, int(len(dataset) * VAL_SPLIT))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # ── model ─────────────────────────────────────────────────────────
    model = build_policy(
        args.policy,
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        chunk_size=args.chunk_size,
        d_model=args.d_model,
        depth=args.depth,
        state_keys=effective_state_keys,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── training loop ─────────────────────────────────────────────────
    best_val = float("inf")
    current_loss_name = LOSS_START
    loss_switched = args.loss_schedule == "fixed"
    train_metric_history: list[float] = []
    val_metric_history: list[float] = []
    if args.loss_schedule == "plateau_switch":
        print(
            "Loss schedule: "
            f"{LOSS_START} -> {args.loss_end} | "
            f"min_epochs_before_switch={args.min_epochs_before_switch} | "
            f"window={args.switch_window} | "
            f"metric={LOSS_SWITCH_METRIC} | "
            f"threshold={args.switch_threshold}"
        )
    else:
        print("Loss schedule: fixed mse")
    print(
        "Early stopping: "
        f"window={EARLY_STOP_WINDOW} | "
        f"metric={LOSS_SWITCH_METRIC} | "
        f"threshold={EARLY_STOP_THRESHOLD}"
    )

    # Derive action space tag from action keys (e.g. "ee_xyz", "joints")
    action_space = "unknown"
    if effective_action_keys:
        for k in effective_action_keys:
            base = k.split("[")[0]  # strip column slices
            if base != "action_gripper":
                action_space = base.removeprefix("action_")
                break

    save_name = f"best_model_{action_space}_{args.policy}.pt"

    n_dagger_eps = 0
    for zp in zarr_paths:
        z = zarr_lib.open_group(str(zp), mode="r")
        n_dagger_eps += z.attrs.get("num_dagger_episodes", 0)
    if n_dagger_eps > 0:
        save_name = f"best_model_{action_space}_{args.policy}_dagger{n_dagger_eps}ep.pt"
    # Default: checkpoints/<task>/
    if "multi_cube" in str(args.zarr):
        ckpt_dir = Path("./checkpoints/multi_cube")
    else:
        ckpt_dir = Path("./checkpoints/single_cube")
    save_path = ckpt_dir / save_name
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_name=current_loss_name,
            lp_p=args.lp_p,
            smooth_l1_beta=SMOOTH_L1_BETA,
        )
        val_loss = evaluate(
            model,
            val_loader,
            device,
            loss_name=current_loss_name,
            lp_p=args.lp_p,
            smooth_l1_beta=SMOOTH_L1_BETA,
        )
        train_metric_history.append(train_loss)
        val_metric_history.append(val_loss)
        scheduler.step()

        avg_delta: float | None = None
        if not loss_switched and epoch >= args.min_epochs_before_switch:
            history = (
                val_metric_history if LOSS_SWITCH_METRIC == "val" else train_metric_history
            )
            do_switch, avg_delta = should_switch_loss(
                history,
                window=args.switch_window,
                threshold=args.switch_threshold,
            )
            if do_switch:
                current_loss_name = args.loss_end
                loss_switched = True
                print(
                    f"Switching loss at epoch {epoch}: "
                    f"{LOSS_START} -> {args.loss_end} "
                    f"(avg |delta {LOSS_SWITCH_METRIC}| over last {args.switch_window} epochs = "
                    f"{avg_delta:.6f})"
                )

        tag = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "normalizer": {
                        "state_mean": normalizer.state_mean,
                        "state_std": normalizer.state_std,
                        "action_mean": normalizer.action_mean,
                        "action_std": normalizer.action_std,
                    },
                    "chunk_size": args.chunk_size,
                    "policy_type": args.policy,
                    "state_keys": effective_state_keys,
                    "action_keys": effective_action_keys,
                    "state_dim": int(states.shape[1]),
                    "action_dim": int(actions.shape[1]),
                    "d_model": args.d_model,
                    "depth": args.depth,
                    "loss_start": LOSS_START,
                    "loss_end": args.loss_end,
                    "loss_schedule": args.loss_schedule,
                    "loss_used": current_loss_name,
                    "lp_p": args.lp_p,
                    "smooth_l1_beta": SMOOTH_L1_BETA,
                    "val_loss": val_loss,
                },
                save_path,
            )
            tag = " ✓ saved"

        delta_tag = ""
        if avg_delta is not None:
            delta_tag = f" | avg Δ{LOSS_SWITCH_METRIC} {avg_delta:.6f}"
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss={current_loss_name} | train {train_loss:.6f} | "
            f"val {val_loss:.6f}{delta_tag}{tag}"
        )

        early_stop_history = (
            val_metric_history if LOSS_SWITCH_METRIC == "val" else train_metric_history
        )
        should_stop, stop_avg_delta = should_switch_loss(
            early_stop_history,
            window=EARLY_STOP_WINDOW,
            threshold=EARLY_STOP_THRESHOLD,
        )
        if should_stop:
            print(
                f"Early stopping at epoch {epoch}: "
                f"avg |delta {LOSS_SWITCH_METRIC}| over last {EARLY_STOP_WINDOW} epochs = "
                f"{stop_avg_delta:.6f}"
            )
            break

    print(f"\nBest val loss: {best_val:.6f}")
    print(f"Checkpoint: {save_path}")


if __name__ == "__main__":
    main()
