"""Knowledge distillation for BitLinear model recovery.

Trains a BitLinear-converted student model to recover quality lost during
ternary quantization by distilling from the original full-precision teacher.

The combined loss is::

    L = α · CE(student, labels) + β · KL(student/T, teacher/T) · T²

with temperature annealing from T_start → T_end over training.

Usage::

    from src.distill import DistillationTrainer, DistillationConfig

    trainer = DistillationTrainer(student, teacher, DistillationConfig())
    trainer.train(dataloader, num_steps=5000)
    trainer.plot_loss_curves()
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DistillationConfig:
    """Full configuration for distillation training.

    Call :meth:`auto_configure` after creation to set hardware-optimal
    batch sizes automatically.
    """

    # --- Model ---
    student_name: str = "Qwen/Qwen3.5-0.8B-Base"
    teacher_name: str = "Qwen/Qwen3.5-0.8B-Base"

    # --- Training ---
    num_steps: int = 5_000
    batch_size: int = 2
    seq_length: int = 1024
    gradient_accumulation: int = 8
    learning_rate: float = 2e-4
    min_lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_grad_norm: float = 1.0

    # --- Distillation ---
    alpha_ce: float = 0.5
    beta_kd: float = 1.0
    temperature_start: float = 4.0
    temperature_end: float = 1.0
    temperature_schedule: str = "cosine"  # "cosine" or "linear"

    # --- Checkpointing ---
    checkpoint_every: int = 500
    checkpoint_dir: str = "./checkpoints"

    # --- Logging ---
    log_every: int = 10
    eval_every: int = 500
    eval_samples: int = 50

    # --- Hardware ---
    gradient_checkpointing: bool = True
    mixed_precision: bool = True  # BF16 for non-BitLinear modules

    # --- Quality gate ---
    quality_gate_factor: float = 2.0
    quality_gate_patience: int = 1000

    def auto_configure(self, vram_gb: float | None = None) -> None:
        """Auto-set batch size and teacher based on detected GPU VRAM.

        Args:
            vram_gb: GPU VRAM in GB.  If ``None``, auto-detect via torch.
        """
        if vram_gb is None:
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
            else:
                vram_gb = 0.0

        if vram_gb >= 40:  # A100
            self.batch_size = 8
            self.gradient_accumulation = 4
            self.teacher_name = "Qwen/Qwen3.5-2B-Base"
        elif vram_gb >= 24:  # L4 / 4090
            self.batch_size = 4
            self.gradient_accumulation = 8
        elif vram_gb >= 15:  # T4 / 3060
            self.batch_size = 2
            self.gradient_accumulation = 16
        else:  # CPU or very small GPU
            self.batch_size = 1
            self.gradient_accumulation = 32

        logger.info(
            "Auto-configured for %.1f GB VRAM: batch=%d, grad_accum=%d, "
            "teacher=%s",
            vram_gb,
            self.batch_size,
            self.gradient_accumulation,
            self.teacher_name,
        )

    @property
    def effective_batch_tokens(self) -> int:
        """Tokens per optimizer step."""
        return self.batch_size * self.seq_length * self.gradient_accumulation


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class DistillationTrainer:
    """Knowledge distillation trainer with temperature annealing.

    Args:
        student: BitLinear-converted model (trainable).
        teacher: Original full-precision model (frozen).
        config: Training configuration.
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        config: DistillationConfig | None = None,
    ) -> None:
        self.student = student
        self.teacher = teacher
        self.config = config or DistillationConfig()

        # Freeze teacher.
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Enable gradient checkpointing on student.
        if self.config.gradient_checkpointing:
            if hasattr(self.student, "gradient_checkpointing_enable"):
                self.student.gradient_checkpointing_enable()

        # Optimizer.
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95),
        )

        # Cosine LR schedule with warmup.
        def lr_lambda(step: int) -> float:
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)
            progress = (step - self.config.warmup_steps) / max(
                1, self.config.num_steps - self.config.warmup_steps
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            min_ratio = self.config.min_lr / self.config.learning_rate
            return min_ratio + (1.0 - min_ratio) * cosine

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )

        # Tracking.
        self.global_step = 0
        self.losses: list[float] = []
        self.ce_losses: list[float] = []
        self.kd_losses: list[float] = []
        self.lrs: list[float] = []
        self.eval_results: list[dict] = []

    # ------------------------------------------------------------------
    # Temperature annealing
    # ------------------------------------------------------------------

    def get_temperature(self) -> float:
        """Current distillation temperature based on training progress."""
        progress = min(self.global_step / max(1, self.config.num_steps), 1.0)
        t_start = self.config.temperature_start
        t_end = self.config.temperature_end

        if self.config.temperature_schedule == "cosine":
            return t_end + (t_start - t_end) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
        # Linear fallback.
        return t_start - (t_start - t_end) * progress

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float]:
        """Compute combined CE + KL distillation loss.

        Returns:
            Tuple of (total_loss, ce_loss_scalar, kd_loss_scalar).
        """
        T = self.get_temperature()

        # Shift for causal LM (predict next token).
        s_logits = student_logits[..., :-1, :].contiguous()
        t_logits = teacher_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Cross-entropy on hard labels.
        loss_ce = F.cross_entropy(
            s_logits.view(-1, s_logits.size(-1)),
            shift_labels.view(-1),
        )

        # KL divergence on soft labels (temperature-scaled).
        loss_kd = F.kl_div(
            F.log_softmax(s_logits / T, dim=-1),
            F.softmax(t_logits / T, dim=-1),
            reduction="batchmean",
        ) * (T * T)

        total = self.config.alpha_ce * loss_ce + self.config.beta_kd * loss_kd
        return total, loss_ce.item(), loss_kd.item()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        dataloader: Any,
        num_steps: int | None = None,
    ) -> None:
        """Run the distillation training loop.

        Args:
            dataloader: Iterable yielding dicts with ``"input_ids"`` tensors.
            num_steps: Override ``config.num_steps``.
        """
        num_steps = num_steps or self.config.num_steps
        cfg = self.config
        device = next(self.student.parameters()).device

        self.student.train()
        self.optimizer.zero_grad()

        data_iter = iter(dataloader)
        accum_loss = 0.0
        accum_ce = 0.0
        accum_kd = 0.0
        step_start = time.time()
        start_step = self.global_step

        for step_i in range(num_steps):
            # --- Get batch ---
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)

            # --- Teacher forward (no grad) ---
            with torch.no_grad():
                teacher_out = self.teacher(input_ids)

            # --- Student forward ---
            student_out = self.student(input_ids)

            # --- Loss ---
            loss, ce_val, kd_val = self.compute_loss(
                student_out.logits, teacher_out.logits, input_ids
            )
            scaled_loss = loss / cfg.gradient_accumulation
            scaled_loss.backward()

            accum_loss += loss.item()
            accum_ce += ce_val
            accum_kd += kd_val

            # --- Optimizer step ---
            if (step_i + 1) % cfg.gradient_accumulation == 0:
                nn.utils.clip_grad_norm_(
                    self.student.parameters(), cfg.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                # Record.
                n = cfg.gradient_accumulation
                avg_loss = accum_loss / n
                avg_ce = accum_ce / n
                avg_kd = accum_kd / n
                self.losses.append(avg_loss)
                self.ce_losses.append(avg_ce)
                self.kd_losses.append(avg_kd)
                self.lrs.append(self.scheduler.get_last_lr()[0])

                accum_loss = 0.0
                accum_ce = 0.0
                accum_kd = 0.0

                # --- Logging ---
                if self.global_step % cfg.log_every == 0:
                    elapsed = time.time() - step_start
                    tok_per_sec = (
                        cfg.log_every
                        * cfg.effective_batch_tokens
                        / elapsed
                    )
                    T = self.get_temperature()
                    lr = self.scheduler.get_last_lr()[0]
                    print(
                        f"Step {self.global_step:>6d} | "
                        f"Loss {avg_loss:.4f} | "
                        f"CE {avg_ce:.4f} | "
                        f"KD {avg_kd:.4f} | "
                        f"T {T:.2f} | "
                        f"LR {lr:.2e} | "
                        f"{tok_per_sec:.0f} tok/s"
                    )
                    step_start = time.time()

                # --- Checkpoint ---
                if self.global_step % cfg.checkpoint_every == 0:
                    self.save_checkpoint()

        print(
            f"\nTraining complete. "
            f"{self.global_step - start_step} optimizer steps."
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        dataloader: Any,
        num_samples: int | None = None,
    ) -> dict[str, float]:
        """Evaluate model perplexity on a dataset.

        Returns:
            Dict with ``"loss"`` and ``"perplexity"`` keys.
        """
        num_samples = num_samples or self.config.eval_samples
        device = next(self.student.parameters()).device

        self.student.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_samples:
                    break
                input_ids = batch["input_ids"].to(device)
                outputs = self.student(input_ids)
                s_logits = outputs.logits[..., :-1, :].contiguous()
                s_labels = input_ids[..., 1:].contiguous()
                loss = F.cross_entropy(
                    s_logits.view(-1, s_logits.size(-1)),
                    s_labels.view(-1),
                    reduction="sum",
                )
                total_loss += loss.item()
                total_tokens += s_labels.numel()

        self.student.train()

        avg_loss = total_loss / max(1, total_tokens)
        ppl = math.exp(min(avg_loss, 100))  # clamp to avoid overflow
        result = {"loss": avg_loss, "perplexity": ppl}
        self.eval_results.append(
            {"step": self.global_step, **result}
        )
        return result

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str | None = None) -> str:
        """Save a training checkpoint.

        Returns:
            Path to the saved checkpoint file.
        """
        ckpt_dir = self.config.checkpoint_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        path = path or os.path.join(
            ckpt_dir, f"checkpoint-{self.global_step}.pt"
        )

        torch.save(
            {
                "global_step": self.global_step,
                "student_state_dict": self.student.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "losses": self.losses,
                "ce_losses": self.ce_losses,
                "kd_losses": self.kd_losses,
                "lrs": self.lrs,
                "eval_results": self.eval_results,
                "config": self.config,
            },
            path,
        )
        print(f"  💾 Checkpoint saved → {path}")
        return path

    def load_checkpoint(self, path: str) -> None:
        """Resume training from a checkpoint.

        Args:
            path: Path to a ``.pt`` checkpoint file.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.student.load_state_dict(ckpt["student_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.global_step = ckpt["global_step"]
        self.losses = ckpt.get("losses", [])
        self.ce_losses = ckpt.get("ce_losses", [])
        self.kd_losses = ckpt.get("kd_losses", [])
        self.lrs = ckpt.get("lrs", [])
        self.eval_results = ckpt.get("eval_results", [])
        print(f"  ✅ Resumed from step {self.global_step}")

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot_loss_curves(self) -> None:
        """Plot training loss curves (inline for notebooks)."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Total loss.
        axes[0].plot(self.losses, alpha=0.3, color="steelblue")
        if len(self.losses) > 20:
            # Smoothed.
            window = min(50, len(self.losses) // 4)
            smoothed = _moving_avg(self.losses, window)
            axes[0].plot(smoothed, color="steelblue", linewidth=2)
        axes[0].set_title("Total Loss")
        axes[0].set_xlabel("Optimizer Step")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, alpha=0.3)

        # CE vs KD.
        axes[1].plot(self.ce_losses, alpha=0.5, label="CE (hard labels)")
        axes[1].plot(self.kd_losses, alpha=0.5, label="KD (soft labels)")
        axes[1].set_title("Loss Components")
        axes[1].set_xlabel("Optimizer Step")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Learning rate.
        axes[2].plot(self.lrs, color="coral")
        axes[2].set_title("Learning Rate")
        axes[2].set_xlabel("Optimizer Step")
        axes[2].set_ylabel("LR")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def create_dataloader(
    config: DistillationConfig,
    tokenizer: Any,
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    dataset_config: str = "sample-10BT",
    split: str = "train",
) -> Any:
    """Create a streaming dataloader from FineWeb-Edu.

    Returns a standard PyTorch DataLoader wrapping an HF streaming dataset.
    """
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    dataset = load_dataset(
        dataset_name,
        name=dataset_config,
        split=split,
        streaming=True,
        trust_remote_code=True,
    )

    def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
        texts = [item["text"] for item in batch]
        encodings = tokenizer(
            texts,
            truncation=True,
            max_length=config.seq_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {"input_ids": encodings["input_ids"]}

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        num_workers=0,  # streaming datasets use main process
    )


def compute_model_perplexity(
    model: nn.Module,
    dataloader: Any,
    num_samples: int = 50,
    device: str | torch.device = "cuda",
) -> dict[str, float]:
    """Standalone perplexity evaluation (no trainer needed).

    Args:
        model: Any causal LM.
        dataloader: Yields dicts with ``"input_ids"``.
        num_samples: Number of batches to evaluate.
        device: Device to run on.

    Returns:
        Dict with ``"loss"`` and ``"perplexity"``.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
            input_ids = batch["input_ids"].to(device)
            outputs = model(input_ids)
            logits = outputs.logits[..., :-1, :].contiguous()
            labels = input_ids[..., 1:].contiguous()
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += labels.numel()

    avg_loss = total_loss / max(1, total_tokens)
    return {
        "loss": avg_loss,
        "perplexity": math.exp(min(avg_loss, 100)),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _moving_avg(data: list[float], window: int) -> list[float]:
    """Simple moving average for loss smoothing."""
    if len(data) < window:
        return data
    result = []
    running = sum(data[:window])
    for i in range(window, len(data)):
        result.append(running / window)
        running += data[i] - data[i - window]
    result.append(running / window)
    return result
