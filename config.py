import argparse
import yaml
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
 
 
@dataclass
class Config:
    # Paths
    data_root: str = "dataset"
    output_dir: str = "outputs"
 
    # Dataset
    target_shape: tuple = (128, 128, 128)
    heatmap_size: tuple = (64, 64, 64)
    train_val_split: float = 0.8
    num_workers: int = 4
    pin_memory: bool = True
 
    # Model
    num_organs: int = 121
    freeze_blocks: int = 0
    dropout_prob: float = 0.2
    backbone: str = "gnn"

    # Point Cloud & GNN Parameters
    num_points: int = 4096
    pointnet_feat_dim: int = 1024
    sex_emb_dim: int = 128
    gnn_hidden_dim: int = 256
    gnn_layers: int = 4
 
    # Training
    epochs: int = 150
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    grad_accum_steps: int = 4
    mixed_precision: bool = True
    seed: int = 42
 
    # Loss weights
    heatmap_weight: float = 1.0
    centroid_weight: float = 2.0
 
    # Scheduler
    scheduler: str = "cosine"       # cosine | plateau | step
    warmup_epochs: int = 5
    lr_min: float = 1e-6
    patience: int = 20              # for plateau / early stopping
 
    # Checkpointing
    save_every: int = 10            # save checkpoint_epoch_X.pth every N epochs
 
    # Logging
    log_tensorboard: bool = True
    log_csv: bool = True
 
    # Derived (set automatically)
    checkpoint_dir: str = ""
    log_dir: str = ""
    tensorboard_dir: str = ""
    predictions_dir: str = ""
 
    def __post_init__(self):
        self.checkpoint_dir = str(Path(self.output_dir) / "checkpoints")
        self.log_dir = str(Path(self.output_dir) / "logs")
        self.tensorboard_dir = str(Path(self.output_dir) / "tensorboard")
        self.predictions_dir = str(Path(self.output_dir) / "predictions")
 
    def make_dirs(self):
        for d in [self.checkpoint_dir, self.log_dir, self.tensorboard_dir, self.predictions_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)
 
    def save(self, path: str):
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)
 
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f)
        c = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        c.__post_init__()
        return c
 
    @classmethod
    def from_args(cls) -> "Config":
        parser = argparse.ArgumentParser(description="CT Organ Localization Training")
        parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
        parser.add_argument("--data_root", type=str)
        parser.add_argument("--output_dir", type=str)
        parser.add_argument("--epochs", type=int)
        parser.add_argument("--batch_size", type=int)
        parser.add_argument("--learning_rate", type=float)
        parser.add_argument("--num_workers", type=int)
        parser.add_argument("--grad_accum_steps", type=int)
        parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
        parser.add_argument("--seed", type=int)
        parser.add_argument("--freeze_blocks", type=int)
        args = parser.parse_args()
 
        if args.config:
            cfg = cls.from_yaml(args.config)
        else:
            cfg = cls()
 
        # CLI overrides
        if args.data_root:      cfg.data_root = args.data_root
        if args.output_dir:     cfg.output_dir = args.output_dir
        if args.epochs:         cfg.epochs = args.epochs
        if args.batch_size:     cfg.batch_size = args.batch_size
        if args.learning_rate:  cfg.learning_rate = args.learning_rate
        if args.num_workers:    cfg.num_workers = args.num_workers
        if args.grad_accum_steps: cfg.grad_accum_steps = args.grad_accum_steps
        if args.no_amp:         cfg.mixed_precision = False
        if args.seed:           cfg.seed = args.seed
        if args.freeze_blocks is not None: cfg.freeze_blocks = args.freeze_blocks
 
        cfg.__post_init__()
        return cfg