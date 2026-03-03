from dataclasses import dataclass


@dataclass
class TDNNConfig:
    tdnn_path: str
    embedding_size: int = 192
    input_dim: int = 80
    mid_channels: int = 512


@dataclass
class TrainConfig:
    num_classes: int
    log_dir: str
    ckpt_dir: str
    num_epochs: int = 20
    step_size_up: int = 1000
    batch_size: int = 128
    eval_freq: int = 100
    ckpt_freq: int = 500
    margin: float = 0.2
    scale: float = 30.0
    min_lr: float = 1e-8
    max_lr: float = 1e-3
    tdnn_weight_decay: float = 2e-5
    classifier_weight_decay: float = 2e-4


@dataclass
class DatasetConfig:
    root_dir: str
    train_split_ratio: float = 0.99
    n_mels: int = 80
    sample_rate: int = 16000
    window_length: int = int(0.025 * sample_rate)
    hop_length: int = int(0.010 * sample_rate)
    mfcc_select_length: int = 200
