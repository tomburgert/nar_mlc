from dataclasses import dataclass
from typing import List, Union, Optional
from dataclasses import field


@dataclass
class GeneralParameter:
    seed: int = 1
    cuda_no: str = '2'
    max_epochs: int = 20
    model: str = ''
    addn: float = 0.0
    subn: float = 0.0
    num_workers: int = 4
    batch_size: int = 128
    dataset: str = 'BigEarthNet'
    pin_memory: bool = True
    eval_on_test: bool = True
    slurm_bypass: bool = False
    mixed_noise: bool = True


@dataclass
class DataAugmentationParameter:
    train_augmentation: str = 'none'
    test_augmentation: str = 'none'
    p: float = 0.5
    p_list: Optional[List[float]] = None
    magnitude: Optional[int] = None
    brightness_limit : Optional[float] = None
    contrast_limit : Optional[float] = None
    max_edge: Optional[float] = None
    min_edge: Optional[float] = None
    sigma: Optional[List[float]] = field(default_factory=list)
    var_max: Optional[int] = None
    per_channel: Optional[bool] = None
    grid_size: Optional[int] = 3
    max_holes: Optional[int] = None
    min_holes: Optional[int] = None
    dropout_prob: Optional[float] = None
    shift: Optional[int] = None
    num_bits: Optional[int] = None
    randaug_op_names: Optional[List[str]] = field(default_factory=list)
    randaug_magnitude: Optional[int] = None
    resize_size: Optional[List[int]] = field(default_factory=list)
    scale: Optional[List[float]] = field(default_factory=list)
    cond_rrc_min_scale: Optional[float] = 1.0
    ratio : Optional[List[float]] = field(default_factory=list)
    angle: Optional[int] = None
    alpha: Optional[float] = None
    shear_x: Optional[int] = None
    shear_y: Optional[int] = None
    shear: Optional[int] = None
    threshold: Optional[int] = None
    pct_x: Optional[float] = None
    pct_y: Optional[float] = None
    pct: Optional[float] = None


@dataclass
class NoiseRobustLearningParameter:
    use_nrl: bool = False
    use_elr: bool = False
    use_sat: bool = False
    use_asl: bool = False
    use_ral: bool = False
    use_oracle: bool = False
    use_balancemix: bool = False
    co_teaching: bool = False
    nrl_type: str = 'threshold'             # or [threshold, oracle, topk]
    class_min_add: float = 0.2              # minimum class size (based on noisy labels) to run NRL
    class_min_sub: float = 0.0
    warm_start: int = 0                     # first epoch after which we perform NRL
    p1_zero: float = 1.0                    # flip 0
    p2_zero: float = 1.0                    # ignore 0
    p1_one: Union[float, str, int] = 0      # flip 1
    p2_one: Union[float, str, int] = 'cw'   # ignore 1 
    oracle_type_noisy_one: str = 'none'     # [none, flip, ignore]
    oracle_type_clean_one: str = 'none'
    oracle_type_noisy_zero: str = 'none'
    oracle_type_clean_zero: str = 'none'
    use_ema: bool = False
    ema_mode: str = 'weights'
    ema_decay: float = 0.99
    ema_warmup_epoch: int = 0
    use_strong_weak: bool = False
    use_dynamic_thresholds: bool = False
    slope: float = 0.2
    intersect: float = 0.5
    elr_lam: float = 3.0
    elr_beta: float = 0.7
    sat_momentum: float = 0.9
    sat_warmup_epochs: int = 5
    asl_gamma_neg: float = 4.0
    asl_gamma_pos: float = 1.0
    asl_clip: float = 0.05
    ral_lamb: float = 1.5
    ral_epsilon_neg: float = 0.0
    ral_epsilon_pos: float = 1.0
    ral_epsilon_pos_pow: float = -2.5
    bmix_alpha: float = 0.4


@dataclass
class Dataset:
    intersection_8country: bool = True
    all_percentiles: bool = False
    global_pctl: bool = False
    lmdb_path: str = ''
    labels_path: str = ''
    train_csv: str = ''
    val_csv: str = ''
    test_csv: str = ''
    num_classes: Optional[int] = None
    num_channels: Optional[int] = None


@dataclass
class Network:
    name: str = 'resnet18'
    pretrained: bool = False


@dataclass
class Optimizer:
    min_lr: float = 0.002
    momentum: float = 0.9
    weight_decay: float = 0.005
    gamma: float = 0.1
    milestones: List[int] = field(default_factory=list)
    loss_pos_weight: bool = False
    use_ema: bool = False


@dataclass
class Logging:
    exp_dir: str = ''
    ckpt_path: Optional[str] = None
    save_checkpoint : bool = False


@dataclass
class Tracking:
    should_track_train_probs: bool = False
    should_track_val_probs: bool = False
    should_track_labels: bool = False
    should_track_weights: bool = False
    should_track_noise_statistics: bool = False
    log_subtractive_flip_statistics: bool = False


@dataclass
class NoiseMLCConfig:
    params: GeneralParameter = field(default_factory=GeneralParameter)
    dataset: Dataset = field(default_factory=Dataset)
    model: Network = field(default_factory=Network)
    nrl: NoiseRobustLearningParameter = field(default_factory=NoiseRobustLearningParameter)
    dataaug: DataAugmentationParameter = field(default_factory=DataAugmentationParameter)
    optim: Optimizer = field(default_factory=Optimizer)
    logging: Logging = field(default_factory=Logging)
    tracking: Tracking = field(default_factory=Tracking)
