from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # Environment
    num_good: int = 2           # cooperative good agents
    num_adversaries: int = 1    # #uses custom simple adversary
    max_cycles: int = 25

    # Training
    num_episodes: int = 10_000
    batch_size: int = 1024
    buffer_size: int = 1_000_000
    warmup_steps: int = 2_000

    # Optimiser
    lr_actor: float = 1e-2
    lr_critic: float = 1e-2

    # RL
    gamma: float = 0.95
    tau: float = 0.01

    # Exploration noise (Gaussian, linearly decayed)
    noise_std_start: float = 0.3
    noise_std_end: float = 0.01
    noise_decay_steps: int = 100_000

    # Network
    hidden_dim: int = 64

    # Logging / Evaluation
    log_interval: int = 200
    eval_episodes: int = 100
    rolling_window: int = 100

    # Reproducibility
    seed: int = 42
    device: str = "cpu"

    # Multi-config scaling (used by main.py)
    use_multi_adversary: bool = False
    multi_n_good_list: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    multi_num_episodes: int = 3_000

    @property
    def run_tag(self) -> str:
        return f"good{self.num_good}_adv{self.num_adversaries}"
