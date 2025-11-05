"""
Configuration classes for MCL (Multiple Choice Learning).
"""

from dataclasses import dataclass, field


@dataclass
class MCLConfig:
    """
    Configuration for MCL (Multiple Choice Learning) functionality.
    Used alongside standard PEFT LoraConfig.
    """

    # MCL parameters
    num_hyps: int = field(default=1, metadata={"help": "Number of hypotheses for MCL"})

    # WTA (Winner-Take-All) parameters
    wta_training_mode: str = field(
        default="wta",
        metadata={"help": "WTA training mode: 'wta', 'relaxed-wta', 'annealed-wta', etc."},
    )
    wta_params_epsilon: float = field(
        default=0.0, metadata={"help": "Epsilon parameter for relaxed WTA"}
    )
    wta_params_ini_temp: float = field(
        default=1.0, metadata={"help": "Initial temperature for annealed WTA"}
    )
    wta_params_fin_temp: float = field(
        default=0.01, metadata={"help": "Final temperature for annealed WTA"}
    )
    wta_params_decay_rate: float = field(default=0.999, metadata={"help": "Temperature decay rate"})
    wta_params_schedule_mode: str = field(
        default="global_step", metadata={"help": "Temperature schedule mode"}
    )

    def __post_init__(self):
        # Validate MCL parameters
        if self.num_hyps < 1:
            raise ValueError("num_hyps must be >= 1")
