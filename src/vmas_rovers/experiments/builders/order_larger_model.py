from learning.ccea.dataclasses import ExperimentConfig, PolicyConfig, CCEAConfig
from learning.ccea.types import (
    FitnessShapingEnum,
    FitnessCalculationEnum,
    SelectionEnum,
    PolicyEnum,
    InitializationEnum,
)

from dataclasses import asdict
import yaml
from pathlib import Path

BATCH = "order_larger_model"
EXPERIMENTS = ["d_gru", "d_mlp", "g_gru", "g_mlp"]

# DEFAULTS
N_STEPS = 80
N_GENS = 5000
SUBPOP_SIZE = 100
N_GENS_BETWEEN_SAVE = 20
GRU_NEURONS = 17
MLP_NEURONS = 32
OUTPUT_MULTIPLIER = 1.0
WEIGHT_INITIALIZATION = InitializationEnum.KAIMING

GRU_POLICY_CONFIG = PolicyConfig(
    type=PolicyEnum.GRU,
    weight_initialization=WEIGHT_INITIALIZATION,
    hidden_layers=[GRU_NEURONS],
    output_multiplier=OUTPUT_MULTIPLIER,
)

MLP_POLICY_CONFIG = PolicyConfig(
    type=PolicyEnum.MLP,
    weight_initialization=WEIGHT_INITIALIZATION,
    hidden_layers=[MLP_NEURONS, MLP_NEURONS],
    output_multiplier=OUTPUT_MULTIPLIER,
)

D_CCEA = CCEAConfig(
    n_steps=N_STEPS,
    n_gens=N_GENS,
    fitness_shaping=FitnessShapingEnum.D,
    selection=SelectionEnum.SOFTMAX,
    subpopulation_size=SUBPOP_SIZE,
    fitness_calculation=FitnessCalculationEnum.AGG,
    mutation={"mean": 0.0, "min_std_deviation": 0.05, "max_std_deviation": 0.5},
)

G_CCEA = CCEAConfig(
    n_steps=N_STEPS,
    n_gens=N_GENS,
    fitness_shaping=FitnessShapingEnum.D,
    selection=SelectionEnum.SOFTMAX,
    subpopulation_size=SUBPOP_SIZE,
    fitness_calculation=FitnessCalculationEnum.AGG,
    mutation={"mean": 0.0, "min_std_deviation": 0.05, "max_std_deviation": 0.5},
)

# EXPERIMENTS
d_gru = ExperimentConfig(
    use_teaming=False,
    use_fc=False,
    n_gens_between_save=N_GENS_BETWEEN_SAVE,
    policy_config=GRU_POLICY_CONFIG,
    ccea_config=D_CCEA,
)

d_mlp = ExperimentConfig(
    use_teaming=False,
    use_fc=False,
    n_gens_between_save=N_GENS_BETWEEN_SAVE,
    policy_config=MLP_POLICY_CONFIG,
    ccea_config=D_CCEA,
)

g_mlp = ExperimentConfig(
    use_teaming=False,
    use_fc=False,
    n_gens_between_save=N_GENS_BETWEEN_SAVE,
    policy_config=MLP_POLICY_CONFIG,
    ccea_config=G_CCEA,
)

g_gru = ExperimentConfig(
    use_teaming=False,
    use_fc=False,
    n_gens_between_save=N_GENS_BETWEEN_SAVE,
    policy_config=GRU_POLICY_CONFIG,
    ccea_config=G_CCEA,
)

experiments_dicts = {
    {"name": "d_gru", "config": asdict(d_gru)},
    {"name": "d_mlp", "config": asdict(d_mlp)},
    {"name": "g_gru", "config": asdict(g_gru)},
    {"name": "g_mlp", "config": asdict(g_mlp)},
}

dir_path = Path(__file__).parent / "yamls" / BATCH

for exp_dict in experiments_dicts:

    with open(dir_path / f"{exp_dict["name"]}.yaml", "w") as file:
        yaml.dump(exp_dict["config"], file)
