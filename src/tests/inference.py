import pickle
import sys

sys.path.insert(0, "./src")

from learning.ccea import CooperativeCoevolutionaryAlgorithm
from domain.create_env import create_env

experiment_type = "difference"
poi_type = "static"
model = "gru"
trial = 0
experiment_name = "_".join((experiment_type, poi_type, model))
checkpoint_path = f"./src/tests/checkpoint.pickle"
config_path = f"./src/experiments/yamls/{experiment_type}/{poi_type}_{model}.yaml"

best_team = None

with open(checkpoint_path, "rb") as handle:
    checkpoint = pickle.load(handle)
    best_team = checkpoint["best_team"]


ccea = CooperativeCoevolutionaryAlgorithm(config_path, experiment_name, trial)

eval_infos = ccea.evaluateTeams(
    create_env(config_dir=config_path, n_envs=1, device=ccea.device),
    [best_team],
    render=True,
    save_render=False,
)
