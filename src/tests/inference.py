import pickle
import sys

sys.path.insert(0, "./src")

from learning.ccea import CooperativeCoevolutionaryAlgorithm

experiment_type = "standard"
poi_type = "static"
model = "mlp"
trial = 1
experiment_name = "_".join((experiment_type, poi_type, model))
checkpoint_path = (
    "./src/experiments/results/standard_static_mlp/trial_1_static_mlp/checkpoint.pickle"
)
config_path = "./src/experiments/yamls/standard/static_mlp.yaml"

best_team = None

with open(checkpoint_path, "rb") as handle:
    checkpoint = pickle.load(handle)
    best_team = checkpoint["best_team"]


ccea = CooperativeCoevolutionaryAlgorithm(config_path, experiment_name, trial, False)

eval_infos = ccea.evaluateTeams(ccea.create_env(1), [best_team], render=True)
