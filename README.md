### To Install

Run the following commands inside repo after cloning:
1. `git submodule init`
2. `git submodule update`
3. `pip install -r requirements.txt`
4. `cd VectorizedMultiAgentSimulator`
5. `pip install -e .`


### To Run Experiment

Run the following command for running experiments with all default values:
- `python3 src/run_experiment.py`

Run the following command for running experiments in different modalities:

- `python3 src/run_ccea.py --experiment_type=standard --poi_type static --model mlp --trial_id=<trial_id>`

Run N number of trials in parallel (Requires GNU Parallel Package)

- `parallel ./src/run_trial.sh ::: $(seq 0 N)`



