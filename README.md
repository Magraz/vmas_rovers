### To Install

Run the following commands inside `/vmas_rovers` after cloning:
1. `git submodule init`
2. `git submodule update`
3. `cd VectorizedMultiAgentSimulator`
4. `pip install -e .`
5. `cd vmas_rovers`
6. `pip install -e .`
7. `pip install -r requirements.txt`



### To Run Experiment

Run the following command for running experiments in different modalities:

- `python3 src/run_ccea.py --batch=static_spread --name=g_mlp --trial_id=0`

Run N number of trials in parallel (Requires GNU Parallel Package)

- `parallel bash src/run_trial.sh order d_gru ::: $(seq 0 N)`



