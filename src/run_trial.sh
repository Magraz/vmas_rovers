TRIAL_ID=${1}
python3 /home/magraz/vmas_rovers/src/run_ccea.py --experiment_type sanity  --poi_type static --model mlp --trial_id ${TRIAL_ID}
echo "Finished Trial #${TRIAL_ID}"