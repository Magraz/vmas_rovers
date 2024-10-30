TRIAL_ID=${1}
python3 /home/magraz/vmas_rovers/src/run_ccea.py --batch mlp_fc  --name d_mlp --trial_id ${TRIAL_ID}
echo "Finished Trial #${TRIAL_ID}"