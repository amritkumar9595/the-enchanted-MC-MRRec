BATCH_SIZE=1
DEVICE='cuda:0'
DATA_PATH="/media/student1/RemovableVolume/calgary/"            
SAMPLE_RATE=1.0

# <<ACC_FACTOR_5x
# ACC_FACTOR=5
# ACC='5x'
# RECONS_PATH='/media/student1/RemovableVolume/calgary/Recons/acc_'${ACC}'/'
# MODEL_PATH="/media/student1/NewVolume/MR_Reconstruction/experiments/challenge_calgary/acc_5x/best_dun_model.pt"

# python valid.py --batch-size ${BATCH_SIZE}  --device ${DEVICE}  --acceleration ${ACC_FACTOR} --sample-rate ${SAMPLE_RATE}  --data-path ${DATA_PATH} --recons-path ${RECONS_PATH} --model-path ${MODEL_PATH} 
# ACC_FACTOR_5x

# <<ACC_FACTOR_10x
ACC_FACTOR=10
ACC='10x'
RECONS_PATH='/media/student1/RemovableVolume/calgary/Recons/acc_'${ACC}'/'
MODEL_PATH="/media/student1/NewVolume/MR_Reconstruction/experiments/challenge_calgary/acc_10x/best_dun_model.pt"

python valid.py --batch-size ${BATCH_SIZE}  --device ${DEVICE}  --acceleration ${ACC_FACTOR} --sample-rate ${SAMPLE_RATE}  --data-path ${DATA_PATH} --recons-path ${RECONS_PATH} --model-path ${MODEL_PATH} 
# ACC_FACTOR_5x