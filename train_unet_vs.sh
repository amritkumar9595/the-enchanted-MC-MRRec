BATCH_SIZE=1
NUM_EPOCHS=150
DEVICE='cuda:0'
RESUME='False'
DATA_PATH="/media/student1/RemovableVolume/calgary/"            
SAMPLE_RATE=0.02


#<<ACC_FACTOR_5x
ACC_FACTOR=5
ACC='5x'
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/challenge_calgary/acc_'${ACC}'/'
CHECKPOINT="/media/student1/NewVolume/MR_Reconstruction/experiments/challenge_calgary/acc_5x/unet_vs_model.pt"

python train_unet_vs.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR}  --acceleration ${ACC_FACTOR} --sample-rate ${SAMPLE_RATE}  --data-path ${DATA_PATH} --resume ${RESUME} --checkpoint ${CHECKPOINT}
#ACC_FACTOR_5x



<<ACC_FACTOR_10x
ACC_FACTOR=10
ACC='10x'
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/challenge_calgary/acc_'${ACC}'/'
CHECKPOINT="/media/student1/NewVolume/MR_Reconstruction/experiments/challenge_calgary/acc_10x/unet_vs_model.pt"

python train_unet_vs.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR}  --acceleration ${ACC_FACTOR} --sample-rate ${SAMPLE_RATE}  --data-path ${DATA_PATH} --resume ${RESUME} --checkpoint ${CHECKPOINT}
ACC_FACTOR_10x



