BATCH_SIZE=1
NUM_EPOCHS=150
DEVICE='cuda:0'
DATA_PATH="/media/student1/RemovableVolume/calgary/" 
RESUME='False'
SAMPLE_RATE=0.01


#<<ACC_FACTOR_5x
ACC_FACTOR=5
ACC='5x'
CHECKPOINT_PRETRAINED='/media/student1/NewVolume/MR_Reconstruction/experiments/challenge_calgary/acc_5x/best_unet_vs_model.pt'
CHECKPOINT='/media/student1/NewVolume/MR_Reconstruction/experiments/challenge_calgary/acc_5x/dun_model.pt'
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/challenge_calgary/acc_'${ACC}'/'
python train_dun.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR}  --acceleration ${ACC_FACTOR} --sample-rate ${SAMPLE_RATE}  --resume ${RESUME} --checkpoint ${CHECKPOINT} --pretrained ${CHECKPOINT_PRETRAINED}
#ACC_FACTOR_5x


<<ACC_FACTOR_10x
ACC_FACTOR=10
ACC='10x'
CHECKPOINT_PRETRAINED='/media/student1/NewVolume/MR_Reconstruction/experiments/challenge_calgary/acc_10x/best_unet_vs_model.pt'
CHECKPOINT='/media/student1/NewVolume/MR_Reconstruction/experiments/challenge_calgary/acc_10x/dun_model.pt'
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/challenge_calgary/acc_'${ACC}'/'
python train_dun.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR}  --acceleration ${ACC_FACTOR} --sample-rate ${SAMPLE_RATE}  --resume ${RESUME} --checkpoint ${CHECKPOINT} --pretrained ${CHECKPOINT_PRETRAINED}
ACC_FACTOR_10x


