DEVICE='cuda:0'
DATA_PATH="/media/student1/RemovableVolume/calgary/"            
SAMPLE_RATE=1.0

# <<ACC_FACTOR_5x
# ACC_FACTOR=5
# ACC='5x'
# RECONS_PATH='/media/student1/RemovableVolume/calgary/Recons/acc_'${ACC}'/'
# DATA_PATH='/media/student1/RemovableVolume/calgary/Val/'

# python evaluate.py --acceleration ${ACC_FACTOR} --data-path ${DATA_PATH} --recons-path ${RECONS_PATH} 
# ACC_FACTOR_5x

# <<ACC_FACTOR_10x
ACC_FACTOR=10
ACC='10x'
RECONS_PATH='/media/student1/RemovableVolume/calgary/Recons/acc_'${ACC}'/'
DATA_PATH='/media/student1/RemovableVolume/calgary/Val/'

python evaluate.py --acceleration ${ACC_FACTOR} --data-path ${DATA_PATH} --recons-path ${RECONS_PATH} 
# ACC_FACTOR_10x