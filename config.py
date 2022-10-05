#data
DATAPATH = 'data/fed'#TODO  /a2il/data/fed  '/data_local/xuangong/data' only for deepbull2
MOSAIC_KD_DATA="/home/menglong/workspace/code/referred/MosaicKD/data"  #TODO
N_PARTIES = 20  #TODO 20
INIT_EPOCHS = 500
LOCAL_CKPTPATH = 'ckpt/fed' #TODO
#local training
LR = 0.0025
LR_MIN = 0.001
BATCHSIZE = 256
NUM_WORKERS = 6

#fed
ROUNDS = 200
LOCAL_PERCENT = 1
OPTIMIZER = 'SGD'
FED_BATCHSIZE = 64
STU_LR = 5e-3 # 0.1
STU_LR_MIN = 1e-4 # 1e-5
STU_WD = 1e-4
CKPTPATH = './ckpt'
DIS_LR = 1e-3 # 1e-3
GEN_LR = 1e-3 # 1e-3

#generator
GEN_Z_DIM = 100

#print
PRINT_FREQ = 1