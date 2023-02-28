import torch



DEVICE = torch.device('cpu')
BATCH_SIZE = 128
TEST_BATCH_SIZE = 1000
EPOCHS = 50
LR = 0.01
LR_decay = 0.001
MOMENTUM = 0.5
SEED = 1
LOG_INTERVAL = 80
SAVE_INTERVAL = 100
TRAINING_CONDITION = 'stn'
BOUNDE_GRID_TYPE = 'bounded_stn'
ANGLE = 60
SPAN_RANGE = 0.9
GRID_SIZE= 4
IMAGE_HEIGHT, IMAGE_WIDTH = [256, 256]
NUM_CLASSIFY = 24
