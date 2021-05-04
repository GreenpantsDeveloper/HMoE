##### All adjustable parameters will be found here #####

### Network Validity Testing Params
INPUT_SIZE                  = 28*28
OUTPUT_SIZE                 = 10


### Model definitions
MODEL_DEPTH                 = 3                         # 0 for a basic MLP; 1+ for an HMoE model
MODEL_PATH                  = 'models/'
MODEL_NAME                  = 'test'


### Training Params
NUM_EPOCHS                  = 20
BATCH_SIZE                  = 2**9
OPTIMIZER                   = 'adam'
LOSS                        = 'mse'
EARLY_STOPPING_PATIENCE     = 50


# HMoE Params
HMOE_KERNEL_INITIALIZER     = 'he_normal'

# Complexity of Experts
EXPERTS_LAYERS              = [30, 30, 30]
EXPERTS_ACTIVATION          = 'PReLU'

# Complexity of Managers
MANAGERS_LAYERS             = [30, 30, 30]
MANAGERS_ACTIVATION         = 'PReLU'
