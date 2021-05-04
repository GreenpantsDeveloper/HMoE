import os, sys
import numpy as np

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Layer, Dense, Activation, LSTM, PReLU, add, Reshape, Lambda, Input, Concatenate
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K

from config import INPUT_SIZE, OUTPUT_SIZE, MODEL_DEPTH
from config import HMOE_KERNEL_INITIALIZER
from config import EXPERTS_LAYERS, EXPERTS_ACTIVATION
from config import MANAGERS_LAYERS, MANAGERS_ACTIVATION



# Returns Hierarchical Mixture of Experts (HMoE) of arbitrary depth (see `config.py`)
def build_HMoE():
    # Define the input tensor
    common_input = Input(shape=INPUT_SIZE)

    ### Allow building a single MLP at depth 0
    if MODEL_DEPTH == 0:
        # Single expert as the MLP
        return build_MLP(input_tensor=common_input, output_shape=OUTPUT_SIZE, layer_size=EXPERTS_LAYERS, kernel_initializer=HMOE_KERNEL_INITIALIZER, activation=EXPERTS_ACTIVATION)

    ### Build leaf nodes: Experts
    num_experts = 2**MODEL_DEPTH
    print("\nBuilding %d Experts" % num_experts)
    experts = [build_expert(common_input) for _ in range(num_experts)]

    ### Build non-leaf nodes: Managers & WeightedSum layers
    previous_layer_models = experts # `previous_layer_models` being the models in the hierarchical layer
    for current_depth in reversed(range(MODEL_DEPTH)): # at depth == 2, yields [1, 0]
        num_managers = 2**current_depth
        print("Building %d Manager(s)" % num_managers)
        merge_blocks = []

        ## Build Managers & WeightedSum outputs
        for i in range(num_managers):
            # Get previous layer's outputs
            expert1, expert2 = (previous_layer_models[2*i], previous_layer_models[2*i + 1])

            # Define manager's input
            manager_input = common_input

            # Build manager & merge block
            manager = build_manager(common_input)
            merge_block = Model(common_input, get_combined(expert1, expert2, manager))
            merge_blocks.append(merge_block)

        previous_layer_models = merge_blocks
    
    ### Finalize the model
    final_manager = previous_layer_models[0]
    print("The hierarchy is done!")
    return final_manager



# Initialize a single Expert
def build_expert(input_tensor):
    end_block = mlp(input_tensor, layer_size=EXPERTS_LAYERS, kernel_initializer=HMOE_KERNEL_INITIALIZER, activation=EXPERTS_ACTIVATION)
    return finish_model(input_tensor, end_block, OUTPUT_SIZE, HMOE_KERNEL_INITIALIZER)

# Initialize a single Manager
def build_manager(input_tensor):
    end_block = mlp(input_tensor, layer_size=MANAGERS_LAYERS, kernel_initializer=HMOE_KERNEL_INITIALIZER, activation=MANAGERS_ACTIVATION)
    end_block = Dense(2)(end_block)
    out = Activation('softmax')(end_block)
    return Model(input_tensor, out)

# Define the Manager's gating output from the experts
def get_combined(left_block, right_block, manager):
    return WeightedSum()(g=manager.output, o1=left_block.output, o2=right_block.output)


# Build MLP from parameters
def build_MLP(input_tensor, output_shape, layer_size, kernel_initializer, activation, input_shape=None):
    # Use 'input_shape' instead of 'input tensor' for a newly initialized MLP
    if input_shape is not None:
        input_tensor = Input(shape=input_shape)
    
    end_block = mlp(input_tensor, layer_size, kernel_initializer, activation)
    return finish_model(input_tensor, end_block, output_shape, kernel_initializer)

# Finish model with regression & activation layers
def finish_model(input_tensor, end_block, output_shape, kernel_initializer):
    out = regression_layer(end_block, output_shape, kernel_initializer)
    return Model(input_tensor, out)

# Regression layer
def regression_layer(end_block, output_shape, kernel_initializer ):
    out = Dense(output_shape, kernel_initializer=kernel_initializer, bias_initializer='zeros')(end_block)
    return Activation('linear')(out)

# Activation layer
def activation_layer(activation) :
    if (activation == 'PReLU'):
        return PReLU(alpha_initializer='zero', weights=None)
    return Activation(activation)

# Build MLP model
def mlp(input_tensor, layer_size, kernel_initializer, activation):
    layer = input_tensor
    for layer_size in layer_size:
        layer = Dense(layer_size, kernel_initializer=kernel_initializer, bias_initializer='zeros')(layer)
        layer = activation_layer(activation)(layer)
    return layer


# WeightedSum class for the Manager combining two outputs
class WeightedSum(Layer):
    def __init__(self, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)

    def call(self, g, o1, o2):
        return K.transpose(K.transpose(o1) * g[:, 0] + K.transpose(o2) * g[:, 1])

    def compute_output_shape(self, input_shape):
        return input_shape[0]
