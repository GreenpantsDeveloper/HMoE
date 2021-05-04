import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback

from cyclical_learning_rate import CyclicLR

from config import NUM_EPOCHS, EARLY_STOPPING_PATIENCE


class WorkingModelCheckpoint(Callback):
    def __init__(self, model=None, save_file=None, **kwargs):
        super(WorkingModelCheckpoint, self).__init__(**kwargs)
        self.model = model
        self.save_file = "models/model.hdf5" if save_file is None else save_file

    def on_train_begin(self, logs={}):
        self.minloss = float('inf')
        return
 
    # Save model if the validation loss is at its lowest point
    def on_epoch_end(self, epoch, logs={}):
        if self.minloss > logs.get('val_loss'):
            self.minloss = logs.get('val_loss')
            self.model.save(self.save_file)
            #print('\t\tModel saved!')
        return

def get_cb_early_stopping(patience):
    return EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0)

def get_cb_checkpoint(model, save_file):
    return WorkingModelCheckpoint(model, save_file=save_file)

def get_cb_cyclic_lr():
    return CyclicLR(
        base_lr=0.001,
        max_lr=0.006,
        step_size=2000.,
        mode='triangular',
        gamma=1.,
        scale_fn=None,
        scale_mode='cycle'
    )
