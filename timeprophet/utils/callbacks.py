import numpy as np
from pytorch_lightning.callbacks import Callback, EarlyStopping, LearningRateMonitor


class AdjustLearningRate(Callback):

    def __init__(self, base_lr: float, verbose: bool = False):
        self.base_lr = base_lr
        self.verbose = verbose

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        lr = self.base_lr * (0.95**((epoch) // 1))
        for param_group in trainer.optimizers[0].param_groups:
            param_group['lr'] = lr
        if self.verbose:
            print(f"Learning rate adjusted to {lr}")


class TrainEarlyStopping(EarlyStopping):

    def on_validation_end(self, trainer, pl_module):
        # override this to disable early stopping at the end of val loop
        pass

    def on_train_end(self, trainer, pl_module):
        # instead, do it at the end of training loop
        self._run_early_stopping_check(trainer)


class TemperatureScaling(Callback):

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.min = 1

    def on_train_epoch_start(self, trainer, pl_module):
        t_max = 30
        t_min = self.min
        max_epoch = 10
        epoch = trainer.current_epoch
        if epoch > max_epoch:
            temperature = t_min
        else:
            temperature = t_max - (t_max - t_min) * epoch / max_epoch

        pl_module.model.temperature = temperature
        # print(pl_module.model.temperature)

    def on_test_epoch_start(self, trainer, pl_module):
        pl_module.model.temperature = self.min
        pass
