import pytorch_lightning.loggers

__all__ = ["TensorBoardLogger"]


class TensorBoardLogger(pytorch_lightning.loggers.TensorBoardLogger):

    def __init__(self, save_dir, task: str, model: str, dataset: str,
                 input_length: int, output_length: int, *args, **kwargs):
        super().__init__(
            save_dir, "/".join(
                [task, dataset,
                 str(input_length),
                 str(output_length), model]), *args, **kwargs)
