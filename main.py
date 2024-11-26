import os

import torch
import yaml
from pytorch_lightning.cli import LightningCLI


class CLI(LightningCLI):

    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.init_args.input_len",
                              "model.init_args.model.init_args.input_len")
        parser.link_arguments("data.init_args.output_len",
                              "model.init_args.model.init_args.output_len")

        parser.link_arguments("data.x_features_num",
                              "model.init_args.model.init_args.input_features",
                              apply_on="instantiate")
        parser.link_arguments("data.y_features_num",
                              "model.init_args.model.init_args.output_features",
                              apply_on="instantiate")

        parser.link_arguments("model.class_path",
                              "trainer.logger.init_args.task",
                              compute_fn=lambda s: s.split('.')[-1].strip())
        parser.link_arguments("model.init_args.model.class_path",
                              "trainer.logger.init_args.model",
                              compute_fn=lambda s: s.split('.')[-1].strip())
        parser.link_arguments(
            "data.init_args.dataset_path",
            "trainer.logger.init_args.dataset",
            compute_fn=lambda s: s.split('/')[-1].split('.')[0].strip())
        parser.link_arguments("data.init_args.input_len",
                              "trainer.logger.init_args.input_length")
        parser.link_arguments("data.init_args.output_len",
                              "trainer.logger.init_args.output_length")


torch.set_float32_matmul_precision("high")

cli = CLI(run=False)

cli.trainer.fit(cli.model, cli.datamodule)
best_model_path = cli.trainer.checkpoint_callback.best_model_path
last_model_path = cli.trainer.checkpoint_callback.last_model_path

best_result = cli.trainer.test(
    cli.model,
    datamodule=cli.datamodule,
    ckpt_path=best_model_path,
)

last_result = cli.trainer.test(
    cli.model,
    datamodule=cli.datamodule,
    ckpt_path=last_model_path,
)

result = {
    "best": best_result,
    "last": last_result,
    "best_path": best_model_path
}
with open(os.path.join(cli.trainer.logger.log_dir, "test_result.yaml"),
          "w",
          encoding='utf-8') as f:
    yaml.dump(result, f)
