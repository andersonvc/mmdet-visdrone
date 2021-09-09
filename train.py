import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.append(str(os.getenv("BASE_DIR")))

import argparse
import logging
from pathlib import Path
import shutil
import subprocess

import mlflow
import mmcv
from mmcv import Config
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdetection.tools.deployment.pytorch2onnx import pytorch2onnx

from visdrone.datasets import VisDroneDataset
from visdrone.hooks import CustomMlflowLoggerHook, CustomTensorboardLoggerHook


mmcv_logger = mmcv.utils.logging.get_logger(
    __name__, log_file=None, log_level=logging.ERROR, file_mode="w"
)
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.ERROR
)


def init_mlflow_run(args):
    """
    Using the argparse inputs, this function will initialize the mlflow experiment and run objects.
    If necessary, this function also creates a new experiment object.

    Output: (experiment_id, run_id)
    """

    mlflow.set_tracking_uri(f'{os.getenv("MLFLOW_BASE_DIR")}')

    # Create experiment if it doesn't already exist
    tmp_experiment_name = mlflow.tracking.MlflowClient().get_experiment_by_name(
        args.experiment_name
    )
    if tmp_experiment_name == None:
        mlflow.create_experiment(args.experiment_name)
    mlflow.set_experiment(args.experiment_name)

    experiment_id = (
        mlflow.tracking.MlflowClient()
        .get_experiment_by_name(args.experiment_name)
        .experiment_id
    )

    experiment_run = mlflow.tracking.MlflowClient().create_run(experiment_id)
    run_id = experiment_run.info.run_id

    custom_config_entries = [
        f"{k}:{v}"
        for k, v in vars(args).items()
        if k not in {"experiment_name", "config_file"}
    ]
    custom_run_name = "run_" + "_".join(custom_config_entries)
    mlflow.tracking.MlflowClient().set_tag(run_id, "mlflow.runName", custom_run_name)

    return experiment_id, run_id


def init_mmdet_model(args):
    """
    Builds the mmdetection configuration model used for training and inference
    """

    cfg = Config.fromfile(f"{args.config_file}")

    # Override references to data dir
    for entry in [cfg.data.train, cfg.data.val, cfg.data.test]:
        entry.img_prefix = f'{os.getenv("MLFLOW_DATA_DIR")}/{entry.img_prefix}'

    # save only the final trained model's weights
    cfg.checkpoint_config.interval = cfg.epoch_count

    return cfg


def update_custom_parameters(cfg, args):
    """
    Updates the mmdet model with any config parameters set for this run;
    Logs all custom parameters;
    Returns a string that can be use as the name for the custom run & dict of our revised config params
    """

    custom_params = {
        k: v
        for k, v in vars(args).items()
        if k not in {"experiment_name", "config_file"}
    }

    def update_model(param_name, cfg_model_object):
        # if param is passed in as an argvar, we want to update the mmdet model
        if param_name in custom_params:
            cfg_model_object = custom_params[param_name]
        else:
            custom_params[param_name] = cfg_model_object

    # Assign parameter values
    if cfg.model.type.lower() == "fasterrcnn":
        update_model("model", cfg.model.type)
        update_model("backbone", cfg.model.backbone.type)
        if "neck" in cfg.model:
            update_model("neck", cfg.model.neck.type)
        else:
            update_model("neck", "None")
        update_model("optim", cfg.optimizer.type)
        update_model("lr", cfg.optimizer.lr)
        update_model("momentum", cfg.optimizer.momentum)
        custom_run_name = f"FRCNN.{cfg.model.backbone.type}-{custom_params['optim']}-lr:{custom_params['lr']}-mom:{custom_params['momentum']}"
    elif cfg.model.type.lower() == "cascadercnn":
        update_model("model", cfg.model.type)
        update_model("backbone", cfg.model.backbone.type)
        if "neck" in cfg.model:
            update_model("neck", cfg.model.neck.type)
        else:
            update_model("neck", "None")
        update_model("optim", cfg.optimizer.type)
        update_model("lr", cfg.optimizer.lr)
        update_model("momentum", cfg.optimizer.momentum)
        custom_run_name = f"CRCNN.{cfg.model.backbone.type}-{custom_params['optim']}-lr:{custom_params['lr']}-mom:{custom_params['momentum']}"

    else:
        raise Exception("Training script only configured for FasterRCNN models")

    return custom_run_name, custom_params


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Custom Model")
    parser.add_argument("--experiment_name")
    parser.add_argument("--config_file")
    parser.add_argument("--lr", default=0.02)
    args = parser.parse_args()

    # Initialize mmdet model and corresponding mlflow experiment/run
    mmdet_config = init_mmdet_model(args)
    experiment_id, run_id = init_mlflow_run(args)

    # Update model parameters and assign custom run name
    run_name, custom_params = update_custom_parameters(mmdet_config, args)
    mlflow.tracking.MlflowClient().set_tag(run_id, "mlflow.runName", run_name)

    with mlflow.start_run(experiment_id=experiment_id, run_id=run_id) as run:

        # Log custom parameters in mlflow dashboard
        for k, v in custom_params.items():
            mlflow.log_param(k, v)

        artifact_path = run.info.artifact_uri
        mmdet_config.work_dir = artifact_path

        # Build custom datasets -- resize_dims is needed as part of our bounding box filter logic
        datasets = [
            build_dataset(
                mmdet_config.data.train, {"resize_dims": mmdet_config.resize_dims}
            ),
        ]

        logging.info("generating model")
        model = build_detector(
            mmdet_config.model, train_cfg=mmdet_config.get("train_cfg")
        )
        model.CLASSES = VisDroneDataset.CLASSES
        model.init_weights()

        with open(f"{artifact_path}/mmdet_model.py", "w+") as f:
            f.writelines(mmdet_config.pretty_text)

        # Train model
        logging.info("starting model training loop")
        train_detector(model, datasets, mmdet_config, distributed=False, validate=True)

        # Generate onnx file for trained model
        img_width, img_height = mmdet_config.resize_dims
        try:
            rc = subprocess.run(
                [
                    f"python",
                    f"{os.getenv('BASE_DIR')}/mmdetection/tools/deployment/pytorch2onnx.py",
                    f"{artifact_path}/mmdet_model.py",
                    f"{artifact_path}/latest.pth",
                    f"--output-file",
                    f"{artifact_path}/model.onnx",
                    f"--input-img",
                    f"{os.getenv('BASE_DIR')}/sample.jpg",
                    f"--shape",
                    f"{img_width}",
                    f"{img_height}",
                ],
                capture_output=True,
                text=True,
            )
        except Exception as e:
            logging.error(f"Failed to generate ONNX file: {e}")
