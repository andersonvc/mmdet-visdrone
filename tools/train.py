import argparse
import logging
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

load_dotenv()
assert None not in {
    os.getenv("BASE_DIR"),
    os.getenv("MLFLOW_DATA_DIR"),
    os.getenv("MLFLOW_MODEL_DIR"),
}, "Not all environment variables set. Be sure to edit .env"
sys.path.append(str(os.getenv("BASE_DIR")))

import mlflow
import mmcv
from mmcv import Config
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.models import build_detector
from mmdet.apis import train_detector

from visdrone.datasets import VisDroneDataset
from visdrone.baseline_configs import (
    configure_fixtures,
    configure_dataloader,
    configure_scheduler,
    configure_model,
)


logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.DEBUG
)

parser = argparse.ArgumentParser(description="Train Custom Model")
parser.add_argument("--config_file", help="path to mmdet training config file")
parser.add_argument("--experiment_name", help="assign experiment name")
parser.add_argument(
    "--work_dir",
    help="parent directory to store experiment logs & weights",
    default=os.environ.get("MFLOW_DATA_DIR"),
)
args = parser.parse_args()

logging.info("loading config file")
try:
    # Build the entire experiment run config from composite files
    dataloader = configure_dataloader()
    fixtures = configure_fixtures(experiment_name="basic_cascade_rcnn_r50_1x")
    scheduler = configure_scheduler(
        optimizer_type="Adam", learning_rate=0.02, total_epochs=12
    )
    model = configure_model(model_name="CascadeRCNN")

    cfg = Config()
    cfg.merge_from_dict({**dataloader, **fixtures, **scheduler, **model})

except Exception as e:
    logging.error(" ".join(["failed to load config_file:", args.config_file]))
    print(e)

cfg.work_dir = args.work_dir


logging.info("loading datasets")
datasets = [build_dataset(cfg.data.train)]

logging.info("generating model")
model = build_detector(cfg.model, train_cfg=cfg.get("train_cfg"))
model.CLASSES = VisDroneDataset.CLASSES
model.init_weights()

# Create work_dir
mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
logging.info("starting model training loop")
train_detector(model, datasets, cfg, distributed=False, validate=True)
