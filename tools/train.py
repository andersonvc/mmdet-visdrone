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
    cfg = Config.fromfile(args.config_file)
except Exception as e:
    logging.error(" ".join(["failed to load config_file:", args.config_file]))
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
