import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.append(str(os.getenv("BASE_DIR")))

from mmcv import Config
from hyperopt import fmin, tpe, hp, Trials


def set_hyperopt_space():
    """
    Generates the hyperparameter search space that can be traversed during trials.
    At the moment, only the model's optimizer and backbone can be tuned using hyperopt.
    """
    optim_space = {
        "optim": hp.choice(
            "optim",
            [
                {
                    "type": "SGD",
                    "lr": hp.uniform("SGD-lr", 0.0001, 0.1),
                    "momentum": hp.uniform("SGD-momentum", 0.0001, 0.9),
                    "weight_decay": hp.uniform("SGD-weight_decay", 0.0001, 0.1),
                    "nesterov": hp.choice("SGD-nesterov", (True, False)),
                },
                {"type": "Adam", "lr": hp.uniform("Adam-lr", 0.00001, 0.1)},
            ],
        )
    }

    backbone_space = {
        "backbone": hp.choice("backbone", ["ResNet50", "ResNet101", "Res2Net101"])
    }

    hyperopt_space = {**optim_space, **backbone_space}

    return hyperopt_space


def parse_hyperopt_args(hyperopt_args):
    """
    When the trial starts, hyperopt will select all the hyperparameters to tweak from it's potential search space.
    The structure of the provided search space is a bit wonky. This function will transform the selected values
    (currently only the optimizer and backbone) and generate the relevant mmdetection config parameter dictionary.

    Parameters:
      hyperopt_args (dict): populated hyperparameter 'space' object. Current valid keys are 'optim' and 'backbone'

    Returns:
      mmdet_config (dict): modified mmdet Config dict, which can overwrite the baseline Config object to incorporate hyperopt_args
      custom_params (dict): dict containing custom parameters used in this trial run.
      run_name (str): creates a short descriptive name for run; useful for plotting ablation studies.
    """

    cfg = Config()

    # Configure Optimizer
    if hyperopt_args["optim"]["type"] == "SGD":
        cfg.optimizer = hyperopt_args["optim"]
    elif hyperopt_args["optim"]["type"] == "Adam":
        cfg.optimizer = hyperopt_args["optim"]

    # Configure Scheduler
    cfg.optimizer_config = {"grad_clip": {"max_norm": 35, "norm_type": 2}}
    cfg.lr_config = {
        "policy": "step",
        "warmup": "linear",
        "warmup_iters": 500,
        "warmup_ratio": 0.001,
        "step": [8, 11],
    }
    cfg.runner = {"type": "EpochBasedRunner", "max_epochs": 12}
    cfg.total_epochs = 12

    # Configure Backbone
    if hyperopt_args["backbone"] == "ResNet50":
        tmp_cfg = Config.fromfile(
            f"{os.getenv('BASE_DIR')}/configs/faster_rcnn/resnet50_backbone.py"
        )
        cfg.merge_from_dict(tmp_cfg)
    elif hyperopt_args["backbone"] == "ResNet101":
        tmp_cfg = Config.fromfile(
            f"{os.getenv('BASE_DIR')}/configs/faster_rcnn/resnet101_backbone.py"
        )
        cfg.merge_from_dict(tmp_cfg)
    elif hyperopt_args["backbone"] == "Res2Net101":
        tmp_cfg = Config.fromfile(
            f"{os.getenv('BASE_DIR')}/configs/faster_rcnn/res2net_backbone.py"
        )
        cfg.merge_from_dict(tmp_cfg)
    elif hyperopt_args["backbone"] == "ResNeSt":
        tmp_cfg = Config.fromfile(
            f"{os.getenv('BASE_DIR')}/configs/faster_rcnn/resnest_backbone.py"
        )
        cfg.merge_from_dict(tmp_cfg)

    # Configure Neck
    # Just using feature pyramids for now.

    # Configure RPN/Head
    # Nothing to change here; using defaults for baseline FasterRCNN model.

    custom_params = {
        "backbone": hyperopt_args["backbone"],
        **hyperopt_args["optim"],
        "neck": "FPN",
        "model": "FRCNN",
    }
    custom_params["optim"] = hyperopt_args["optim"]["type"]
    del custom_params["type"]

    # Run name
    run_name = f"{custom_params['model']}-{custom_params['backbone']}-{custom_params['neck']}-{custom_params['optim']}-lr:{custom_params['lr']:.1e}"

    return cfg, custom_params, run_name


def run_hyperopt_trial(experiment_name, config_file, train_fn, trial_cnt=50):
    """
    Retrains the visdrone model multiple times using different hyperparameters.
    Tree-structured Parzen Estimator (TPE) is used to search the parameter space.
    Trained models are compared against one another via the mAP score on the validation data.

    Parameters:
      experiment_name (str) -> Name of the model; all runs will be grouped under this experiment name
      config_file (str) -> Path to the baseline mmdet Config file that defines the run's data/model/scheduler/parameters
      train_fn (lambda function) -> Function used to setup & train the model

    Returns:
      None -> Outputs are logged/stored on the MLFlow server
    """

    trials = Trials()

    hyperopt_space = set_hyperopt_space()

    best = fmin(
        fn=train_fn,
        space={
            "experiment_name": experiment_name,
            "config_file": config_file,
            **hyperopt_space,
        },
        algo=tpe.suggest,
        max_evals=trial_cnt,
        trials=trials,
    )
