import os
from mmcv import Config


def configure_fixtures(experiment_name="test", eval_metric="mAP"):
    """
    configure logging & reporting instrumentation
    """

    return Config(
        {
            "log_level": "INFO",
            "work_dir": f'{os.getenv("MLFLOW_MODEL_DIR")}/experiments/{experiment_name}',
            "log_config": {
                "interval": 50,
                "hooks": [
                    {"type": "MlflowLoggerHook", "exp_name": experiment_name},
                    {"type": "TensorboardLoggerHook"},
                ],
            },
            "evaluation": {"metric": eval_metric, "interval": 1},
            "checkpoint_config": {"interval": 1},
            "dist_params": {"backend": "nccl"},
            "load_from": None,
            "resume_from": None,
        }
    )


def configure_dataloader(
    dataset_type="VisDroneDataset",
    samples_per_gpu=2,
    workers_per_gpu=2,
    transform_dims=(1400, 788),
    pixel_mean=[96.56215221, 98.2655526, 94.69506836],
    pixel_std=[31.36219647, 34.50400645, 33.31346927],
):

    image_norm_coef = {"mean": pixel_mean, "std": pixel_std, "to_rgb": True}

    #####################
    ## Default Configs ##
    #####################

    train_pipeline = [
        {"type": "LoadImageFromFile"},
        {"type": "LoadAnnotations", "with_bbox": True},
        {"type": "Resize", "img_scale": transform_dims, "keep_ratio": False},
        {"type": "RandomFlip", "flip_ratio": 0.5},
        {"type": "Normalize", **image_norm_coef},
        {"type": "Pad", "size_divisor": 32},
        {"type": "DefaultFormatBundle"},
        {"type": "Collect", "keys": ["img", "gt_bboxes", "gt_labels"]},
    ]

    test_pipeline = [
        {"type": "LoadImageFromFile"},
        {
            "type": "MultiScaleFlipAug",
            "img_scale": transform_dims,
            "flip": False,
            "transforms": [
                {"type": "Resize", "keep_ratio": False},
                {"type": "RandomFlip"},
                {"type": "Normalize", **image_norm_coef},
                {"type": "Pad", "size_divisor": 32},
                {"type": "ImageToTensor", "keys": ["img"]},
                {"type": "Collect", "keys": ["img"]},
            ],
        },
    ]

    dataset_type = "VisDroneDataset"
    data = {
        "samples_per_gpu": samples_per_gpu,
        "workers_per_gpu": workers_per_gpu,
        "train": {
            "type": dataset_type,
            "ann_file": None,
            "img_prefix": f"{os.getenv('MLFLOW_DATA_DIR')}/VisDrone2019-DET-train/images",
            "pipeline": train_pipeline,
        },
        "val": {
            "type": dataset_type,
            "ann_file": None,
            "img_prefix": f"{os.getenv('MLFLOW_DATA_DIR')}/VisDrone2019-DET-val/images",
            "pipeline": test_pipeline,
        },
        "test": {
            "type": dataset_type,
            "ann_file": None,
            "img_prefix": f"{os.getenv('MLFLOW_DATA_DIR')}/VisDrone2019-DET-val/images",
            "pipeline": test_pipeline,
        },
    }

    seed = 0
    # set_random_seed(0, deterministic=False)
    gpu_ids = range(1)  # Assuming we only have 1 GPU
    workflow = [("train", 1)]

    return Config({"data": data, "seed": 0, "gpu_ids": gpu_ids, "workflow": workflow})


def configure_scheduler(
    optimizer_type="Adam",
    learning_rate=0.001,
    use_grad_clipping=True,
    total_epochs=12,
    learning_policy="step",
    warmup_iterations=500,
    warmup_ratio=0.333,
    policy_steps=[8, 11],
    **kwargs,
):

    #######################
    ## Automated Configs ##
    #######################

    # optimizer
    optimizer = {"type": optimizer_type, "lr": learning_rate}

    # optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.001)

    optimizer_config = (
        {"grad_clip": {"max_norm": 35, "norm_type": 2}} if use_grad_clipping else {}
    )

    # learning policy
    lr_config = {
        "policy": learning_policy,
        "warmup": "linear",
        "warmup_iters": warmup_iterations,
        "warmup_ratio": warmup_ratio,
        "step": policy_steps,
    }

    # runtime settings
    runner = {"type": "EpochBasedRunner", "max_epochs": total_epochs}

    return Config(
        {
            "optimizer": optimizer,
            "optimizer_config": optimizer_config,
            "lr_config": lr_config,
            "runner": runner,
        }
    )


def configure_model(model_name="CascadeRCNN"):
    model_dir = (f'{os.getenv("BASE_DIR")}/visdrone/models',)

    if model_name.lower() == "cascadercnn":
        exec(
            open(
                f"{os.getenv('BASE_DIR')}/visdrone/models/cascade_rcnn_r50_fpn.py", "r"
            ).read(),
            globals(),
        )
    else:
        raise Exception(f"model_name doesnt exist: {model_name}")

    return Config({"model": model})
