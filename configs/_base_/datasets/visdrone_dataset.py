resize_dims = (1400, 788)

normalize_image = {
    "type": "Normalize",
    "mean": [96.56215221, 98.2655526, 94.69506836],
    "std": [31.36219647, 34.50400645, 33.31346927],
    "to_rgb": True,
}

resize_image = {"type": "Resize", "img_scale": resize_dims, "keep_ratio": False}

pad_image = {"type": "Pad", "size_divisor": 32}

train_pipeline = [
    {"type": "LoadImageFromFile"},
    {"type": "LoadAnnotations", "with_bbox": True},
    {"type": "RandomFlip", "flip_ratio": 0.5},
    resize_image,
    normalize_image,
    pad_image,
    {"type": "DefaultFormatBundle"},
    {"type": "Collect", "keys": ["img", "gt_bboxes", "gt_labels"]},
]

test_pipeline = [
    {"type": "LoadImageFromFile"},
    {
        "type": "MultiScaleFlipAug",
        "img_scale": resize_dims,
        "flip": False,
        "transforms": [
            resize_image,
            {"type": "RandomFlip"},
            normalize_image,
            pad_image,
            {"type": "ImageToTensor", "keys": ["img"]},
            {"type": "Collect", "keys": ["img"]},
        ],
    },
]

data = {
    "samples_per_gpu": 2,
    "workers_per_gpu": 2,
    "train": {
        "type": "VisDroneDataset",
        "ann_file": None,
        "img_prefix": "VisDrone2019-DET-train/images",
        "pipeline": train_pipeline,
    },
    "val": {
        "type": "VisDroneDataset",
        "ann_file": None,
        "img_prefix": "VisDrone2019-DET-val/images",
        "pipeline": test_pipeline,
    },
    "test": {
        "type": "VisDroneDataset",
        "ann_file": None,
        "img_prefix": "VisDrone2019-DET-val/images",
        "pipeline": test_pipeline,
    },
}
workflow = [("train", 1)]
