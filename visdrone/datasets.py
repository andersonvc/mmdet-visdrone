import os
from PIL import Image
import pandas as pd
import numpy as np
from mmdet.datasets import build_dataset
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset

"""
load_annotation tweaked to fix issue with custom dataset creation. 
When resizing ground truth images, if the updated annotated bbox is too small, the localization loss will be NaN. 
To fix this, we will ensure the bbox for the transformed image is >= 5x5 pixels.
Additionally, if an image contains no instances, we will exclude it from the training / validation set.
"""

custom_dataset = "VisDroneDataset"
if not DATASETS.get(custom_dataset):

    @DATASETS.register_module()
    class VisDroneDataset(CustomDataset):

        ALLOWED_TRUNCATION = 1  # 0=None, 1=1%-50%
        ALLOWED_OCCLUSION = 1  # 0=None, 1=1%-50%, 2=50%-100%
        CLASSES = (
            "Ignored",
            "Pedestrian",
            "People",
            "Bicycle",
            "Car",
            "Van",
            "Truck",
            "Tricycle",
            "Tricycle-awn",
            "Bus",
            "Motor",
            "Others",
        )

        def __init__(self, transform_dims=(1400, 788), **kwargs):
            self.transform_dims = transform_dims
            super().__init__(**kwargs, filter_empty_gt=True)

        def load_annotations(self, filepath):

            self.sample_filenames = [
                v.split(".")[0] for v in os.listdir(self.img_prefix)
            ]
            annotation_prefix = self.img_prefix.split("images")[0] + "/annotations"
            mmdet_annotations = []
            for i, image_id in enumerate(self.sample_filenames):
                if image_id == "":
                    continue
                image_filename = f"{self.img_prefix}/{image_id}.jpg"

                annotation_filename = f"{annotation_prefix}/{image_id}.txt"
                img = Image.open(image_filename)
                w, h = img.size

                ann = pd.read_csv(annotation_filename, header=None)

                width_resize = self.transform_dims[0] / w
                height_resize = self.transform_dims[1] / h

                # Filter out occluded and truncated bounding boxes
                # Filter out reshaped images smaller than 5 pixels wide or 5 pixels high
                ann = ann[
                    (ann.iloc[:, 6] <= self.ALLOWED_TRUNCATION)
                    & (ann.iloc[:, 7] <= self.ALLOWED_OCCLUSION)
                    & (ann.iloc[:, 2] * width_resize >= 5)
                    & (ann.iloc[:, 3] * height_resize >= 5)
                ].reset_index(drop=True)

                bboxes = np.array(
                    [
                        ann.iloc[:, 0],
                        ann.iloc[:, 1],
                        ann.iloc[:, 0] + ann.iloc[:, 2],
                        ann.iloc[:, 1] + ann.iloc[:, 3],
                    ],
                    dtype=np.float32,
                ).T
                labels = np.array(ann.iloc[:, 5], dtype=int)

                if labels.shape[0] <= 1 or labels.shape[0] > 500:
                    continue

                record = {
                    "filename": image_filename.split("/")[-1],
                    "width": w,
                    "height": h,
                    "ann": {
                        "bboxes": bboxes,
                        "labels": labels,
                    },
                }

                mmdet_annotations.append(record)

            return mmdet_annotations
