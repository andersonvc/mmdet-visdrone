_base_ = [
    "./_base_/models/faster_rcnn_r50_fpn.py",
    "./_base_/schedules/schedule_1x.py",
    "./_base_/datasets/visdrone_dataset.py",
    "./_base_/default_runtime.py",
]

gpu_count = 1
batch_size = 2
epoch_count = 12
class_label_count = 12

model = {"roi_head": {"bbox_head": {"num_classes": class_label_count}}}
gpu_ids = range(gpu_count)
data = {"samples_per_gpu": batch_size, "workers_per_gpu": batch_size}
