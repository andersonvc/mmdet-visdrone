# Setting Up Project for First Time

#### Setting up conda environment
> cd mmdet-visdrone
> conda env update -f env.yaml
> conda activate visdrone-full
> git pull --recurse-submodules mmdetection
> pip install -r mmdetection/requirements/build.txt
> pip install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools"
> pip install -v -e ./mmdetection

#### Download Visdrone Dataset
Data is available here: http://aiskyeye.com/download/ and extracted into the `./data` directory.

#### Running a training session
- In a separate terminal, open a tensorboard dashboard to track training progress.
   >cd mmdet-visdrone
   >conda activate visdrone-full
   >tensorboard --logdir=experiments
- In the main terminal, run the train.py script in tools. This requires you to provide a work_dir and config_file.
  > cd mmdet-visdrone
  > conda activate visdrone-full
  > python tools/train.py --config_file ./configs/tmp_model.py --work_dir ./experiments/tmp_model
- The tmp_model.py is just a placeholder frcnn model who's training profile isn't completely terrible.  