FROM nvidia/cuda:11.4.2-devel-ubuntu20.04
ENV TZ=America/New_York

RUN apt-get update --fix-missing && \
    apt-get -y install apt-utils && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq curl wget bzip2 \
    ca-certificates git libgl1 libglib2.0-0 python3.8 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y python3.8-dev python3-pip

COPY requirements.txt .

RUN pip3 install -r requirements.txt
RUN git clone https://github.com/open-mmlab/mmcv.git && \
    cd mmcv && \
    MMCV_WITH_OPS=1 MAX_JOBS=16 python3 -m pip install -e .

RUN echo $(python3 -c "import torch; print(torch.cuda.is_available())")

WORKDIR /app

COPY train.py /app
COPY pytorch2onnx.py /app

ENTRYPOINT echo GPU enabled: $(python3 -c "import torch; print(torch.cuda.is_available())") && \
    python3 /app/train.py --experiment_name test --config_file ${BASE_DIR}/configs/faster_rcnn_r50_fpn_1x2_1x_visdrone.py