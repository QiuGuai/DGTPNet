# DGTPNet
Dynamic Gaussian Transformer PointCloud Network

# Environment

- PyTorch >= 1.7.0
- python >= 3.7
- CUDA >= 9.0
- GCC >= 4.9 
- torchvision
- timm
- open3d
- tensorboardX

```
pip install -r requirements.txt
```

# Building Pytorch Extensions for Chamfer Distance, PointNet++ and kNN

*NOTE:* PyTorch >= 1.7 and GCC >= 4.9 are required.

### Chamfer Distance

```
bash install.sh
```

### PointNet++

```
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```

### GPU kNN

```
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

Note: If you still get `ModuleNotFoundError: No module named 'gridding'` or something similar then run these steps

```
    1. cd into extensions/Module (eg extensions/gridding)
    2. run `python setup.py install`
```

That will fix the `ModuleNotFoundError`.

# Dataset and Pretrained Models

You can download the dataset and the pretrained model from the links below:

ShapeNet-55 Dataset: [Google Drive Link](https://drive.google.com/file/d/1jUB5yD7DP97-EqqU2A9mmr61JpNwZBVK/view?usp=sharing)

Trained ODGNet Model Weights: [Google Drive Link](https://drive.google.com/drive/folders/1Azz3rQSzax7dh14vXBbQXjEbuMQBIbui?usp=drive_link)

# Usage

### Training

Train the point cloud completion model using Occlusion + Fog mode, run:

```
bash ./scripts/train.sh 0 \
    --config ./cfgs/ShapeNet55_models/DGTPNet.yaml \
    --exp_name example
```

Train the point cloud completion model using online cropping mode, run:

```
mv ./runner.py ./runner_fog.py
mv ./runner_frop.py ./runner.py
bash ./scripts/train.sh 0 \
    --config ./cfgs/ShapeNet55_models/DGTPNet.yaml \
    --exp_name example
```


### Evaluation

Evaluate a pre-trained model on the Occlusion + Fog Dataset, run:

```
bash ./scripts/test.sh 0 \
    --config ./cfgs/ShapeNet55_models/DGTPNet.yaml \
    --exp_name example
```

Evaluate a pre-trained model on the online cropping Dataset, run:

```
mv ./runner.py ./runner_fog.py
mv ./runner_frop.py ./runner.py
bash ./scripts/test.sh 0 \
    --config ./cfgs/ShapeNet55_models/DGTPNet.yaml \
    --exp_name example
```

### Inference

Inference sample(s) with pretrained model

Method1:

```
python tools/inference.py \
${POINTR_CONFIG_FILE} ${POINTR_CHECKPOINT_FILE} \
[--pc_root <path> or --pc <file>] \
[--save_vis_img] \
[--out_pc_root <dir>] \
```

For example:

```
python tools/inference.py \
cfgs/PCN_models/DGTPNet.yaml \
pretrained/DGTPNet.pth \
--pc_root demo/ \ 
--save_vis_img  \
--out_pc_root show/DGTPNet/
```

Method2:

```
bash ./scripts/test.sh \
    <GPU_ID> \
    --ckpts <POINTR_CHECKPOINT_FILE> \
    --config <POINTR_CONFIG_FILE> \
    --pcd_path <input_point_cloud_file> \
    --output_dir <output_directory>
```

For example:

```
bash ./scripts/test.sh \
    0 \
    --ckpts ./pretrained/fog/DGTPNet.pth \
    --config ./cfgs/ShapeNet55_models/DGTPNet.yaml \
    --pcd_path ./demo/fog.txt \
    --output_dir ./show/DGTPNet
```

# Branch-a
The ODGNet trained on the Occlusion + Fog Dataset.

# Acknowledgement
Some codes are borrowed from [PoinTr](https://github.com/yuxumin/pointr) and [ODGNet](https://github.com/corecai163/ODGNet)