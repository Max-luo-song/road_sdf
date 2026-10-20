# 针孔转鱼眼
## 🚩 功能

- ✅ 可以指定单张图片、单个视频、或图片文件夹进行针孔转鱼眼
- ✅ 可以设置畸变系数和畸变强度
- ✅ 可以开启超分辨率功能
- ✅ 可以指定运行的GPU
- ✅ 更多功能请从run_pinhole2fisheye的传递参数查看

---


## 环境安装

```bash
conda create -n pinhole2fisheye python=3.8 -y
conda activate pinhole2fisheye

pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install opencv-python==4.6.0.66
pip install scipy==1.8.1
pip install addict future lmdb pyyaml requests Pillow tqdm scikit-image==0.21.0
pip install basicsr==1.4.2 --no-deps
 pip install facexlib==0.3.0
# pip install gfpgan==1.3.8
python setup.py develop
```

---


## 运行命令

1. 对单张图片进行针孔转鱼眼

```bash
python run_pinhole2fisheye.py -i examples/imgs/image-0.jpg
```

2. 对单个视频进行针孔转鱼眼，设置更高的畸变强度，开启超分辨率，指定超分辨率模型，指定GPU 0

```bash
python run_pinhole2fisheye.py -i examples/video.mp4 -f 120 --sr -n RealESRGAN_x2plus -g 0
```

3. 对某个目录下的图片进行针孔转鱼眼

```bash
python run_pinhole2fisheye.py -i examples/imgs
```
