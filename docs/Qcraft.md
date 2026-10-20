# 轻舟数据预处理指南
1. 准备内容：
    * 原始轻舟场景数据（包含每一帧的子文件夹以及传感器信息）
    * 绘制的 ego masks（图像以原相机名命名）
    * 标注好的 label 文件（每帧一个 `.json` 文件）
2. 放置方式：
    * 将 `ego_masks/` 文件夹和 `label/` 文件夹直接放在场景目录下（与每一帧的子文件夹同级）
    * 将场景目录直接放在 `data/qcraft/raw/` 下，不允许有任何中间层级
3. 目录结构如下：
```shell
data/qcraft/
 └── raw/
      └── 20250702_133223_Q2517_60_75  # 场景名（只有这一层目录）
           ├──1751434405.477/
           ├──1751434405.577/
           ├──1751434405.677/
           ├──...
           ├──ego_masks/
           │      ├──CAM_PBQ_FRONT_LEFT_RESET_OPTICAL_H99.png
           │      ├──CAM_PBQ_FRONT_RIGHT_RESET_OPTICAL_H99.png
           │      ├──CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H110.png
           │      └──...
           ├──label/
           │      ├──1751434405.477.json
           │      ├──1751434405.577.json
           │      ├──1751434405.677.json
           │      └──...
           ├──camera_params.json
           ├──data_frame_car_info.json
           ├──data_frame_car_info.pb
           ├──data_frame_seq.json
           └──data_frame_seq.pb.txt
```
4. 运行`scripts/qcraft/preprocess_data.sh`，其中`--skip_front_wide_side_cameras`参数决定是否跳过前向侧方两路广角镜头。例如，ThorU数据没有前视左侧广角和前视右侧广角，所以需要添加该参数。
5. 运行`scripts/qcraft/extract_masks.sh`提取sky mask和road mask。
6. 预处理数据的目录结构如下：
```shell
data/qcraft/
 └── processed/training/
      └── 20250702_133223_Q2517_60_75
           ├──dynamic_masks/  # 动态物体框的mask：{timestep:06d}_{cam_id}.png
           │      ├──all/    # 所有类别
           │      │   ├──000000_0.png
           │      │   ├──000000_1.png
           │      │   └──...
           │      ├──human/    # 行人
           │      └──vehicle/  # 车辆
           ├──sky_masks/  # 天空mask：{timestep:06d}_{cam_id}.png
           │      ├──000000_0.png
           │      ├──000000_1.png
           │      └──...
           ├──road_masks/  # 路面mask：{timestep:06d}_{cam_id}.png
           │      ├──000000_0.png
           │      ├──000000_1.png
           │      └──...
           ├──images/  # RGB图像：{timestep:06d}_{cam_id}.png
           │      ├──000000_0.png
           │      ├──000000_1.png
           │      ├──...
           │      └──timestamps.json  # 时间戳
           ├──ego_masks/    # 自车的mask：{cam_id}.png
           │      ├──0.png  
           │      ├──1.png
           │      └──...
           ├──ego_pose/  # 世界坐标系下的自车位姿
           │      ├──000000.txt    # 当前帧的ego pose: {timestep:06d}.txt
           │      ├──000000_0.txt  # 相机时间戳下的ego pose（不是camera pose）: {timestep:06d}_{cam_id}.txt
           │      ├──000000_1.txt
           │      └──...
           ├──lidar_pose/  # 世界坐标系下的main LiDAR位姿：{timestep:06d}.txt
           │      ├──000000.txt
           │      ├──000001.txt
           │      └──...
           ├──extrinsics/  # Main LiDAR坐标系下的相机外参：{cam_id}.txt
           │      ├──0.txt
           │      ├──1.txt
           │      └──...
           ├──intrinsics/  # 相机内参：{cam_id}.txt
           │      ├──0.txt
           │      ├──1.txt
           │      └──...
           ├──instances/  # 实例信息
           │      ├──frame_instances.json  # 每一帧出现的实例
           │      └──instances_info.json  # 各实例的类别、ID、出现帧号等信息
           ├──lidar/
           │      ├──bin/    # 每一帧的LiDAR数据：{timestep:06d}.bin
           │      │   ├──000000.bin
           │      │   ├──000001.bin
           │      │   └──...
           │      ├──actor/  # 动态物体：{obj_id}/
           │      │   ├──0/  # 当前物体出现帧的彩色点云: {timestep:06d}.ply
           │      │   │  ├──000000.ply
           │      │   │  ├──000001.ply
           │      │   │  └──...
           │      │   ├──1/
           │      │   ├──2/
           │      │   └──...
           │      ├──background/  # 静态背景的彩色点云：{timestep:06d}.ply
           │      │   ├──000000.ply
           │      │   ├──000001.ply
           │      │   └──...
           │      └──depth/  # LiDAR点向各相机投影的深度：{timestep:06d}_{cam_id}.npz
           │          ├──000000_0.npz
           │          ├──000000_1.npz
           │          └──...
           ├──sensor_info.json  # 传感器信息，包含相机配置和main LiDAR信息
           └──timestamps.json  # 时间戳
```