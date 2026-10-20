import numpy as np
import mmcv
from mmdet.datasets.builder import PIPELINES
from pyquaternion import Quaternion


@PIPELINES.register_module()
class LoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool, optional): Whether to convert the img to float32.
            Defaults to False.
        color_type (str, optional): Color type of the file.
            Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type="unchanged"):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results["img_filename"]
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1
        )
        if self.to_float32:
            img = img.astype(np.float32)
        results["filename"] = filename
        # unravel to list, see `DefaultFormatBundle` in formatting.py
        # which will transpose each image separately and then stack into array
        results["img"] = [img[..., i] for i in range(img.shape[-1])]
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        # Set initial values for default meta_keys
        results["pad_shape"] = img.shape
        results["scale_factor"] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results["img_norm_cfg"] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False,
        )
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(to_float32={self.to_float32}, "
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class LoadPointsFromFile(object):
    """Load Points From File.

    Load points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int, optional): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int], optional): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool, optional): Whether to use shifted height.
            Defaults to False.
        use_color (bool, optional): Whether to use color features.
            Defaults to False.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(
        self,
        coord_type,
        load_dim=6,
        use_dim=[0, 1, 2],
        shift_height=False,
        use_color=False,
        file_client_args=dict(backend="disk"),
    ):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert (
            max(use_dim) < load_dim
        ), f"Expect all used dimensions < {load_dim}, got {use_dim}"
        assert coord_type in ["CAMERA", "LIDAR", "DEPTH"]

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith(".npy"):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results["pts_filename"]
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3], np.expand_dims(height, 1), points[:, 3:]], 1
            )
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(
                    color=[
                        points.shape[1] - 3,
                        points.shape[1] - 2,
                        points.shape[1] - 1,
                    ]
                )
            )

        results["points"] = points
        return results

@PIPELINES.register_module()
class LoadLiDARPointsFromPCD(object):
    """Load Points From File.

    Load points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int, optional): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int], optional): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool, optional): Whether to use shifted height.
            Defaults to False.
        use_color (bool, optional): Whether to use color features.
            Defaults to False.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(
        self,
        coord_type,
        load_dim=6,
        use_dim=[0, 1, 2],
        shift_height=False,
    ):
        self.shift_height = shift_height
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert (
            max(use_dim) < load_dim
        ), f"Expect all used dimensions < {load_dim}, got {use_dim}"
        assert coord_type in ["CAMERA", "LIDAR", "DEPTH"]

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim

    def _load_points(self, pts_filename, lidar2ego):
        pc_ego = parse_lidar_pcd_file(pts_filename)  # x y z intensity
        pc_ego = np.stack(
            [pc_ego[field].astype(np.float32) for field in pc_ego.dtype.names],
            axis=1,
        )  # x y z intensity

        # 增加 ring 列
        ring_col = np.full((pc_ego.shape[0], 1), 0, dtype=np.float32)  # [N, 1]
        pc_ego = np.hstack([pc_ego, ring_col])  # x y z intensity lidar_id

        # 转换到 LiDAR 坐标系
        xyz_ego = pc_ego[:, :3]  # [N, 3]
        xyz_ego_homo = np.hstack([xyz_ego, np.ones((xyz_ego.shape[0], 1))])  # [N, 4]
        xyz_lidar_homo = (np.linalg.inv(lidar2ego) @ xyz_ego_homo.T).T  # [N, 4]
        xyz_lidar = xyz_lidar_homo[:, :3]  # [N, 3]

        intensity = pc_ego[:, 3:4]
        pc_lidar = np.hstack([xyz_lidar, intensity, ring_col])  # [N, 5]

        # # 车身点
        # egocar_mask = (np.abs(xyz_lidar[:, 0]) <= half_l) & (np.abs(xyz_lidar[:, 1]) <= half_w)
        # # 地面以下的点
        # below_ground_mask = xyz_ego[:, 2] < -3.0
        # pointcloud_mask = ~egocar_mask & ~below_ground_mask

        # pc_lidar = pc_lidar[pointcloud_mask]
        # pc_ego = pc_ego[pointcloud_mask]
        # print(f"Filtered {egocar_mask.sum()} ego car points and {below_ground_mask.sum()} below ground points.")

        pc_lidar = pc_lidar.astype(np.float32)
        return pc_lidar

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = Quaternion(results["lidar2ego_rotation"]).rotation_matrix
        lidar2ego[:3, 3] = np.array(results["lidar2ego_translation"])

        pts_filename = results["pts_filename"]
        points = self._load_points(pts_filename, lidar2ego)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3], np.expand_dims(height, 1), points[:, 3:]], 1
            )
            attribute_dims = dict(height=3)

        results["points"] = points
        return results

def parse_lidar_pcd_file(pcd_path, return_fields=False):
    with open(pcd_path, "rb") as f:
        # 读取头部信息
        header = []
        while True:
            line = f.readline().decode("utf-8").strip()

            if line.startswith("DATA"):
                data_type = line.split()[1]
                break
            if line:
                header.append(line)

        # 解析头部信息
        def get_header_value(key):
            for h in header:
                if h.startswith(key):
                    return h.split()[1:]
            return None

        fields = get_header_value("FIELDS")
        sizes = [int(s) for s in get_header_value("SIZE")]
        types = get_header_value("TYPE")
        counts = [int(c) for c in get_header_value("COUNT")]
        points = int(get_header_value("POINTS")[0])

        # 构建 dtype
        dtype_list = []
        for i, field in enumerate(fields):
            count = counts[i]
            size = sizes[i]
            typ = types[i]

            if typ == "F":
                np_type = np.float32 if size == 4 else np.float64
            elif typ == "U":
                np_type = (
                    np.uint8
                    if size == 1
                    else np.uint16
                    if size == 2
                    else np.uint32
                    if size == 4
                    else np.uint64
                )
            elif typ == "I":
                np_type = (
                    np.int8
                    if size == 1
                    else np.int16
                    if size == 2
                    else np.int32
                    if size == 4
                    else np.int64
                )
            else:
                raise ValueError(f"不支持的类型: {typ}")

            for j in range(count):
                field_name = f"{field}_{j}" if count > 1 else field
                dtype_list.append((field_name, np_type))

        dtype = np.dtype(dtype_list)

        # 根据 data_type 读取数据
        if data_type.lower() == "ascii":
            # 读取剩余所有行
            content = f.read().decode("utf-8").strip().splitlines()
            data_list = []
            for line in content:
                parts = line.strip().split()
                if not parts:
                    continue
                # 将字符串转为 float 或 int
                data_list.append([float(p) for p in parts])
            data_np = np.array(data_list, dtype=np.float32)

            # 如果字段数不一致（部分行缺失），做下安全检查
            if data_np.shape[1] != len(dtype_list):
                raise ValueError(
                    f"数据列数 {data_np.shape[1]} 与头部字段数 {len(dtype_list)} 不匹配"
                )

            # 转成结构化数组
            structured_data = np.zeros(data_np.shape[0], dtype=dtype)
            for i, name in enumerate(dtype.names):
                structured_data[name] = data_np[:, i]
            data = structured_data

        elif data_type.lower() == "binary":
            data = np.fromfile(f, dtype=dtype, count=points)

        else:
            raise ValueError(f"未知的数据类型: {data_type}")
        
        if return_fields:
            return data, fields
        return data
