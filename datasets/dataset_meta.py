DATASETS_CONFIG = {
    "chery": {
        # Pinhole
        0: {
            "camera_name": "front_wide",
            "original_size": (2160, 3840),
            "egocar_visible": True,
            "is_fisheye": False,
        },
        1: {
            "camera_name": "front_main",
            "original_size": (2160, 3840),
            "egocar_visible": False,
            "is_fisheye": False,
        },
        2: {
            "camera_name": "left_front",
            "original_size": (1280, 1920),
            "egocar_visible": False,
            "is_fisheye": False,
        },
        3: {
            "camera_name": "left_rear",
            "original_size": (1280, 1920),
            "egocar_visible": False,
            "is_fisheye": False,
        },
        4: {
            "camera_name": "right_front",
            "original_size": (1280, 1920),
            "egocar_visible": False,
            "is_fisheye": False,
        },
        5: {
            "camera_name": "right_rear",
            "original_size": (1280, 1920),
            "egocar_visible": False,
            "is_fisheye": False,
        },
        6: {
            "camera_name": "rear_main",
            "original_size": (1280, 1920),
            "egocar_visible": False,
            "is_fisheye": False,
        },
        # Fisheye
        7: {
            "camera_name": "fisheye_left",
            "original_size": (1536, 1920),
            "egocar_visible": False,
            "is_fisheye": True,
        },
        8: {
            "camera_name": "fisheye_rear",
            "original_size": (1536, 1920),
            "egocar_visible": False,
            "is_fisheye": True,
        },
        9: {
            "camera_name": "fisheye_front",
            "original_size": (1536, 1920),
            "egocar_visible": False,
            "is_fisheye": True,
        },
        10: {
            "camera_name": "fisheye_right",
            "original_size": (1536, 1920),
            "egocar_visible": False,
            "is_fisheye": True,
        },
    },
    "qcraft": {  # 轻舟，相机名称有修改
        0: {
            "camera_name": "front_wide_110",  # 广角前视 FOV110
            "original_size": (512, 1024),
            "egocar_visible": True,
            "is_fisheye": False,
        },
        1: {
            "camera_name": "front_wide_60",  # 广角前视 FOV60
            "original_size": (512, 1024),
            "egocar_visible": False,
            "is_fisheye": False,
        },
        2: {
            "camera_name": "front_tele_30",  # 长焦前视 FOV30
            "original_size": (512, 1024),
            "egocar_visible": False,
            "is_fisheye": False,
        },
        3: {
            "camera_name": "front_tele_15",  # 长焦前视 FOV15
            "original_size": (512, 1024),
            "egocar_visible": False,
            "is_fisheye": False,
        },
        4: {
            "camera_name": "front_wide_left_60",  # 广角左前 FOV60
            "original_size": (512, 1024),
            "egocar_visible": True,
            "is_fisheye": False,
        },
        5: {
            "camera_name": "front_left_99",  # 左前 FOV99
            "original_size": (512, 1024),
            "egocar_visible": True,
            "is_fisheye": False,
        },
        6: {
            "camera_name": "rear_left_99",  # 左后 FOV99
            "original_size": (512, 1024),
            "egocar_visible": True,
            "is_fisheye": False,
        },
        7: {
            "camera_name": "rear_left_30",  # 左后 FOV30
            "original_size": (256, 512),
            "egocar_visible": True,
            "is_fisheye": False,
        },
        8: {
            "camera_name": "front_wide_right_60",  # 广角右前 FOV60
            "original_size": (512, 1024),
            "egocar_visible": True,
            "is_fisheye": False,
        },
        9: {
            "camera_name": "front_right_99",  # 右前 FOV99
            "original_size": (512, 1024),
            "egocar_visible": True,
            "is_fisheye": False,
        },
        10: {
            "camera_name": "rear_right_99",  # 右后 FOV99
            "original_size": (512, 1024),
            "egocar_visible": True,
            "is_fisheye": False,
        },
        11: {
            "camera_name": "rear_right_30",  # 右后 FOV30
            "original_size": (256, 512),
            "egocar_visible": True,
            "is_fisheye": False,
        },
        12: {
            "camera_name": "rear_50",  # 后视 FOV50
            "original_size": (512, 1024),
            "egocar_visible": False,
            "is_fisheye": False,
        },
    },
    "waymo": {
        0: {
            "camera_name": "front_camera",
            "original_size": (1280, 1920),
            "egocar_visible": False,
        },
        1: {
            "camera_name": "front_left_camera",
            "original_size": (1280, 1920),
            "egocar_visible": False,
        },
        2: {
            "camera_name": "front_right_camera",
            "original_size": (1280, 1920),
            "egocar_visible": False,
        },
        3: {
            "camera_name": "left_camera",
            "original_size": (866, 1920),
            "egocar_visible": False,
        },
        4: {
            "camera_name": "right_camera",
            "original_size": (866, 1920),
            "egocar_visible": False,
        },
    },
    "pandaset": {
        0: {
            "camera_name": "front_camera",
            "original_size": (1080, 1920),
            "egocar_visible": False,
        },
        1: {
            "camera_name": "front_left_camera",
            "original_size": (1080, 1920),
            "egocar_visible": False,
        },
        2: {
            "camera_name": "front_right_camera",
            "original_size": (1080, 1920),
            "egocar_visible": False,
        },
        3: {
            "camera_name": "left_camera",
            "original_size": (1080, 1920),
            "egocar_visible": False,
        },
        4: {
            "camera_name": "right_camera",
            "original_size": (1080, 1920),
            "egocar_visible": False,
        },
        5: {
            "camera_name": "back_camera",
            "original_size": (1080, 1920),
            "egocar_visible": True,
        },
    },
    "argoverse": {
        0: {
            "camera_name": "ring_front_center",
            "original_size": (2048, 1550),
            "egocar_visible": True,
        },
        1: {
            "camera_name": "ring_front_left",
            "original_size": (1550, 2048),
            "egocar_visible": False,
        },
        2: {
            "camera_name": "ring_front_right",
            "original_size": (1550, 2048),
            "egocar_visible": False,
        },
        3: {
            "camera_name": "ring_side_left",
            "original_size": (1550, 2048),
            "egocar_visible": False,
        },
        4: {
            "camera_name": "ring_side_right",
            "original_size": (1550, 2048),
            "egocar_visible": False,
        },
        5: {
            "camera_name": "ring_rear_left",
            "original_size": (1550, 2048),
            "egocar_visible": True,
        },
        6: {
            "camera_name": "ring_rear_right",
            "original_size": (1550, 2048),
            "egocar_visible": True,
        },
    },
    "nuscenes": {
        0: {
            "camera_name": "CAM_FRONT",
            "original_size": (900, 1600),
            "egocar_visible": False,
        },
        1: {
            "camera_name": "CAM_FRONT_LEFT",
            "original_size": (900, 1600),
            "egocar_visible": False,
        },
        2: {
            "camera_name": "CAM_FRONT_RIGHT",
            "original_size": (900, 1600),
            "egocar_visible": False,
        },
        3: {
            "camera_name": "CAM_BACK_LEFT",
            "original_size": (900, 1600),
            "egocar_visible": False,
        },
        4: {
            "camera_name": "CAM_BACK_RIGHT",
            "original_size": (900, 1600),
            "egocar_visible": False,
        },
        5: {
            "camera_name": "CAM_BACK",
            "original_size": (900, 1600),
            "egocar_visible": True,
        },
    },
    "kitti": {
        0: {
            "camera_name": "CAM_LEFT",
            "original_size": (375, 1242),
            "egocar_visible": False,
        },
        1: {
            "camera_name": "CAM_RIGHT",
            "original_size": (375, 1242),
            "egocar_visible": False,
        },
    },
    "nuplan": {
        0: {
            "camera_name": "CAM_F0",
            "original_size": (1080, 1920),
            "egocar_visible": False,
        },
        1: {
            "camera_name": "CAM_L0",
            "original_size": (1080, 1920),
            "egocar_visible": True,
        },
        2: {
            "camera_name": "CAM_R0",
            "original_size": (1080, 1920),
            "egocar_visible": True,
        },
        3: {
            "camera_name": "CAM_L1",
            "original_size": (1080, 1920),
            "egocar_visible": False,
        },
        4: {
            "camera_name": "CAM_R1",
            "original_size": (1080, 1920),
            "egocar_visible": False,
        },
        5: {
            "camera_name": "CAM_L2",
            "original_size": (1080, 1920),
            "egocar_visible": True,
        },
        6: {
            "camera_name": "CAM_R2",
            "original_size": (1080, 1920),
            "egocar_visible": True,
        },
        7: {
            "camera_name": "CAM_B0",
            "original_size": (1080, 1920),
            "egocar_visible": False,
        },
    },
}