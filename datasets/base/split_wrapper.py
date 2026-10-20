from typing import List, Tuple
import torch
from enum import IntEnum

from .pixel_source import ScenePixelSource


class ModelType(IntEnum):
    RigidNodes = 0
    SMPLNodes = 1
    DeformableNodes = 2
    TrafficLightNodes = 3


class SplitWrapper(torch.utils.data.Dataset):

    # a sufficiently large number to make sure we don't run out of data
    _num_iters = 1000000

    def __init__(
        self,
        datasource: ScenePixelSource,
        split_indices: List[int] = None,
        split: str = "train",
    ):
        super().__init__()
        self.datasource = datasource
        self.split_indices = split_indices
        self.split = split

    def get_image(self, idx, camera_downscale) -> Tuple[dict, dict]:
        downscale_factor = 1 / camera_downscale * self.datasource.downscale_factor
        self.datasource.update_downscale_factor(downscale_factor)
        image_infos, cam_infos = self.datasource.get_image(self.split_indices[idx])
        self.datasource.reset_downscale_factor()
        return image_infos, cam_infos

    def next(self, camera_downscale, propose_by_camera=True) -> Tuple[dict, dict]:
        assert self.split == "train", "Only train split supports next()"
        
        if (any(ModelType.TrafficLightNodes in model_type for model_type in self.datasource.instances_model_types)
            and propose_by_camera):
            # Propose an image index based on camera weights
            img_idx = self.datasource.propose_training_image_by_camera(
                candidate_indices=self.split_indices
            )
        else:
            img_idx = self.datasource.propose_training_image(
                candidate_indices=self.split_indices
            )
        downscale_factor = 1 / camera_downscale * self.datasource.downscale_factor
        self.datasource.update_downscale_factor(downscale_factor)
        image_infos, cam_infos = self.datasource.get_image(img_idx)
        self.datasource.reset_downscale_factor()
        
        return image_infos, cam_infos
    
    def __getitem__(self, idx) -> dict:
        return self.get_image(idx, camera_downscale=1.0)

    def __len__(self) -> int:
        return len(self.split_indices)

    @property
    def num_iters(self) -> int:
        return self._num_iters

    def set_num_iters(self, num_iters) -> None:
        self._num_iters = num_iters
    
    def front_camera_trajectory(self):
        return self.datasource.front_camera_trajectory
