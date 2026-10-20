from typing import Dict, List, Tuple, Optional, Union
from omegaconf import OmegaConf
import random
import logging
import os
import torch
import numpy as np
from torch.nn import Parameter

from models.human_body import phalp_colors, SMPLTemplate, get_on_mesh_init_geo_values, batch_rigid_transform, quaternion_to_matrix
from models.gaussians.basics import *
from models.nodes.rigid import RigidNodes
from models.gaussians.vanilla import VanillaGaussians
from pytorch3d.ops import knn_points

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from plyfile import PlyData, PlyElement

CACHE_DIR = os.path.join(os.environ.get("HOME"), ".cache")
CACHE_DIR_4DHUMANS = os.path.join(CACHE_DIR, "4DHumans")

RGB_tuples = torch.tensor(np.vstack([phalp_colors] * 10), dtype=torch.float32) / 255.0
logger = logging.getLogger()

class SMPLNodes(RigidNodes):
    def __init__(
        self,
        **kwargs
    ):
        self.smpl_points_num = 6890
        super().__init__(**kwargs)
        
        self.use_voxel_deformer=self.ctrl_cfg.use_voxel_deformer
        # overide here, because we use only one dimension for scale
        if self.ball_gaussians:
            self._scales = torch.zeros(1, 1, device=self.device)
        
    @property
    def num_instances(self):
        return self.instances_fv.shape[1]
    @property
    def num_frames(self):
        return self.instances_fv.shape[0]

    def _has_initialized_smpl_instances(self) -> bool:
        required_attrs = (
            "template",
            "point_ids",
            "instances_fv",  #fv为判断该帧是否有效（可见）
            "instances_trans",
            "instances_quats",
            "smpl_qauts",
            "instances_size",
        )
        if not all(hasattr(self, attr) for attr in required_attrs):
            return False
        return self.instances_fv.shape[1] > 0

    def load_ply_as_gs(
        self,
        ply_path: str,
        device: torch.device = torch.device("cpu"),
        sh_degree: int = 3,
        scale_factor: float = 1.0,
    ):
        # 从 PLY 中读取 Gaussian 参数；如果文件里同时保存了关节、蒙皮权重或 voxel 模板，也一起组装到返回资产里。
        asset = super().load_ply_as_gs(
            ply_path=ply_path,
            device=device,
            sh_degree=sh_degree,
            scale_factor=scale_factor,
        )

        plydata = PlyData.read(ply_path)
        vertex = plydata["vertex"]
        vertex_names = vertex.data.dtype.names

        weight_names = sorted(
            [name for name in vertex_names if name.startswith("w_")],
            key=lambda s: int(s.split("_")[-1]),
        )
        joint_element = plydata["joint"] if any(el.name == "joint" for el in plydata.elements) else None

        if weight_names or joint_element is not None:
            if not weight_names or joint_element is None:
                raise RuntimeError(
                    f"Incomplete SMPL template in ply {ply_path}: both joint element and w_* fields are required"
                )

            num_points = vertex.count
            weights = torch.stack(
                [torch.from_numpy(vertex[name]).to(torch.float32) for name in weight_names],
                dim=1,
            ).to(device=device)
            if weights.shape[0] != num_points:
                raise RuntimeError(
                    f"Invalid SMPL weights in ply {ply_path}: {weights.shape[0]} vs expected {num_points}"
                )

            joint_names = joint_element.data.dtype.names
            required_joint_fields = ("x", "y", "z")
            if not all(field in joint_names for field in required_joint_fields):
                raise RuntimeError(
                    f"Invalid joint element in ply {ply_path}: missing one of {required_joint_fields}"
                )
            joints = torch.stack(
                [
                    torch.from_numpy(joint_element[field]).to(torch.float32)
                    for field in required_joint_fields
                ],
                dim=1,
            ).to(device=device)

            asset["smpl_template"] = {
                "J_canonical": joints,
                "W": weights,
            }

            if self.use_voxel_deformer:
                voxel_element_names = {el.name for el in plydata.elements}
                has_voxel_template = {
                    "voxel_lbs_base",
                    "voxel_w_correction",
                    "voxel_meta",
                }.issubset(voxel_element_names)
                if has_voxel_template:
                    lbs_element = plydata["voxel_lbs_base"]
                    corr_element = plydata["voxel_w_correction"]
                    meta_element = plydata["voxel_meta"]

                    lbs_names = lbs_element.data.dtype.names
                    corr_names = corr_element.data.dtype.names
                    meta_names = meta_element.data.dtype.names
                    if lbs_names != ("value",) or corr_names != ("value",):
                        raise RuntimeError(
                            f"Invalid voxel template in ply {ply_path}: voxel elements must contain only 'value'"
                        )
                    required_meta_fields = (
                        "offset_0",
                        "offset_1",
                        "offset_2",
                        "scale_0",
                        "lbs_ndim",
                        "lbs_dim_0",
                        "lbs_dim_1",
                        "lbs_dim_2",
                        "lbs_dim_3",
                        "lbs_dim_4",
                        "corr_ndim",
                        "corr_dim_0",
                        "corr_dim_1",
                        "corr_dim_2",
                        "corr_dim_3",
                        "corr_dim_4",
                    )
                    if not all(field in meta_names for field in required_meta_fields):
                        raise RuntimeError(
                            f"Invalid voxel meta in ply {ply_path}: missing one of {required_meta_fields}"
                        )
                    if meta_element.count != 1:
                        raise RuntimeError(
                            f"Invalid voxel meta in ply {ply_path}: expected exactly one row, got {meta_element.count}"
                        )

                    lbs_flat = torch.from_numpy(lbs_element["value"].astype(np.float32)).to(device=device)
                    corr_flat = torch.from_numpy(corr_element["value"].astype(np.float32)).to(device=device)
                    meta_row = meta_element.data[0]
                    lbs_ndim = int(meta_row["lbs_ndim"])
                    corr_ndim = int(meta_row["corr_ndim"])
                    if lbs_ndim < 1 or lbs_ndim > 5:
                        raise RuntimeError(f"Invalid voxel lbs ndim in ply {ply_path}: {lbs_ndim}")
                    if corr_ndim < 1 or corr_ndim > 5:
                        raise RuntimeError(f"Invalid voxel correction ndim in ply {ply_path}: {corr_ndim}")
                    expected_lbs_shape = tuple(int(meta_row[f"lbs_dim_{dim}"]) for dim in range(lbs_ndim))
                    expected_corr_shape = tuple(int(meta_row[f"corr_dim_{dim}"]) for dim in range(corr_ndim))
                    if lbs_flat.numel() != int(np.prod(expected_lbs_shape)):
                        raise RuntimeError(
                            f"Invalid voxel lbs size in ply {ply_path}: {lbs_flat.numel()} vs expected {int(np.prod(expected_lbs_shape))}"
                        )
                    if corr_flat.numel() != int(np.prod(expected_corr_shape)):
                        raise RuntimeError(
                            f"Invalid voxel correction size in ply {ply_path}: {corr_flat.numel()} vs expected {int(np.prod(expected_corr_shape))}"
                        )

                    offset = torch.tensor(
                        [[meta_row["offset_0"], meta_row["offset_1"], meta_row["offset_2"]]],
                        dtype=torch.float32,
                        device=device,
                    )
                    scale = torch.tensor(
                        [[meta_row["scale_0"]]],
                        dtype=torch.float32,
                        device=device,
                    )

                    asset["smpl_template"]["voxel_deformer"] = {
                        "lbs_voxel_base": lbs_flat.reshape(expected_lbs_shape),
                        "voxel_w_correction": corr_flat.reshape(expected_corr_shape),
                        "offset": offset,
                        "scale": scale,
                    }

        return asset
    
    def create_from_pcd(self, instance_pts_dict: Dict[str, torch.Tensor]) -> None:
        """
        instance_pts_dict: {
            id in dataset: {
                "class_name": str,
                "pts": torch.Tensor, (N, 3)
                "colors": torch.Tensor, (N, 3)
                "poses": torch.Tensor, (num_frame, 4, 4)
                "size": torch.Tensor, (3, )
                "frame_info": torch.Tensor, (num_frame)
                "num_pts": int,
            },
        }
        """
        """
        For version 1:
            we simplified gaussian properties:
            - means: (Frame_N, N, 3), Not Optimized
            - scales: (N, 1), Optimized, we use gaussians with the same scale on xyz
            - quats: (N, 4), Not Optimized
            - features_dc: (N, 3), Optimized
            - features_rest: (N, num_sh_bases, 3), Optimized
            - opacities: (N, 1), Optimized
        """
        # collect all instances
        smpl_betas, smpl_qauts = [], []
        instances_quats, instances_trans, instances_size = [], [], []
        instances_fv, point_ids = [], []
        # instances_pts, instances_colors = [], []
        for id_in_model, (id_in_dataset, v) in enumerate(instance_pts_dict.items()):
            smpl_qauts.append(v["smpl_quats"][:, 1:, :].unsqueeze(1))
            instances_quats.append(v["smpl_quats"][:, 0, :].unsqueeze(1))
            instances_trans.append(v["smpl_trans"].unsqueeze(1))
            instances_fv.append(v["frame_info"].unsqueeze(1))
            smpl_betas.append(v["smpl_betas"].unsqueeze(0))
            instances_size.append(v["size"])
            # instances_pts.append(v["pts"])
            # instances_colors.append(v["colors"])
            point_ids.append(torch.full((self.smpl_points_num, 1), id_in_model, dtype=torch.long))
        
        smpl_qauts = torch.cat(smpl_qauts, dim=1).to(self.device)                # (num_frame, num_instances, 23, 4)
        instances_quats = torch.cat(instances_quats, dim=1).to(self.device)      # (num_frame, num_instances, 4)
        instances_trans = torch.cat(instances_trans, dim=1).to(self.device)      # (num_frame, num_instances, 3)
        instances_fv = torch.cat(instances_fv, dim=1).to(self.device)            # (num_frame, num_instances)
        smpl_betas = torch.cat(smpl_betas, dim=0).to(self.device)                # (num_instances, 10)
        instances_size = torch.stack(instances_size).to(self.device)             # (num_instances, 3)
        point_ids = torch.cat(point_ids, dim=0).to(self.device)                  # (self.smpl_points_num*num_instances, 1)
        self.instances_fv    = instances_fv                            # (num_frame, num_instances)
    
        self.template = SMPLTemplate(
            smpl_model_path=f'{CACHE_DIR_4DHUMANS}/data/smpl/SMPL_NEUTRAL.pkl',
            num_human=smpl_betas.shape[0],
            init_beta=smpl_betas,
            cano_pose_type="da_pose",
            use_voxel_deformer=self.use_voxel_deformer
        )
        if self.use_voxel_deformer:
            self.template.voxel_deformer.enable_voxel_correction()
        
        opacity_init_value = torch.tensor(self.ctrl_cfg.opacity_init_value)
        x, q, s, o = get_on_mesh_init_geo_values(
            self.template,
            opacity_init_logit=torch.logit(opacity_init_value),
        )
        if self.ball_gaussians:
            s = s.mean(-1, keepdim=True)
        x = x.to(dtype=torch.float32, device=self.device)
        s = s.to(dtype=torch.float32, device=self.device)
        q = q.to(dtype=torch.float32, device=self.device)
        o = o.to(dtype=torch.float32, device=self.device)
        
        # knn init
        self.update_knn(x)
        
        if self.ctrl_cfg.constrain_xyz_offset:
            self.on_mesh_x = x.clone()
        
        # NOTE: In the future, we will also use colors of lidars to get the initialization of colors
        self.template = self.template.to(self.device)
        for fi in range(self.num_frames):
            instance_mask = instances_fv[fi]
            if instance_mask.sum() == 0:
                continue

            theta = torch.cat(
                (instances_quats[fi].unsqueeze(1), smpl_qauts[fi]), dim=1
            )
            masked_theta = theta[instance_mask]
            masked_theta = masked_theta / masked_theta.norm(dim=-1, keepdim=True)
            W, A = self.template(
                masked_theta = masked_theta, 
                instances_mask = instance_mask
            )
            T = torch.einsum("bnj, bjrc -> bnrc", W, A)
            R = T[:, :, :3, :3] # [N, 3, 3]
            t = T[:, :, :3, 3]  # [N, 3]
            
            reshaped_means = x.reshape(self.num_instances, self.smpl_points_num, 3)
            deformed_means = torch.einsum(
                "bnij,bnj->bni", R, reshaped_means[instance_mask]         
            ) + t  # [N, 6890, 3]
            bbox_min = deformed_means.min(dim=1)[0]
            bbox_max = deformed_means.max(dim=1)[0]
            local_shift = (bbox_min + bbox_max) / 2
            instances_trans[fi, instance_mask] = instances_trans[fi, instance_mask] - local_shift
        
        self._means     = Parameter(x, requires_grad=not self.ctrl_cfg.freeze_x)
        self._scales    = Parameter(s, requires_grad=not self.ctrl_cfg.freeze_s)
        self._quats     = Parameter(q, requires_grad=not self.ctrl_cfg.freeze_q)
        self._opacities = Parameter(o, requires_grad=not self.ctrl_cfg.freeze_o)
        
        self.instances_quats = Parameter(instances_quats.unsqueeze(2)) # (num_frame, num_instances, 1, 4)
        self.instances_trans = Parameter(instances_trans)              # (num_frame, num_instances, 3)
        self.smpl_qauts      = Parameter(smpl_qauts)                   # (num_frame, num_instances, 23, 4)
        self.instances_size  = instances_size                          # (num_instances, 3)
        self.point_ids       = point_ids                               # (self.smpl_points_num*num_instances, 1)
        
        dim_sh = num_sh_bases(self.sh_degree)
        # NOTE: init_colors actually is for visualization, we use random color here
        # init_colors = RGB_tuples[self.point_ids.squeeze().cpu()].to(self.device)
        init_colors  = torch.rand((self.num_points, 3), device=self.device)
        fused_color  = RGB2SH(init_colors) # float range [0, 1] 
        shs = torch.zeros((fused_color.shape[0], dim_sh, 3)).float().to(self.device)
        if self.sh_degree > 0:
            shs[:, 0, :3] = fused_color
            shs[:, 1:, 3:] = 0.0
        else:
            shs[:, 0, :3] = torch.logit(init_colors, eps=1e-10)
        self._features_dc   = Parameter(shs[:, 0, :],  requires_grad=not self.ctrl_cfg.freeze_shs_dc)
        self._features_rest = Parameter(shs[:, 1:, :], requires_grad=not self.ctrl_cfg.freeze_shs_rest)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = self.get_gaussian_param_groups()
        param_groups[self.class_prefix+"ins_rotation"] = [self.instances_quats]
        param_groups[self.class_prefix+"ins_translation"] = [self.instances_trans]
        param_groups[self.class_prefix+"smpl_rotation"] = [self.smpl_qauts]
        if self.use_voxel_deformer:
            param_groups[self.class_prefix+"w_dc_vox"] = [self.template.voxel_deformer.voxel_w_correction]
        #     param_groups[self.class_prefix+"w_rest_vox"] = [self.template.voxel_deformer.additional_correction]
        return param_groups
    
    @property
    def num_points(self):
        return self._means.shape[0]

    def update_knn(self, x: torch.Tensor) -> None:
        reshaped_x = x.reshape(self.num_instances, self.smpl_points_num, 3)
        _, nn_ind, _ = knn_points(reshaped_x, reshaped_x, K=self.ctrl_cfg.knn_neighbors, return_nn=False)
        self.nn_ind = nn_ind
    
    def postprocess_per_train_step(
        self,
        step: int,
        optimizer: torch.optim.Optimizer,
        radii: torch.Tensor,
        xys_grad: torch.Tensor,
        last_size: int,
    ) -> None:
        self.radii = radii
        self.xys_grad = xys_grad
        knn_update_interval = self.ctrl_cfg.get("knn_update_interval", 1000000)
        if self.step % knn_update_interval == 0:
            self.update_knn(self._means)
    
    def transform_means(self, means: torch.Tensor) -> torch.Tensor:
        """
        transform the means of instances to world space
        according to the pose at the current frame
        """
        assert means.shape[0] == self.point_ids.shape[0], \
            "its a bug here, we need to pass the mask for points_ids"
        instance_mask = self.instances_fv[self.cur_frame]
        if self.in_test_set and (
            self.cur_frame - 1 > 0 and self.cur_frame + 1 < self.num_frames
        ):
            _prev_masked_theta = torch.cat((self.instances_quats[self.cur_frame - 1], self.smpl_qauts[self.cur_frame - 1]), dim=1)[instance_mask]
            _next_masked_theta = torch.cat((self.instances_quats[self.cur_frame + 1], self.smpl_qauts[self.cur_frame + 1]), dim=1)[instance_mask]
            _cur_masked_theta = torch.cat((self.instances_quats[self.cur_frame], self.smpl_qauts[self.cur_frame]), dim=1)[instance_mask]
            interpolated_theta = interpolate_quats(_prev_masked_theta, _next_masked_theta)
            
            inter_valid_mask = self.instances_fv[self.cur_frame - 1, instance_mask] & self.instances_fv[self.cur_frame + 1, instance_mask]
            masked_theta = torch.where(
                inter_valid_mask[:, None, None], interpolated_theta, _cur_masked_theta
            )
        else:
            theta = torch.cat(
                (self.instances_quats[self.cur_frame], self.smpl_qauts[self.cur_frame]), dim=1
            )
            masked_theta = theta[instance_mask]
        masked_theta = self.quat_act(masked_theta)
        W, A = self.template(
            masked_theta = masked_theta, 
            instances_mask = instance_mask,
            xyz_canonical = means.reshape(self.num_instances, self.smpl_points_num, 3) if self.use_voxel_deformer else None
        )
        T = torch.einsum("bnj, bjrc -> bnrc", W, A)
        R = T[:, :, :3, :3] # [N, 3, 3]
        t = T[:, :, :3, 3]  # [N, 3]
        
        reshaped_means = means.reshape(self.num_instances, self.smpl_points_num, 3)
        deformed_means = torch.einsum(
            "bnij,bnj->bni", R, reshaped_means[instance_mask]         
        ) + t  # [N, 6890, 3]
        
        means_container = torch.zeros_like(reshaped_means)
        means_container.index_add_(0, instance_mask.nonzero().squeeze(), deformed_means)
        means_container = means_container.reshape(-1, 3)

        if self.in_test_set and (
            self.cur_frame - 1 > 0 and self.cur_frame + 1 < self.num_frames
        ):
            _prev_ins_trans = self.instances_trans[self.cur_frame - 1]
            _next_ins_trans = self.instances_trans[self.cur_frame + 1]
            _cur_ins_trans = self.instances_trans[self.cur_frame]
            interpolated_trans = (_prev_ins_trans + _next_ins_trans) * 0.5
            
            inter_valid_mask = self.instances_fv[self.cur_frame - 1] & self.instances_fv[self.cur_frame + 1]
            trans_cur_frame = torch.where(
                inter_valid_mask[:, None], interpolated_trans, _cur_ins_trans
            )
        else:
            trans_cur_frame = self.instances_trans[self.cur_frame] # (num_instances, 3)
        trans_per_pts = trans_cur_frame[self.point_ids[..., 0]]
        
        # transform the means to world space
        means_container += trans_per_pts
        return means_container
    
    def transform_means_and_quats(self, means: torch.Tensor, quats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        transform the means and quats of gaussians to world space
        according to the pose at the current frame
        """
        assert means.shape[0] == self.point_ids.shape[0], \
            "its a bug here, we need to pass the mask for points_ids"
        instance_mask = self.instances_fv[self.cur_frame]
        if self.in_test_set and (
            self.cur_frame - 1 > 0 and self.cur_frame + 1 < self.num_frames
        ):
            _prev_masked_theta = torch.cat((self.instances_quats[self.cur_frame - 1], self.smpl_qauts[self.cur_frame - 1]), dim=1)[instance_mask]
            _next_masked_theta = torch.cat((self.instances_quats[self.cur_frame + 1], self.smpl_qauts[self.cur_frame + 1]), dim=1)[instance_mask]
            _cur_masked_theta = torch.cat((self.instances_quats[self.cur_frame], self.smpl_qauts[self.cur_frame]), dim=1)[instance_mask]
            interpolated_theta = interpolate_quats(_prev_masked_theta, _next_masked_theta)
            
            inter_valid_mask = self.instances_fv[self.cur_frame - 1, instance_mask] & self.instances_fv[self.cur_frame + 1, instance_mask]
            masked_theta = torch.where(
                inter_valid_mask[:, None, None], interpolated_theta, _cur_masked_theta
            )
        else:
            theta = torch.cat(
                (self.instances_quats[self.cur_frame], self.smpl_qauts[self.cur_frame]), dim=1
            )
            masked_theta = theta[instance_mask]
        masked_theta = self.quat_act(masked_theta)
        W, A = self.template(
            masked_theta = masked_theta, 
            instances_mask = instance_mask,
            xyz_canonical = means.reshape(self.num_instances, self.smpl_points_num, 3) if self.use_voxel_deformer else None
        )
        T = torch.einsum("bnj, bjrc -> bnrc", W, A)
        R = T[:, :, :3, :3] # [N, 3, 3]
        t = T[:, :, :3, 3]  # [N, 3]
        
        reshaped_means = means.reshape(self.num_instances, self.smpl_points_num, 3)
        deformed_means = torch.einsum(
            "bnij,bnj->bni", R, reshaped_means[instance_mask]         
        ) + t  # [N, 6890, 3]
        
        means_container = torch.zeros_like(reshaped_means)
        means_container.index_add_(0, instance_mask.nonzero().squeeze(), deformed_means)
        means_container = means_container.reshape(-1, 3)

        if self.in_test_set and (
            self.cur_frame - 1 > 0 and self.cur_frame + 1 < self.num_frames
        ):
            _prev_ins_trans = self.instances_trans[self.cur_frame - 1]
            _next_ins_trans = self.instances_trans[self.cur_frame + 1]
            _cur_ins_trans = self.instances_trans[self.cur_frame]
            interpolated_trans = (_prev_ins_trans + _next_ins_trans) * 0.5
            
            inter_valid_mask = self.instances_fv[self.cur_frame - 1] & self.instances_fv[self.cur_frame + 1]
            trans_cur_frame = torch.where(
                inter_valid_mask[:, None], interpolated_trans, _cur_ins_trans
            )
        else:
            trans_cur_frame = self.instances_trans[self.cur_frame] # (num_instances, 3)
        trans_per_pts = trans_cur_frame[self.point_ids[..., 0]]
        
        # transform the means to world space
        means_container += trans_per_pts
        
        reshaped_quats = quats.reshape(self.num_instances, self.smpl_points_num, 4)
        R_quats = matrix_to_quaternion(R)
        deformed_quats = quat_mult(
            self.quat_act(R_quats),
            self.quat_act(reshaped_quats[instance_mask])
        )
        quats_container = torch.zeros_like(reshaped_quats)
        quats_container.index_add_(0, instance_mask.nonzero().squeeze(), deformed_quats)
        # fill other with [1, 0, 0, 0]
        quats_container.index_add_(0, (~instance_mask).nonzero().squeeze(), torch.tensor([[[1., 0., 0., 0.]]], device=self.device).repeat((~instance_mask).sum(), 6890, 1))
        quats_container = quats_container.reshape(-1, 4)
        return means_container, quats_container
    
    def get_gaussians(self, cam: dataclass_camera) -> Dict[str, torch.Tensor]:
        filter_mask = torch.ones_like(self._means[:, 0], dtype=torch.bool)
        self.filter_mask = filter_mask
        # NOTE: hack here, need to consider a gaussian filter for efficient rendering
        
        instance_mask = self.instances_fv[self.cur_frame]
        if instance_mask.sum() == 0:
            return None
                
        if self.ball_gaussians:
            world_means = self.transform_means(self._means)
            world_quats = self._quats
        else:
            world_means, world_quats = self.transform_means_and_quats(self._means, self._quats)
        
        # get colors of gaussians
        colors = torch.cat((self._features_dc[:, None, :], self._features_rest), dim=1)
        if self.sh_degree > 0:
            viewdirs = world_means.detach() - cam.camtoworlds.data[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.ctrl_cfg.sh_degree_interval, self.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors)
            rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
        else:
            rgbs = torch.sigmoid(colors[:, 0, :])
        
        valid_mask = self.get_pts_valid_mask()
            
        activated_opacities = self.get_opacity * valid_mask.float().unsqueeze(-1)
        if self.ball_gaussians:
            activated_scales = torch.exp(self._scales.repeat(1, 3))
        else:
            activated_scales = torch.exp(self._scales)
        activated_rotations = self.quat_act(world_quats)
        actovated_colors = rgbs
        
        # collect gaussians information
        gs_dict = dict(
            _means=world_means[filter_mask],
            _opacities=activated_opacities[filter_mask],
            _rgbs=actovated_colors[filter_mask],
            _scales=activated_scales[filter_mask],
            _quats=activated_rotations[filter_mask],
        )
        # check nan in gs_dict
        for k, v in gs_dict.items():
            if torch.isnan(v).any():
                raise RuntimeError(f"NaN detected in gaussian {k} at step {self.step}")
            if torch.isinf(v).any():
                raise RuntimeError(f"Inf detected in gaussian {k} at step {self.step}")
        
        self._gs_cache = {
            "_scales": activated_scales[filter_mask],
        }
        return gs_dict

    def generate_legend_image(
        self,
        ids: torch.Tensor,        # shape (N,)
        colors: torch.Tensor,     # shape (N,3) float 0~1
        image_output_pth: str
        ) -> np.ndarray:
        """
        依据 ids 与 colors 生成 legend png，并返回 numpy(H,W,3,uint8)。

        若 save_path 给定则落盘。
        """
        ids_np     = ids.to('cpu').numpy()
        colors_np  = colors.to('cpu').numpy()
        N          = len(ids_np)

        fig_h = max(2, N * 0.35)              # 高度随类别数增
        fig, ax = plt.subplots(figsize=(2.0, fig_h), dpi=100)

        # 反向画让 0 在最下方（完全可按自己习惯）
        for row, (idx, col) in enumerate(zip(ids_np[::-1], colors_np[::-1])):
            # 左边色块
            ax.add_patch(
                patches.Rectangle(
                    (0, row), width=1.0, height=1.0, facecolor=col, edgecolor="none"
                )
            )
            # 右边文字
            ax.text(
                1.2, row + 0.5, str(int(idx)),
                va="center", ha="left", fontsize=10, color="black"
            )

        ax.set_xlim(0, 3)
        ax.set_ylim(0, N)
        ax.axis("off")
        fig.tight_layout(pad=0)

        if image_output_pth is not None and not os.path.exists(image_output_pth):
            fig.savefig(image_output_pth, bbox_inches="tight", pad_inches=0)
            print(f"[legend] saved image -> {image_output_pth}")

        plt.close(fig)

    def color_legned_gaussians(
        self,
        image_output_pth: str
        ) -> Dict[str, torch.Tensor]: 
        unique_vals, counts = torch.unique(self.point_ids, return_counts=True)

        # To find instance
        N = len(unique_vals)
        cmap = cm.get_cmap('gist_ncar')          # 连续调色板
        colors_np = cmap(np.linspace(0, 1, N, endpoint=False))[:, :3]   # (N,3)  float 0~1

        colors = torch.from_numpy(colors_np).to(
            dtype=self._features_dc.dtype,          # 通常是 torch.float32 或 float16
            device=self._features_dc.device         # same GPU
        )
        
        # colors_sh_format = colors - 0.5
        colors_logit_format = torch.logit(torch.clamp(colors, 1e-6, 1-1e-6))

        for index, unique_val in enumerate(unique_vals):
            instance_mask = (self.point_ids.squeeze(-1) == unique_val)
            self._features_dc[instance_mask, :] = colors_logit_format[index]
        self.ctrl_cfg.sh_degree=0

        self.generate_legend_image(unique_vals, colors, image_output_pth)



    def compute_reg_loss(self):
        loss_dict = super().compute_reg_loss()

        instance_mask = self.instances_fv[self.cur_frame]
        if instance_mask.sum() == 0:
            return loss_dict
        
        # temporal smooth regularization
        temporal_smooth_reg = self.reg_cfg.get("temporal_smooth_reg", None)
        if temporal_smooth_reg is not None:
            joint_smooth_reg = temporal_smooth_reg.get("joint_smooth", None)
            if joint_smooth_reg is not None:
                if self.cur_frame >= 1 and self.cur_frame < self.num_frames - 1:
                    valid_mask = (
                        self.instances_fv[self.cur_frame - 1] & \
                        self.instances_fv[self.cur_frame + 1] & \
                        self.instances_fv[self.cur_frame]
                    )
                    cur_theta = torch.cat(
                        (self.instances_quats[self.cur_frame], self.smpl_qauts[self.cur_frame]), dim=1
                    )[valid_mask]
                    next_theta = torch.cat(
                        (self.instances_quats[self.cur_frame + 1], self.smpl_qauts[self.cur_frame + 1]), dim=1
                    )[valid_mask]
                    prev_theta = torch.cat(
                        (self.instances_quats[self.cur_frame - 1], self.smpl_qauts[self.cur_frame - 1]), dim=1
                    )[valid_mask]
                    thetas = torch.vstack([prev_theta, cur_theta, next_theta])
                    thetas = self.quat_act(thetas)
                    J_transformed, _ = batch_rigid_transform(
                        quaternion_to_matrix(thetas),
                        self.template.J_canonical[valid_mask].repeat(3, 1, 1),
                        self.template._template_layer.parents,
                    )
                    
                    cur_trans = self.instances_trans[self.cur_frame, valid_mask]
                    next_trans = self.instances_trans[self.cur_frame + 1, valid_mask]
                    prev_trans = self.instances_trans[self.cur_frame - 1, valid_mask]
                    trans = torch.vstack([prev_trans, cur_trans, next_trans])
                    J_transformed += trans.unsqueeze(-2)
                    J_transformed = J_transformed.reshape(3, -1, 24, 3)
                    
                    velocity_prev = (J_transformed[1] - J_transformed[0])
                    velocity_next = (J_transformed[2] - J_transformed[1])
                    # l2 loss
                    loss_dict["smpl_temporal_smooth"] = (velocity_next - velocity_prev).abs().mean() \
                        * joint_smooth_reg.w
        
        # voxel deformer regularization
        voxel_deformer_reg = self.reg_cfg.get("voxel_deformer_reg", None)
        if voxel_deformer_reg is not None and self.use_voxel_deformer:
            w_std = self.template.voxel_deformer.get_tv("dc")
            w_rest_std = self.template.voxel_deformer.get_tv("rest")
            w_norm = self.template.voxel_deformer.get_mag("dc")
            w_rest_norm = self.template.voxel_deformer.get_mag("rest")
            
            loss_dict["voxel_deformer_reg"] = \
                voxel_deformer_reg.lambda_std_w * w_std + \
                voxel_deformer_reg.lambda_std_w_rest * w_rest_std + \
                voxel_deformer_reg.lambda_w_norm * w_norm + \
                voxel_deformer_reg.lambda_w_rest_norm * w_rest_norm
        
        # knn regularization
        knn_reg = self.reg_cfg.get("knn_reg", None)
        if knn_reg is not None:
            K = self.ctrl_cfg.knn_neighbors
            instances_mask = self.instances_fv[self.cur_frame]
            nn_ind = self.nn_ind[instances_mask] # (num_instances, smpl_points_num, knn_neighbors)
            
            if not self.ctrl_cfg.freeze_shs_dc:
                valid_shs_dc = self._features_dc.reshape(self.num_instances, self.smpl_points_num, 3)[instances_mask] # (num_instances, smpl_points_num, 3)
                nn_ind_expanded = nn_ind.unsqueeze(-1).expand(-1, -1, -1, 3)
                knn_shs_dc = torch.gather(valid_shs_dc.unsqueeze(2).expand(-1, -1, K, -1), 1, nn_ind_expanded) # (num_instances, smpl_points_num, knn_neighbors, 3)
                shs_dc_std = knn_shs_dc.std(dim=2).mean()
                loss_dict["knn_reg_dc"] = shs_dc_std * knn_reg.lambda_std_shs_dc
            
            if not self.ctrl_cfg.freeze_shs_rest and self.sh_degree > 0:
                dim_sh = num_sh_bases(self.sh_degree)
                valid_shs_rest = self._features_rest.reshape(self.num_instances, self.smpl_points_num, -1)[instances_mask] # (num_instances, smpl_points_num, (dim_sh-1)*3)
                nn_ind_expanded = nn_ind.unsqueeze(-1).expand(-1, -1, -1, (dim_sh-1)*3)
                knn_shs_rest = torch.gather(valid_shs_rest.unsqueeze(2).expand(-1, -1, K, -1), 1, nn_ind_expanded) # (num_instances, smpl_points_num, knn_neighbors, (dim_sh-1)*3)
                shs_rest_std = knn_shs_rest.std(dim=2).mean()
                loss_dict["knn_reg_rest"] = shs_rest_std * knn_reg.lambda_std_shs_rest
            
            if not self.ctrl_cfg.freeze_o:
                valid_o = self.get_opacity.reshape(self.num_instances, self.smpl_points_num, 1)[instances_mask] # (num_instances, smpl_points_num, 1)
                nn_ind_expanded = nn_ind.unsqueeze(-1).expand(-1, -1, -1, 1)
                knn_o = torch.gather(valid_o.unsqueeze(2).expand(-1, -1, K, -1), 1, nn_ind_expanded)
                o_std = knn_o.std(dim=2).mean()
                loss_dict["knn_reg_o"] = o_std * knn_reg.lambda_std_o

            if not self.ctrl_cfg.freeze_s:
                scale_dim = 1 if self.ball_gaussians else 3
                valid_s = self.get_scaling.reshape(self.num_instances, self.smpl_points_num, scale_dim)[instances_mask] # (num_instances, smpl_points_num, 1)
                nn_ind_expanded = nn_ind.unsqueeze(-1).expand(-1, -1, -1, scale_dim)
                knn_s = torch.gather(valid_s.unsqueeze(2).expand(-1, -1, K, -1), 1, nn_ind_expanded)
                s_std = knn_s.std(dim=2).mean()
                loss_dict["knn_reg_s"] = s_std * knn_reg.lambda_std_s
            
            if not self.ctrl_cfg.freeze_q:
                valid_q = self._quats.reshape(self.num_instances, self.smpl_points_num, 4)[instances_mask] # (num_instances, smpl_points_num, 4)
                nn_ind_expanded = nn_ind.unsqueeze(-1).expand(-1, -1, -1, 4)
                knn_q = torch.gather(valid_q.unsqueeze(2).expand(-1, -1, K, -1), 1, nn_ind_expanded)
                q_std = knn_q.std(dim=2).mean()
                loss_dict["knn_reg_q"] = q_std * knn_reg.lambda_std_q
            
            # valid_x = self._means.reshape(self.num_instances, self.smpl_points_num, 3)[instances_mask] # (num_instances, smpl_points_num, 3)
            # nn_ind_expanded = nn_ind.unsqueeze(-1).expand(-1, -1, -1, 3)
            # knn_x = torch.gather(valid_x.unsqueeze(2).expand(-1, -1, K, -1), 1, nn_ind_expanded)
            # x_std = knn_x.std(dim=2).mean()
            # loss_dict["knn_reg_x"] = x_std * knn_reg.lambda_std_x
            
        x_offset_reg = self.reg_cfg.get("x_offset", None)
        if x_offset_reg is not None and self.ctrl_cfg.constrain_xyz_offset and not self.ctrl_cfg.freeze_x:
            instances_mask = self.instances_fv[self.cur_frame]
            valid_x = self._means.reshape(self.num_instances, self.smpl_points_num, 3)[instances_mask] # (num_instances, smpl_points_num, 3)
            valid_x_on_mesh = self.on_mesh_x.reshape(self.num_instances, self.smpl_points_num, 3)[instances_mask]
            x_offset = (valid_x - valid_x_on_mesh).norm(dim=-1).mean()
            
            loss_dict["x_offset"] = x_offset * x_offset_reg.w

        return loss_dict
    
    def state_dict(self) -> Dict:
        state_dict = VanillaGaussians.state_dict(self)
        state_dict.update({
            "points_ids": self.point_ids,
            "instances_size": self.instances_size,
            "instances_fv": self.instances_fv,
        })
        return state_dict

    def load_state_dict(self, state_dict: Dict, **kwargs) -> str:
        self.point_ids = state_dict.pop("points_ids")
        self.instances_size = state_dict.pop("instances_size")
        self.instances_fv = state_dict.pop("instances_fv")
        self.instances_trans = Parameter(
            torch.zeros(self.num_frames, self.num_instances, 3, device=self.device)
        )
        self.instances_quats = Parameter(
            torch.zeros(self.num_frames, self.num_instances, 1, 4, device=self.device)
        )
        self.smpl_qauts = Parameter(
            torch.zeros(self.num_frames, self.num_instances, 23, 4, device=self.device)
        )
        self.template = SMPLTemplate(
            smpl_model_path=f'{CACHE_DIR_4DHUMANS}/data/smpl/SMPL_NEUTRAL.pkl',
            num_human=self.num_instances,
            init_beta=torch.zeros(self.num_instances, 10, device=self.device),
            cano_pose_type="da_pose",
            use_voxel_deformer=self.use_voxel_deformer,
            is_resume=True
        ).to(self.device)
        if self.use_voxel_deformer:
            self.template.voxel_deformer.enable_voxel_correction()
        msg = VanillaGaussians.load_state_dict(self, state_dict, **kwargs)
        self.update_knn(self._means)
        return msg

    def get_instance_activated_gs_dict(self, ins_id: int) -> Dict[str, torch.Tensor]:
        pts_mask = self.point_ids[..., 0] == ins_id
        if pts_mask.sum() < 100:
            return None
        local_means = self._means[pts_mask]
        activated_opacities = torch.sigmoid(self._opacities[pts_mask])
        activated_scales = torch.exp(self._scales[pts_mask].repeat(1, 3) if self.ball_gaussians else self._scales[pts_mask])
        activated_local_rotations = self.quat_act(self._quats[pts_mask])
        gaussian_dict = {
            "means": local_means,
            "opacities": activated_opacities,            
            "scales": activated_scales,
            "quats": activated_local_rotations,
            "sh_dcs": self._features_dc[pts_mask],
            "sh_rests": self._features_rest[pts_mask],
            "ids": self.point_ids[pts_mask],
        }
        return gaussian_dict

    def deform_gaussian_points(
        self, gaussian_dict: Dict[str, torch.Tensor], cur_normalized_time: float,
    ) -> Dict[str, torch.Tensor]:
        """
        deform the points
        """
        means = gaussian_dict["means"]
        cur_frame = torch.argmin(
            torch.abs(self.normalized_timestamps - cur_normalized_time)
        )
        ins_id = gaussian_dict["ids"].flatten()[0]
        if not self.instances_fv[cur_frame, ins_id]:
            # find the nearest frame that has the instance
            for i in range(1, self.num_frames):
                if cur_frame - i >= 0:
                    if self.instances_fv[cur_frame-i, ins_id]:
                        cur_frame = cur_frame - i
                        break
                if cur_frame + i < self.num_frames:
                    if self.instances_fv[cur_frame+i, ins_id]:
                        cur_frame = cur_frame + i
                        break
        instance_mask = torch.zeros(self.num_instances, device=self.device)
        instance_mask[ins_id] = 1
        instance_mask = instance_mask.bool()
        masked_theta = torch.cat(
            [torch.tensor([1.0, 1.0, 0.0, 0.0], device=self.device).unsqueeze(0),
             self.smpl_qauts[cur_frame, ins_id]], dim=0
        ).unsqueeze(0)
        
        W, A = self.template(
            masked_theta = self.quat_act(masked_theta),
            instances_mask = instance_mask,
            xyz_canonical = means.reshape(1, self.smpl_points_num, 3).repeat(self.num_instances, 1, 1) if self.use_voxel_deformer else None
        )
        T = torch.einsum("bnj, bjrc -> bnrc", W, A)
        R = T[..., :3, :3].squeeze() # [N, 3, 3]
        t = T[..., :3, 3].squeeze()  # [N, 3]
        
        deformed_means = torch.einsum(
            "nij,nj->ni", R, means       
        ) + t  # [6890, 3]
        gaussian_dict["means"] = deformed_means

        # placeholder for quats rotating: TODO
        gaussian_dict["quats"] = gaussian_dict["quats"]
        return gaussian_dict

    def collect_gaussians_from_ids(self, ids: List[int]) -> Dict:
        # 按实例 id 把当前场景里的 Gaussian 参数、逐帧运动和 SMPL 模板取出来，后续用于复制、替换或导出。
        gaussian_dict = {}
        for id in ids:
            if id not in gaussian_dict:
                smpl_template = {
                    "J_canonical": self.template.J_canonical[id],
                    "W": self.template.W[id],
                }
                if self.use_voxel_deformer:
                    smpl_template["voxel_deformer"] = {
                        "lbs_voxel_base": self.template.voxel_deformer.lbs_voxel_base[id],
                        "voxel_w_correction": self.template.voxel_deformer.voxel_w_correction[id],
                        "offset": self.template.voxel_deformer.offset[id],
                        "scale": self.template.voxel_deformer.scale[id],
                    }
                instance_raw_dict = {
                    "_means": self._means[self.point_ids[..., 0] == id],
                    "_scales": self._scales[self.point_ids[..., 0] == id],
                    "_quats": self._quats[self.point_ids[..., 0] == id],
                    "_features_dc": self._features_dc[self.point_ids[..., 0] == id],
                    "_features_rest": self._features_rest[self.point_ids[..., 0] == id],
                    "_opacities": self._opacities[self.point_ids[..., 0] == id],
                    "point_ids": self.point_ids[self.point_ids[..., 0] == id],
                    "instances_fv": self.instances_fv[:, id],
                    "instances_trans": self.instances_trans.data[:, id],
                    "instances_quats": self.instances_quats.data[:, id],
                    "smpl_qauts": self.smpl_qauts.data[:, id],
                    "smpl_template": smpl_template,
                }
                gaussian_dict[id] = instance_raw_dict
        return gaussian_dict

    def _clone_tensor(self, tensor: torch.Tensor, cpu: bool = False) -> torch.Tensor:
        # 复制一份不带梯度关系的张量，导出到 pt 文件时可以顺手转到 CPU。
        tensor = tensor.detach().clone()
        return tensor.cpu() if cpu else tensor

    def _build_template_asset_from_dict(self, instance_dict: Dict, cpu: bool = False) -> Dict:
        # 从实例字典中提取 SMPL 模板部分，整理成可以写入 pt/PLY 资产的结构。
        template = {
            "J_canonical": self._clone_tensor(instance_dict["smpl_template"]["J_canonical"], cpu=cpu),
            "W": self._clone_tensor(instance_dict["smpl_template"]["W"], cpu=cpu),
        }
        if self.use_voxel_deformer:
            template["voxel_deformer"] = {
                "lbs_voxel_base": self._clone_tensor(
                    instance_dict["smpl_template"]["voxel_deformer"]["lbs_voxel_base"],
                    cpu=cpu,
                ),
                "voxel_w_correction": self._clone_tensor(
                    instance_dict["smpl_template"]["voxel_deformer"]["voxel_w_correction"],
                    cpu=cpu,
                ),
                "offset": self._clone_tensor(
                    instance_dict["smpl_template"]["voxel_deformer"]["offset"],
                    cpu=cpu,
                ),
                "scale": self._clone_tensor(
                    instance_dict["smpl_template"]["voxel_deformer"]["scale"],
                    cpu=cpu,
                ),
            }
        return template

    def _build_gaussian_asset_from_dict(self, instance_dict: Dict, cpu: bool = False) -> Dict:
        # 从实例字典中提取 Gaussian 参数，并附上对应的 SMPL 模板，形成可复用的外部资产。
        return {
            "gaussian": {
                "_means": self._clone_tensor(instance_dict["_means"], cpu=cpu),
                "_scales": self._clone_tensor(instance_dict["_scales"], cpu=cpu),
                "_quats": self._clone_tensor(instance_dict["_quats"], cpu=cpu),
                "_features_dc": self._clone_tensor(instance_dict["_features_dc"], cpu=cpu),
                "_features_rest": self._clone_tensor(instance_dict["_features_rest"], cpu=cpu),
                "_opacities": self._clone_tensor(instance_dict["_opacities"], cpu=cpu),
            },
            "smpl_template": self._build_template_asset_from_dict(instance_dict, cpu=cpu),
        }

    def _build_motion_asset_from_dict(
        self,
        instance_dict: Dict,
        sparse: bool = False,
        cpu: bool = False,
    ) -> Dict:
        # 把实例的逐帧平移、旋转和 SMPL 姿态整理成 motion 资产；稀疏模式只保存有效帧。
        motion_fv = self._clone_tensor(instance_dict["instances_fv"], cpu=cpu)
        if sparse:
            valid_frame_idx = torch.nonzero(motion_fv, as_tuple=False).squeeze(-1)
            motion = {
                "num_valid_frames": int(valid_frame_idx.numel()),
                "instances_trans_valid": self._clone_tensor(
                    instance_dict["instances_trans"][valid_frame_idx],
                    cpu=cpu,
                ),
                "instances_quats_valid": self._clone_tensor(
                    instance_dict["instances_quats"][valid_frame_idx][:, 0],
                    cpu=cpu,
                ),
                "smpl_qauts_valid": self._clone_tensor(
                    instance_dict["smpl_qauts"][valid_frame_idx],
                    cpu=cpu,
                ),
            }
        else:
            motion = {
                "instances_fv": motion_fv,
                "instances_trans": self._clone_tensor(instance_dict["instances_trans"], cpu=cpu),
                "instances_quats": self._clone_tensor(instance_dict["instances_quats"], cpu=cpu),
                "smpl_qauts": self._clone_tensor(instance_dict["smpl_qauts"], cpu=cpu),
            }
        return {"motion": motion}

    def _get_scene_instance_asset(
        self,
        instance_id: int,
        sparse_motion: bool = False,
        cpu: bool = False,
    ) -> Dict:
        # 将当前场景里的一个 SMPL 实例完整打包，包含外观 Gaussian、身体模板和运动轨迹。
        instance_dict = self.collect_gaussians_from_ids([instance_id])[instance_id]
        asset = self._build_gaussian_asset_from_dict(instance_dict, cpu=cpu)
        asset.update(
            self._build_motion_asset_from_dict(
                instance_dict,
                sparse=sparse_motion,
                cpu=cpu,
            )
        )
        return asset

    def _build_reference_from_valid_sequences(
        self,
        trans_valid: torch.Tensor,
        quats_valid: torch.Tensor,
        appear_frames: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 根据一段只包含有效帧的平移和根旋转，生成与当前场景帧数一致的参考轨迹。
        return self._build_reference_from_valid_sequences_for_frames(
            trans_valid=trans_valid,
            quats_valid=quats_valid,
            appear_frames=appear_frames,
            num_frames=self.num_frames,
            fv_device=self.instances_fv.device,
            fv_dtype=self.instances_fv.dtype,
            trans_device=self.instances_trans.device,
            trans_dtype=self.instances_trans.dtype,
            quats_device=self.instances_quats.device,
            quats_dtype=self.instances_quats.dtype,
        )

    def _build_reference_from_valid_sequences_for_frames(
        self,
        trans_valid: torch.Tensor,
        quats_valid: torch.Tensor,
        appear_frames: Optional[int],
        num_frames: int,
        fv_device: torch.device,
        fv_dtype: torch.dtype,
        trans_device: torch.device,
        trans_dtype: torch.dtype,
        quats_device: torch.device,
        quats_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 把有效帧序列填充到指定总帧数：有效区间写入真实值，前后帧沿用边界姿态。
        if trans_valid.ndim != 2 or trans_valid.shape[-1] != 3:
            raise ValueError(f"Expected trans_valid shape (N, 3), got {trans_valid.shape}")
        if quats_valid.ndim != 2 or quats_valid.shape[-1] != 4:
            raise ValueError(f"Expected quats_valid shape (N, 4), got {quats_valid.shape}")

        start_frame = 0 if appear_frames is None else max(0, int(appear_frames))
        if start_frame >= num_frames:
            start_frame = num_frames

        valid_count = min(trans_valid.shape[0], quats_valid.shape[0], max(0, num_frames - start_frame))

        ref_instances_fv = torch.zeros(
            (num_frames,),
            device=fv_device,
            dtype=fv_dtype,
        )
        ref_instances_trans = torch.zeros(
            (num_frames, 3),
            device=trans_device,
            dtype=trans_dtype,
        )
        ref_instances_quats = torch.zeros(
            (num_frames, 4),
            device=quats_device,
            dtype=quats_dtype,
        )
        ref_instances_quats[:, 0] = torch.ones_like(ref_instances_quats[:, 0])

        if valid_count > 0:
            end_frame = start_frame + valid_count
            ref_instances_fv[start_frame:end_frame] = True
            ref_instances_trans[start_frame:end_frame] = trans_valid[:valid_count]
            ref_instances_quats[start_frame:end_frame] = quats_valid[:valid_count]
            ref_instances_trans[:start_frame] = trans_valid[0]
            ref_instances_quats[:start_frame] = quats_valid[0]
            if end_frame < num_frames:
                ref_instances_trans[end_frame:] = trans_valid[valid_count - 1]
                ref_instances_quats[end_frame:] = quats_valid[valid_count - 1]

        return (
            ref_instances_fv.unsqueeze(1),
            ref_instances_trans.unsqueeze(1),
            ref_instances_quats.unsqueeze(1).unsqueeze(1),
        )

    def _shift_scene_reference_to_start_frame(
        self,
        ref_instances_fv: torch.Tensor,
        ref_instances_trans: torch.Tensor,
        ref_instances_quats: torch.Tensor,
        appear_frames: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 复用已有实例轨迹作为参考时，将它的有效片段重新放到 appear_frames 指定的起始帧。
        if appear_frames is None:
            return ref_instances_fv, ref_instances_trans, ref_instances_quats

        start_frame = max(0, int(appear_frames))
        visible_idx = torch.nonzero(ref_instances_fv[:, 0].bool(), as_tuple=False).squeeze(-1)
        trans_valid = ref_instances_trans[visible_idx, 0, :]
        quats_valid = ref_instances_quats[visible_idx, 0, 0, :]
        return self._build_reference_from_valid_sequences(
            trans_valid=trans_valid,
            quats_valid=quats_valid,
            appear_frames=start_frame,
        )

    def _find_motion_sidecar_path(self, ply_path: str) -> Optional[str]:
        # 按约定查找和 Gaussian PLY 同名的 *_motion.pt，用来补充 PLY 本身不保存的运动信息。
        stem, _ = os.path.splitext(ply_path)
        sidecar_path = f"{stem}_motion.pt"
        if os.path.exists(sidecar_path):
            return sidecar_path
        return None

    def _load_sidecar_motion_asset(
        self,
        ply_path: str,
    ) -> Tuple[Optional[Dict], Optional[str]]:
        # 如果找到了 PLY 对应的 motion sidecar，就读取其中的运动资产，否则返回空结果。
        sidecar_path = self._find_motion_sidecar_path(ply_path)
        if sidecar_path is None:
            return None, None
        return self._load_motion_asset(sidecar_path), sidecar_path

    def _build_gaussian_asset_from_ply(
        self,
        ply_path: str,
        dataset = None,
    ) -> Dict:
        # 将外部 PLY 转成内部 add/replace 统一使用的资产字典，字段名和设备 dtype 在这里对齐。
        gs = self.load_ply_as_gs(
            ply_path,
            device=self.device,
            sh_degree=self.sh_degree,
            scale_factor=1.0,
        )

        if dataset is not None:
            from models.color_utils import match_gaussian_brightness
            gs, scale = match_gaussian_brightness(gs, dataset, self.sh_degree)
            if scale is not None:
                print(f"[brightness] {ply_path.split('/')[-1]}: scale={scale:.3f}")

        asset = {
            "gaussian": {
                "_means": gs["means"],
                "_scales": gs["scales"],
                "_quats": gs["quats"],
                "_features_dc": gs["features_dc"],
                "_features_rest": gs["features_rest"],
                "_opacities": gs["opacities"],
            },
        }
        if "smpl_template" in gs:
            asset["smpl_template"] = gs["smpl_template"]
        return asset

    def _merge_template_assets(
        self,
        preferred_template: Optional[Dict],
        fallback_template: Optional[Dict],
    ) -> Optional[Dict]:
        # 合并两个来源的模板数据：PLY 中显式导出的模板优先，缺失部分再用 motion/sidecar 里的模板补齐。
        if preferred_template is None:
            return fallback_template
        if fallback_template is None:
            return preferred_template

        merged = dict(fallback_template)
        merged["J_canonical"] = preferred_template["J_canonical"]
        merged["W"] = preferred_template["W"]
        if "voxel_deformer" in preferred_template:
            merged["voxel_deformer"] = preferred_template["voxel_deformer"]
        return merged

    def _resolve_template_asset(
        self,
        preferred_template: Optional[Dict],
        primary_asset: Optional[Dict],
        primary_label: Optional[str],
        sidecar_asset: Optional[Dict],
        sidecar_label: Optional[str],
        context: str,
    ) -> Optional[Dict]:
        # 在添加或替换时决定使用哪个 SMPL 模板，避免 PLY 缺少 voxel 模板时无法恢复身体绑定信息。
        fallback_template = None
        fallback_label = None
        if primary_asset is not None and "smpl_template" in primary_asset:
            fallback_template = primary_asset["smpl_template"]
            fallback_label = primary_label
        elif sidecar_asset is not None and "smpl_template" in sidecar_asset:
            fallback_template = sidecar_asset["smpl_template"]
            fallback_label = sidecar_label

        if fallback_template is not None and (
            preferred_template is None
            or (
                self.use_voxel_deformer
                and "voxel_deformer" not in preferred_template
                and "voxel_deformer" in fallback_template
            )
        ):
            logger.warning(
                "SMPL template for %s is supplemented from pt asset: %s",
                context,
                fallback_label,
            )

        return self._merge_template_assets(
            preferred_template=preferred_template,
            fallback_template=fallback_template,
        )

    def _prepare_gaussian_tensors(
        self,
        gaussian_asset: Dict,
        expected_features_rest_shape: Tuple[int, ...],
    ) -> Dict[str, torch.Tensor]:
        # 把外部 Gaussian 资产转换到当前模型的设备和 dtype，同时检查点数、SH 特征形状等是否匹配。
        gaussian = gaussian_asset["gaussian"]
        prepared = {
            "_means": gaussian["_means"].to(device=self._means.device, dtype=self._means.dtype),
            "_scales": gaussian["_scales"].to(device=self._scales.device, dtype=self._scales.dtype),
            "_quats": gaussian["_quats"].to(device=self._quats.device, dtype=self._quats.dtype),
            "_features_dc": gaussian["_features_dc"].to(device=self._features_dc.device, dtype=self._features_dc.dtype),
            "_features_rest": gaussian["_features_rest"].to(
                device=self._features_rest.device,
                dtype=self._features_rest.dtype,
            ),
            "_opacities": gaussian["_opacities"].to(device=self._opacities.device, dtype=self._opacities.dtype),
        }

        if prepared["_means"].shape[0] != self.smpl_points_num:
            raise ValueError(
                f"Asset points mismatch: got {prepared['_means'].shape[0]}, expected {self.smpl_points_num}. "
                "Please export full-point SMPL Gaussian assets."
            )

        if self._scales.shape[1] == 1 and prepared["_scales"].shape[1] == 3:
            prepared["_scales"] = prepared["_scales"].mean(dim=-1, keepdim=True)
        elif self._scales.shape[1] == 3 and prepared["_scales"].shape[1] == 1:
            prepared["_scales"] = prepared["_scales"].repeat(1, 3)

        if prepared["_features_rest"].shape[1:] != expected_features_rest_shape:
            raise RuntimeError(
                f"features_rest shape mismatch: {prepared['_features_rest'].shape[1:]} vs "
                f"{expected_features_rest_shape}"
            )
        return prepared

    def _normalize_template_asset(
        self,
        template_asset: Dict,
    ) -> Dict:
        # 把外部 SMPL 模板转换到当前 template 的设备和 dtype，方便后面直接 copy 到模型参数里。
        normalized = {
            "J_canonical": template_asset["J_canonical"].to(
                device=self.template.J_canonical.device,
                dtype=self.template.J_canonical.dtype,
            ),
            "W": template_asset["W"].to(
                device=self.template.W.device,
                dtype=self.template.W.dtype,
            ),
        }

        if self.use_voxel_deformer:
            voxel_deformer = template_asset["voxel_deformer"]
            normalized["voxel_deformer"] = {
                "lbs_voxel_base": voxel_deformer["lbs_voxel_base"].to(
                    device=self.template.voxel_deformer.lbs_voxel_base.device,
                    dtype=self.template.voxel_deformer.lbs_voxel_base.dtype,
                ),
                "voxel_w_correction": voxel_deformer["voxel_w_correction"].to(
                    device=self.template.voxel_deformer.voxel_w_correction.device,
                    dtype=self.template.voxel_deformer.voxel_w_correction.dtype,
                ),
                "offset": voxel_deformer["offset"].to(
                    device=self.template.voxel_deformer.offset.device,
                    dtype=self.template.voxel_deformer.offset.dtype,
                ),
                "scale": voxel_deformer["scale"].to(
                    device=self.template.voxel_deformer.scale.device,
                    dtype=self.template.voxel_deformer.scale.dtype,
                ),
            }

        return normalized

    def _apply_template_asset(
        self,
        target_id: int,
        template_asset: Dict,
    ) -> None:
        # 用外部模板覆盖目标实例的关节和权重，并重新计算 A0_inv 这类依赖模板的绑定姿态缓存。
        normalized_template = self._normalize_template_asset(template_asset)
        new_J_canonical = normalized_template["J_canonical"]
        self.template.J_canonical[target_id].copy_(new_J_canonical)
        self.template.W[target_id].copy_(normalized_template["W"])
        _, new_A0 = batch_rigid_transform(
            self.template.canonical_pose[None].to(new_J_canonical),
            new_J_canonical[None],
            self.template._template_layer.parents,
        )
        self.template.A0_inv[target_id].copy_(
            torch.inverse(new_A0).squeeze(0).to(
                device=self.template.A0_inv.device,
                dtype=self.template.A0_inv.dtype,
            )
        )

        if self.use_voxel_deformer:
            voxel_deformer = normalized_template["voxel_deformer"]
            self.template.voxel_deformer.lbs_voxel_base[target_id].copy_(
                voxel_deformer["lbs_voxel_base"]
            )
            self.template.voxel_deformer.voxel_w_correction.data[target_id].copy_(
                voxel_deformer["voxel_w_correction"]
            )
            self.template.voxel_deformer.offset[target_id].copy_(
                voxel_deformer["offset"]
            )
            self.template.voxel_deformer.scale[target_id].copy_(
                voxel_deformer["scale"]
            )

    def _append_instance_from_asset(
        self,
        gaussian_tensors: Dict[str, torch.Tensor],
        template_asset: Dict,
        ref_instances_fv: torch.Tensor,
        ref_instances_trans: torch.Tensor,
        ref_instances_quats: torch.Tensor,
        ref_instances_size: Optional[torch.Tensor],
    ) -> int:
        # 在已有 SMPL 场景中追加一个新人：拼接 Gaussian 参数、实例轨迹、SMPL 姿态和模板数据。
        new_id = self.num_instances
        new_point_ids = torch.full(
            (self.smpl_points_num, 1),
            new_id,
            dtype=self.point_ids.dtype,
            device=self.point_ids.device,
        )
        ref_smpl_qauts = torch.zeros_like(self.smpl_qauts[:, :1]).detach()
        ref_smpl_qauts[..., 0] = 1.0

        self._means = Parameter(torch.cat([self._means.detach(), gaussian_tensors["_means"]], dim=0))
        self._scales = Parameter(torch.cat([self._scales.detach(), gaussian_tensors["_scales"]], dim=0))
        self._quats = Parameter(torch.cat([self._quats.detach(), gaussian_tensors["_quats"]], dim=0))
        self._features_dc = Parameter(torch.cat([self._features_dc.detach(), gaussian_tensors["_features_dc"]], dim=0))
        self._features_rest = Parameter(torch.cat([self._features_rest.detach(), gaussian_tensors["_features_rest"]], dim=0))
        self._opacities = Parameter(torch.cat([self._opacities.detach(), gaussian_tensors["_opacities"]], dim=0))
        self.point_ids = torch.cat([self.point_ids, new_point_ids], dim=0)
        self.instances_fv = torch.cat([self.instances_fv, ref_instances_fv], dim=1)
        self.instances_trans = Parameter(
            torch.cat([self.instances_trans.detach(), ref_instances_trans], dim=1)
        )
        self.instances_quats = Parameter(
            torch.cat([self.instances_quats.detach(), ref_instances_quats], dim=1)
        )
        self.smpl_qauts = Parameter(
            torch.cat([self.smpl_qauts.detach(), ref_smpl_qauts], dim=1)
        )
        if hasattr(self, "instances_size") and ref_instances_size is not None:
            self.instances_size = torch.cat([self.instances_size, ref_instances_size], dim=0)

        normalized_template = self._normalize_template_asset(template_asset)
        self.template.add_instance(new_id, normalized_template)
        if self.template.A0_inv.shape[0] == self.num_instances - 1 and self.template.J_canonical.shape[0] == self.num_instances:
            new_J_canonical = self.template.J_canonical[new_id:new_id + 1]
            _, new_A0 = batch_rigid_transform(
                self.template.canonical_pose[None].to(new_J_canonical),
                new_J_canonical,
                self.template._template_layer.parents,
            )
            self.template.A0_inv = torch.cat([self.template.A0_inv, torch.inverse(new_A0)], dim=0)
        if self.template.A0_inv.shape[0] != self.num_instances:
            raise RuntimeError(
                "SMPL template instance count mismatch after add: "
                f"num_instances={self.num_instances}, A0_inv={self.template.A0_inv.shape[0]}, "
                f"J_canonical={self.template.J_canonical.shape[0]}, W={self.template.W.shape[0]}"
            )
        self.update_knn(self._means)
        return new_id

    def _resolve_first_reference_source(
        self,
        ref_source: Union[int, str],
        appear_frames: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        # 空场景添加第一个 SMPL 时没有可参考的场景实例，所以这里必须从导出的 motion.pt 读取参考轨迹。
        if isinstance(ref_source, int):
            raise ValueError(
                "SMPL add into an empty scene requires ref_id to be an exported motion.pt path; "
                "there is no existing SMPL instance trajectory to reference."
            )

        ref_asset = self._load_motion_asset(str(ref_source))
        motion = ref_asset["motion"]
        # 第一个实例没有场景内轨迹可复制，所以参考轨迹只接受导出的稀疏 trans/quats 有效帧序列。
        if "instances_trans_valid" not in motion or "instances_quats_valid" not in motion:
            raise RuntimeError(
                "Unsupported reference motion format for first SMPL add: require sparse "
                "('instances_trans_valid'+'instances_quats_valid')."
            )

        trans_valid = motion["instances_trans_valid"].to(
            device=self.device,
            dtype=self._means.dtype,
        )
        quats_valid = motion["instances_quats_valid"].to(
            device=self.device,
            dtype=self._quats.dtype,
        )
        # 把 motion.pt 中的有效帧轨迹铺回完整训练帧数，作为第一个实例的初始出现区间和根位姿。
        return (*self._build_reference_from_valid_sequences_for_frames(
            trans_valid=trans_valid,
            quats_valid=quats_valid,
            appear_frames=appear_frames,
            num_frames=self.num_train_images,
            fv_device=self.device,
            fv_dtype=torch.bool,
            trans_device=self.device,
            trans_dtype=self._means.dtype,
            quats_device=self.device,
            quats_dtype=self._quats.dtype,
        ), str(ref_source))

    def _initialize_first_instance_from_asset(
        self,
        ref_source: Union[int, str],
        gaussian_asset: Dict,
        template_asset: Dict,
        motion_asset: Dict,
        offset: List[float],
        appear_frames: Optional[int],
    ) -> int:
        # 当场景还没有任何 SMPL 实例时，用外部资产创建第一个实例并初始化所有相关参数和模板。
        # 此时 self._means 等 Parameter 还不能按已有实例拼接，只能直接用资产里的完整 6890 点初始化。
        gaussian_tensors = self._prepare_gaussian_tensors(
            gaussian_asset=gaussian_asset,
            expected_features_rest_shape=gaussian_asset["gaussian"]["_features_rest"].shape[1:],
        )

        # 先构造 fv、trans 和 root quats，后面才能建立 num_frames x 1 的 Parameter。
        ref_instances_fv, ref_instances_trans, ref_instances_quats, ref_desc = self._resolve_first_reference_source(
            ref_source=ref_source,
            appear_frames=appear_frames,
        )

        offset_tensor = torch.as_tensor(
            offset,
            device=ref_instances_trans.device,
            dtype=ref_instances_trans.dtype,
        )
        if offset_tensor.shape != (3,):
            raise ValueError(
                f"SMPL add offset must have shape (3,), got {tuple(offset_tensor.shape)}"
            )
        if ref_instances_trans.ndim != 3 or ref_instances_trans.shape[1:] != (1, 3):
            raise ValueError(
                f"Expected ref_instances_trans shape (N, 1, 3), got {tuple(ref_instances_trans.shape)}"
            )
        ref_instances_trans += offset_tensor.view(1, 1, 3)

        # 空节点初始化时，Gaussian 参数直接来自外部 PLY，point_ids 全部归到第 0 个 SMPL 实例。
        self._means = Parameter(gaussian_tensors["_means"])
        self._scales = Parameter(gaussian_tensors["_scales"])
        self._quats = Parameter(gaussian_tensors["_quats"])
        self._features_dc = Parameter(gaussian_tensors["_features_dc"])
        self._features_rest = Parameter(gaussian_tensors["_features_rest"])
        self._opacities = Parameter(gaussian_tensors["_opacities"])
        self.point_ids = torch.zeros(
            (self.smpl_points_num, 1),
            dtype=torch.long,
            device=self.device,
        )
        self.instances_fv = ref_instances_fv
        self.instances_trans = Parameter(ref_instances_trans)
        self.instances_quats = Parameter(ref_instances_quats)
        self.smpl_qauts = Parameter(
            torch.zeros(
                self.num_train_images,
                1,
                23,
                4,
                device=self.device,
                dtype=self._quats.dtype,
            )
        )
        self.smpl_qauts.data[..., 0] = 1.0
        # 没有原始 instances_size 可继承时，用 canonical Gaussian 包围盒作为第一个实例的尺寸估计。
        self.instances_size = (
            gaussian_tensors["_means"].max(dim=0).values
            - gaussian_tensors["_means"].min(dim=0).values
        ).unsqueeze(0)

        # 创建只有一个 human slot 的 SMPLTemplate，再把外部模板覆盖进去，保证关节和蒙皮权重来自资产。
        self.template = SMPLTemplate(
            smpl_model_path=f'{CACHE_DIR_4DHUMANS}/data/smpl/SMPL_NEUTRAL.pkl',
            num_human=1,
            init_beta=torch.zeros(1, 10, device=self.device),
            cano_pose_type="da_pose",
            use_voxel_deformer=self.use_voxel_deformer,
            is_resume=True,
        ).to(self.device)
        if self.use_voxel_deformer:
            self.template.voxel_deformer.enable_voxel_correction()
        self._apply_template_asset(target_id=0, template_asset=template_asset)
        # 最后再应用 motion_asset，让 SMPL 关节姿态跟随导入的运动，同时保留上面构造的出现帧范围。
        self._apply_motion_asset(target_id=0, motion_asset=motion_asset, preserve_target_fv=True)
        self.update_knn(self._means)
        print(
            f"[add_instance_with_ply] initialized first SMPL instance 0 from asset, "
            f"ref_id={ref_desc}"
        )
        return 0

    def _load_motion_asset(self, path: str) -> Dict:
        # 读取 pt 格式的 SMPL motion 资产，并确认里面至少包含 motion 这个顶层字段。
        asset = torch.load(path, map_location=self.device)
        if not isinstance(asset, dict) or "motion" not in asset:
            raise RuntimeError(f"Invalid SMPL motion asset: {path}")
        return asset

    def _resize_reference_mask(self, x: torch.Tensor, target_frames: int) -> torch.Tensor:
        # 调整 fv mask 的长度：帧数过多就截断，帧数不足就用 False 补齐。
        src_frames = x.shape[0]
        if src_frames == target_frames:
            return x
        if src_frames > target_frames:
            return x[:target_frames]
        pad = torch.zeros((target_frames - src_frames,), device=x.device, dtype=x.dtype)
        return torch.cat([x, pad], dim=0)

    def _resize_reference_value(self, x: torch.Tensor, target_frames: int, identity_quat: bool = False) -> torch.Tensor:
        # 调整逐帧数据长度：帧数过多截断，帧数不足时重复最后一帧，空序列则生成默认值。
        src_frames = x.shape[0]
        if src_frames == target_frames:
            return x
        if src_frames > target_frames:
            return x[:target_frames]
        if src_frames == 0:
            out = torch.zeros((target_frames,) + tuple(x.shape[1:]), device=x.device, dtype=x.dtype)
            if identity_quat:
                out[..., 0] = 1.0
            return out
        pad = x[-1:].repeat((target_frames - src_frames,) + tuple(1 for _ in range(x.ndim - 1)))
        return torch.cat([x, pad], dim=0)

    def _resolve_reference_source(
        self,
        ref_source: Union[int, str],
        appear_frames: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], str]:
        # 为 add 操作准备参考轨迹：可以直接复制已有实例，也可以从外部 motion.pt 中恢复。
        if isinstance(ref_source, int):
            if ref_source < 0 or ref_source >= self.num_instances:
                raise ValueError(f"Invalid reference instance id {ref_source}")
            ref_instances_fv = self.instances_fv[:, ref_source:ref_source + 1].clone()
            ref_instances_trans = self.instances_trans[:, ref_source:ref_source + 1, :].detach().clone()
            ref_instances_quats = self.instances_quats[:, ref_source:ref_source + 1, ...].detach().clone()
            ref_instances_size = self.instances_size[ref_source:ref_source + 1].clone() if hasattr(self, "instances_size") else None
            ref_instances_fv, ref_instances_trans, ref_instances_quats = self._shift_scene_reference_to_start_frame(
                ref_instances_fv=ref_instances_fv,
                ref_instances_trans=ref_instances_trans,
                ref_instances_quats=ref_instances_quats,
                appear_frames=appear_frames,
            )
            return ref_instances_fv, ref_instances_trans, ref_instances_quats, ref_instances_size, str(ref_source)

        ref_asset = self._load_motion_asset(str(ref_source))
        motion = ref_asset["motion"]
        if "instances_trans_valid" in motion and "instances_quats_valid" in motion:
            trans_valid = motion["instances_trans_valid"].to(
                device=self.instances_trans.device,
                dtype=self.instances_trans.dtype,
            )
            quats_valid = motion["instances_quats_valid"].to(
                device=self.instances_quats.device,
                dtype=self.instances_quats.dtype,
            )
            ref_instances_fv, ref_instances_trans, ref_instances_quats = self._build_reference_from_valid_sequences(
                trans_valid=trans_valid,
                quats_valid=quats_valid,
                appear_frames=appear_frames,
            )
            return ref_instances_fv, ref_instances_trans, ref_instances_quats, None, ref_source

        raise RuntimeError(
            "Unsupported reference motion format: require sparse "
            "('instances_trans_valid'+'instances_quats_valid')."
        )

    def _normalize_motion_source(
        self,
        motion_source: Optional[Union[int, str]],
    ) -> Optional[Union[int, str]]:
        # 规范化 replace_motion 配置，统一处理为空、实例 id、文件路径这三种来源。
        if motion_source is None:
            return None
        if isinstance(motion_source, bool):
            raise TypeError("SMPL replace_motion does not accept bool values; use a file/id or null")
        if isinstance(motion_source, str):
            stripped = motion_source.strip()
            if not stripped:
                raise ValueError("SMPL replace_motion cannot be an empty string; use null to disable motion replacement")
            return stripped
        if isinstance(motion_source, int):
            return motion_source
        raise TypeError(f"Unsupported motion source: {motion_source}")

    def _translate_instance(self, instance_id: int, offset: Union[List[float], torch.Tensor]) -> None:
        # 给指定 SMPL 实例的所有帧平移加上同一个 xyz 偏移，用于简单的位置编辑。
        if instance_id < 0 or instance_id >= self.num_instances:
            valid_ids = torch.unique(self.point_ids[:, 0]).tolist()
            raise ValueError(
                f"Invalid SMPL instance id {instance_id}, valid ids: {valid_ids}"
            )

        offset = torch.as_tensor(
            offset,
            device=self.instances_trans.device,
            dtype=self.instances_trans.dtype,
        )
        if offset.shape != (3,):
            raise ValueError(
                f"SMPL translation offset must have shape (3,), got {tuple(offset.shape)}"
            )

        self.instances_trans[:, instance_id, :] += offset

    def edit_trajectory(
        self,
        instance_id: int,
        offset: List[float],
    ) -> None:
        # edit_trajectory 的 SMPL 实现：当前只支持把整条实例轨迹整体平移。
        self._translate_instance(instance_id=instance_id, offset=offset)

    def remove_instances(self, remove_id_list):
        """
        Soft delete instances by disabling them in every frame.
        This keeps instance ids stable for subsequent edit operations.
        """
        remove_ids = sorted(set(remove_id_list))

        for rid in remove_ids:
            if rid < 0 or rid >= self.num_instances:
                raise ValueError(f"Invalid instance id {rid}")
        self.instances_fv[:, remove_ids] = False


            
    def replace_instances(
        self,
        replace_dict: Dict[int, int],
        motion_source_dict: Optional[Dict[int, Optional[Union[int, str]]]] = None,
    ) -> None:
        """
        Replace target instances with Gaussian/template data copied from other scene instances.
        
        Args:
            replace_dict: {
                ins_id(to be replaced): ins_id(replace with)
                ...
            }
        """
        new_gaussians_dict = self.collect_gaussians_from_ids(list(replace_dict.values()))
        for ins_id, new_id in replace_dict.items():
            new_gaussian = new_gaussians_dict[new_id]
            asset = self._build_gaussian_asset_from_dict(new_gaussian)
            motion_source = None
            if motion_source_dict is not None and ins_id in motion_source_dict:
                motion_source = motion_source_dict[ins_id]
            motion_asset = self._resolve_motion_asset(motion_source)
            self._replace_instance_from_asset(
                target_id=ins_id,
                asset=asset,
                motion_asset=motion_asset,
            )
            print(
                f"[replace_instances] replaced SMPL instance {ins_id} with scene instance {new_id}, "
                f"motion_source={motion_source}"
            )

    def _resolve_motion_asset(
        self,
        motion_source: Optional[Union[int, str]],
    ) -> Optional[Dict]:
        # 为 replace 操作解析可选运动：可以来自场景中的另一个实例、独立 motion.pt，或 PLY 旁边的 sidecar。
        motion_source = self._normalize_motion_source(motion_source)
        if motion_source is None:
            return None
        if isinstance(motion_source, int):
            if motion_source < 0 or motion_source >= self.num_instances:
                raise ValueError(f"Invalid motion instance id {motion_source}")
            source_asset = self._get_scene_instance_asset(
                instance_id=motion_source,
                sparse_motion=False,
                cpu=False,
            )
            return {"motion": source_asset["motion"]}
        if motion_source.endswith(".ply"):
            sidecar_path = self._find_motion_sidecar_path(motion_source)
            if sidecar_path is None:
                raise FileNotFoundError(f"No motion sidecar found for SMPL Gaussian asset: {motion_source}")
            return self._load_motion_asset(sidecar_path)
        return self._load_motion_asset(motion_source)

    def _decode_motion_pose_sequence(
        self,
        motion: Dict,
    ) -> torch.Tensor:
        # 从 motion 资产中取出 SMPL 关节姿态；既支持全帧序列，也支持只保存有效帧的稀疏序列。
        if "smpl_qauts" in motion:
            return motion["smpl_qauts"].to(
                device=self.smpl_qauts.device,
                dtype=self.smpl_qauts.dtype,
            )

        if "smpl_qauts_valid" not in motion:
            raise RuntimeError(
                "Unsupported motion format: require either dense "
                "('smpl_qauts') or sparse "
                "('smpl_qauts_valid')."
            )

        smpl_qauts_valid = motion["smpl_qauts_valid"].to(
            device=self.smpl_qauts.device,
            dtype=self.smpl_qauts.dtype,
        )
        return smpl_qauts_valid

    def _repeat_motion_over_visibility(
        self,
        target_visible_idx: torch.Tensor,
        source_visible_idx: torch.Tensor,
        motion_smpl_qauts: torch.Tensor,
        target_original_fv: torch.Tensor,
        target_id: int,
        preserve_existing_pose: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 当源运动帧数不够时，将源姿态循环重复到目标实例的可见帧，同时保留目标的可见性。
        repeated_motion_fv = target_original_fv.to(
            device=self.instances_fv.device,
            dtype=self.instances_fv.dtype,
        )
        if preserve_existing_pose:
            repeated_motion_smpl_qauts = self.smpl_qauts.data[:, target_id].clone()
        else:
            repeated_motion_smpl_qauts = torch.zeros(
                (target_original_fv.shape[0],) + tuple(self.smpl_qauts[:, target_id].shape[1:]),
                device=self.smpl_qauts.device,
                dtype=self.smpl_qauts.dtype,
            )

        if target_visible_idx.numel() > 0:
            source_pose_seq = motion_smpl_qauts[source_visible_idx]
            repeat_indices = torch.arange(
                target_visible_idx.numel(),
                device=target_visible_idx.device,
                dtype=torch.long,
            ) % source_visible_idx.numel()
            repeated_motion_smpl_qauts[target_visible_idx] = source_pose_seq[repeat_indices]
        return repeated_motion_fv, repeated_motion_smpl_qauts

    def _align_motion_to_target_frames(
        self,
        target_id: int,
        motion_fv: torch.Tensor,
        motion_smpl_qauts: torch.Tensor,
        target_original_fv: torch.Tensor,
        preserve_target_fv: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 把外部 motion 对齐到目标实例：根据是否保留目标可见性，选择重复、裁剪或补齐姿态序列。
        source_visible_idx = torch.nonzero(motion_fv.bool(), as_tuple=False).squeeze(-1)
        target_visible_idx = torch.nonzero(target_original_fv.bool(), as_tuple=False).squeeze(-1)

        if preserve_target_fv:
            if target_visible_idx.numel() > 0 and source_visible_idx.numel() == 0:
                raise RuntimeError("SMPL add motion has no valid frames, cannot map onto reference fv")
            return self._repeat_motion_over_visibility(
                target_visible_idx=target_visible_idx,
                source_visible_idx=source_visible_idx,
                motion_smpl_qauts=motion_smpl_qauts,
                target_original_fv=target_original_fv,
                target_id=target_id,
                preserve_existing_pose=True,
            )

        if target_visible_idx.numel() > source_visible_idx.numel() and source_visible_idx.numel() > 0:
            return self._repeat_motion_over_visibility(
                target_visible_idx=target_visible_idx,
                source_visible_idx=source_visible_idx,
                motion_smpl_qauts=motion_smpl_qauts,
                target_original_fv=target_original_fv,
                target_id=target_id,
                preserve_existing_pose=False,
            )

        target_frames = target_original_fv.shape[0]
        motion_fv = self._resize_reference_mask(motion_fv, target_frames)
        motion_smpl_qauts = self._resize_reference_value(motion_smpl_qauts, target_frames)
        return motion_fv, motion_smpl_qauts

    def _apply_motion_asset(
        self,
        target_id: int,
        motion_asset: Dict,
        preserve_target_fv: bool = False,
    ) -> None:
        # 把对齐后的 motion 写回目标实例，更新它在哪些帧有效以及对应的 SMPL 姿态。
        motion = motion_asset["motion"]
        target_original_fv = self.instances_fv[:, target_id].detach().clone()
        motion_fv = None
        effective_preserve_target_fv = preserve_target_fv
        if "instances_fv" in motion:
            motion_fv = motion["instances_fv"].to(
                device=self.instances_fv.device,
                dtype=self.instances_fv.dtype,
            )
        else:
            valid_count = int(motion.get("num_valid_frames", motion["smpl_qauts_valid"].shape[0]))
            valid_count = min(valid_count, int(motion["smpl_qauts_valid"].shape[0]))
            motion_fv = torch.ones(
                (valid_count,),
                device=self.instances_fv.device,
                dtype=self.instances_fv.dtype,
            )
            effective_preserve_target_fv = True

        motion_smpl_qauts = self._decode_motion_pose_sequence(
            motion=motion,
        )
        motion_fv, motion_smpl_qauts = self._align_motion_to_target_frames(
            target_id=target_id,
            motion_fv=motion_fv,
            motion_smpl_qauts=motion_smpl_qauts,
            target_original_fv=target_original_fv,
            preserve_target_fv=effective_preserve_target_fv,
        )

        self.instances_fv[:, target_id].copy_(motion_fv)
        pose_valid_mask = motion_fv.bool()
        if pose_valid_mask.any():
            target_smpl_qauts = self.smpl_qauts.data[:, target_id]
            target_smpl_qauts[pose_valid_mask] = motion_smpl_qauts[pose_valid_mask]

    def _replace_instance_from_asset(
        self,
        target_id: int,
        asset: Dict,
        motion_asset: Optional[Dict] = None,
    ) -> None:
        # 执行真正的替换：覆盖目标实例的 Gaussian 外观，必要时同步替换模板和运动。
        if target_id < 0 or target_id >= self.num_instances:
            raise ValueError(f"Invalid target_id {target_id}")

        target_mask = self.point_ids[..., 0] == target_id
        n_target = int(target_mask.sum().item())
        if n_target != self.smpl_points_num:
            raise RuntimeError(
                f"Target instance {target_id} has {n_target} points, expected {self.smpl_points_num}"
            )

        gaussian_tensors = self._prepare_gaussian_tensors(
            gaussian_asset=asset,
            expected_features_rest_shape=self._features_rest[target_mask].shape[1:],
        )

        with torch.no_grad():
            self._means.data[target_mask] = gaussian_tensors["_means"]
            self._scales.data[target_mask] = gaussian_tensors["_scales"]
            self._quats.data[target_mask] = gaussian_tensors["_quats"]
            self._features_dc.data[target_mask] = gaussian_tensors["_features_dc"]
            self._features_rest.data[target_mask] = gaussian_tensors["_features_rest"]
            self._opacities.data[target_mask] = gaussian_tensors["_opacities"]

            if "smpl_template" in asset:
                self._apply_template_asset(target_id=target_id, template_asset=asset["smpl_template"])

        if motion_asset is not None:
            self._apply_motion_asset(target_id=target_id, motion_asset=motion_asset)

    def replace_instance_with_ply(
        self,
        target_id: int,
        ply_path: str,
        new_id: int = None,
        motion_source: Optional[Union[int, str]] = None,
    ):
        # 从外部 PLY 读取新外观来替换目标实例，并按配置决定是否同时替换运动和模板。
        del new_id  # Keep the same signature as RigidNodes for edit-config compatibility.
        motion_source = self._normalize_motion_source(motion_source)

        asset = self._build_gaussian_asset_from_ply(ply_path)
        sidecar_asset, sidecar_path = self._load_sidecar_motion_asset(ply_path)

        motion_asset = self._resolve_motion_asset(motion_source)

        merged_template = self._resolve_template_asset(
            preferred_template=asset.get("smpl_template"),
            primary_asset=motion_asset,
            primary_label=str(motion_source) if motion_asset is not None else None,
            sidecar_asset=sidecar_asset,
            sidecar_label=sidecar_path,
            context="replace",
        )
        if merged_template is not None:
            asset["smpl_template"] = merged_template

        self._replace_instance_from_asset(
            target_id=target_id,
            asset=asset,
            motion_asset=motion_asset,
        )
        n = int((self.point_ids[..., 0] == target_id).sum().item())
        print(
            f"[replace_instance_with_ply] replaced SMPL instance {target_id} "
            f"from {ply_path}, points={n}, motion_source={motion_source}"
        )

    def add_instance_with_ply(
        self,
        target_id: Union[int, str],
        ply_path: str,
        offset: List[float],
        motion_path: str,
        appear_frames: Optional[int] = None,
        dataset = None,
    ) -> None:
        # 从外部 Gaussian PLY 添加一个新 SMPL 实例，并用 motion_path 提供它的逐帧运动。
        asset = self._build_gaussian_asset_from_ply(ply_path, dataset=dataset)
        motion_asset = self._load_motion_asset(motion_path)
        sidecar_asset, sidecar_path = self._load_sidecar_motion_asset(ply_path)
        template_asset = self._resolve_template_asset(
            preferred_template=asset.get("smpl_template"),
            primary_asset=motion_asset,
            primary_label=motion_path,
            sidecar_asset=sidecar_asset,
            sidecar_label=sidecar_path,
            context="add",
        )
        if template_asset is None:
            raise RuntimeError("SMPL add requires smpl_template in the Gaussian ply or motion asset")

        if not self._has_initialized_smpl_instances():
            # 空场景没有可拼接的旧 SMPL 张量，必须走单独初始化路径创建第 0 个实例。
            # 这里的 target_id 实际作为 ref_source 使用，应传入导出的 motion.pt 路径来提供参考轨迹。
            new_id = self._initialize_first_instance_from_asset(
                ref_source=target_id,
                gaussian_asset=asset,
                template_asset=template_asset,
                motion_asset=motion_asset,
                offset=offset,
                appear_frames=appear_frames,
            )
            print(
                f"[add_instance_with_ply] added SMPL instance {new_id} from {ply_path}, "
                f"ref_id={target_id}, motion_path={motion_path}"
            )
            return

        gaussian_tensors = self._prepare_gaussian_tensors(
            gaussian_asset=asset,
            expected_features_rest_shape=self._features_rest[: self.smpl_points_num].shape[1:],
        )

        offset_tensor = torch.as_tensor(
            offset,
            device=self.instances_trans.device,
            dtype=self.instances_trans.dtype,
        )
        if offset_tensor.shape != (3,):
            raise ValueError(
                f"SMPL add offset must have shape (3,), got {tuple(offset_tensor.shape)}"
            )

        ref_instances_fv, ref_instances_trans, ref_instances_quats, ref_instances_size, ref_desc = self._resolve_reference_source(
            target_id,
            appear_frames=appear_frames,
        )
        if ref_instances_trans.ndim != 3 or ref_instances_trans.shape[1:] != (1, 3):
            raise ValueError(
                f"Expected ref_instances_trans shape (N, 1, 3), got {tuple(ref_instances_trans.shape)}"
            )
        ref_instances_trans += offset_tensor.view(1, 1, 3)
        if hasattr(self, "instances_size") and ref_instances_size is None:
            ref_instances_size = (
                gaussian_tensors["_means"].max(dim=0).values
                - gaussian_tensors["_means"].min(dim=0).values
            ).unsqueeze(0)

        new_id = self._append_instance_from_asset(
            gaussian_tensors=gaussian_tensors,
            template_asset=template_asset,
            ref_instances_fv=ref_instances_fv,
            ref_instances_trans=ref_instances_trans,
            ref_instances_quats=ref_instances_quats,
            ref_instances_size=ref_instances_size,
        )
        self._apply_motion_asset(
            target_id=new_id,
            motion_asset=motion_asset,
            preserve_target_fv=True,
        )
        print(
            f"[add_instance_with_ply] added SMPL instance {new_id} from {ply_path}, "
            f"ref_id={ref_desc}, motion_path={motion_path}"
        )
             
    def export_instance_motion_to_pt(
        self,
        path: str,
        instance_id: int,
    ) -> None:
        # 只导出某个 SMPL 实例的运动数据，适合作为空场景 add 时的参考轨迹或替换 motion。
        if instance_id < 0 or instance_id >= self.num_instances:
            raise ValueError(f"Invalid instance id {instance_id}")
        asset = self._get_scene_instance_asset(
            instance_id=instance_id,
            sparse_motion=True,
            cpu=True,
        )
        del asset["gaussian"]
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(asset, path)
        print(f"[export_instance_motion_to_pt] saved motion asset to {path}")

    def export_instance_to_ply(
        self,
        path: str,
        instance_id: int = None,
        alpha_thresh: float = 0.001,
    ) -> None:
        # 导出单个 SMPL 实例的 Gaussian 外观到 PLY，并把关节、权重和 voxel 模板元数据一起写进去。
        del alpha_thresh  # SMPL asset replacement requires the full 6890-point canonical asset.
        ids = self.point_ids[..., 0]
        if instance_id is None:
            mask_inst = torch.ones_like(ids, dtype=torch.bool)
        else:
            mask_inst = ids == instance_id

        mask = mask_inst

        if mask.sum() == 0:
            print(f"[export_instance_to_ply] nothing to export for id {instance_id}")
            return

        m = self._means[mask].detach().cpu().numpy()
        sigma = self._scales[mask].detach().cpu().numpy()
        q = self._quats[mask].detach().cpu().numpy()
        op = self._opacities[mask].detach().cpu().numpy().squeeze()
        fdc = self._features_dc[mask].detach().cpu().numpy()
        frest = self._features_rest[mask].detach().cpu().numpy()
        M, K = frest.shape[0], frest.shape[1]

        if instance_id is None:
            if torch.unique(ids[mask]).numel() != 1:
                raise RuntimeError("SMPL export with template embedded requires a single instance_id")
            template_instance_id = int(ids[mask][0].item())
        else:
            template_instance_id = int(instance_id)

        weights = self.template.W[template_instance_id].detach().cpu().numpy()
        if weights.shape[0] != M:
            raise RuntimeError(
                f"SMPL export template/gaussian point mismatch: {weights.shape[0]} vs {M}"
            )

        dtype_list = [
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
            ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ]
        for k in range(K * 3):
            dtype_list.append((f"f_rest_{k}", "f4"))
        dtype_list += [
            ("opacity", "f4"),
            ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
            ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
        ]
        for j in range(weights.shape[1]):
            dtype_list.append((f"w_{j}", "f4"))

        arr = np.empty(M, dtype=dtype_list)
        arr["x"], arr["y"], arr["z"] = m[:, 0], m[:, 1], m[:, 2]
        arr["nx"].fill(0)
        arr["ny"].fill(0)
        arr["nz"].fill(0)
        arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = fdc[:, 0], fdc[:, 1], fdc[:, 2]

        if K > 0:
            frest_flat = frest.reshape(M, -1)
            for k in range(K * 3):
                arr[f"f_rest_{k}"] = frest_flat[:, k]

        arr["opacity"] = op
        arr["scale_0"], arr["scale_1"], arr["scale_2"] = sigma[:, 0], sigma[:, 1], sigma[:, 2]
        arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        for j in range(weights.shape[1]):
            arr[f"w_{j}"] = weights[:, j]

        joints = self.template.J_canonical[template_instance_id].detach().cpu().numpy()
        joint_dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
        joint_arr = np.empty(joints.shape[0], dtype=joint_dtype)
        joint_arr["x"], joint_arr["y"], joint_arr["z"] = joints[:, 0], joints[:, 1], joints[:, 2]

        ply_elements = [
            PlyElement.describe(arr, "vertex"),
            PlyElement.describe(joint_arr, "joint"),
        ]
        if self.use_voxel_deformer:
            lbs_voxel_base_tensor = self.template.voxel_deformer.lbs_voxel_base[template_instance_id].detach().cpu()
            voxel_w_correction_tensor = self.template.voxel_deformer.voxel_w_correction[template_instance_id].detach().cpu()
            lbs_voxel_base = lbs_voxel_base_tensor.numpy().reshape(-1)
            voxel_w_correction = voxel_w_correction_tensor.numpy().reshape(-1)
            offset = self.template.voxel_deformer.offset[template_instance_id].detach().cpu().numpy().reshape(-1)
            scale = self.template.voxel_deformer.scale[template_instance_id].detach().cpu().numpy().reshape(-1)

            lbs_dtype = [("value", "f4")]
            lbs_arr = np.empty(lbs_voxel_base.shape[0], dtype=lbs_dtype)
            lbs_arr["value"] = lbs_voxel_base

            corr_arr = np.empty(voxel_w_correction.shape[0], dtype=lbs_dtype)
            corr_arr["value"] = voxel_w_correction

            meta_dtype = [
                ("offset_0", "f4"),
                ("offset_1", "f4"),
                ("offset_2", "f4"),
                ("scale_0", "f4"),
                ("lbs_ndim", "i4"),
                ("lbs_dim_0", "i4"),
                ("lbs_dim_1", "i4"),
                ("lbs_dim_2", "i4"),
                ("lbs_dim_3", "i4"),
                ("lbs_dim_4", "i4"),
                ("corr_ndim", "i4"),
                ("corr_dim_0", "i4"),
                ("corr_dim_1", "i4"),
                ("corr_dim_2", "i4"),
                ("corr_dim_3", "i4"),
                ("corr_dim_4", "i4"),
            ]
            if lbs_voxel_base_tensor.ndim > 5 or voxel_w_correction_tensor.ndim > 5:
                raise RuntimeError(
                    "Voxel deformer tensors with more than 5 dims cannot be exported to ply metadata"
                )

            meta_arr = np.zeros(1, dtype=meta_dtype)
            meta_arr["offset_0"] = offset[0]
            meta_arr["offset_1"] = offset[1]
            meta_arr["offset_2"] = offset[2]
            meta_arr["scale_0"] = scale[0]
            meta_arr["lbs_ndim"] = lbs_voxel_base_tensor.ndim
            meta_arr["corr_ndim"] = voxel_w_correction_tensor.ndim
            for dim in range(5):
                meta_arr[f"lbs_dim_{dim}"] = 1
                meta_arr[f"corr_dim_{dim}"] = 1
            for dim, dim_size in enumerate(lbs_voxel_base_tensor.shape):
                meta_arr[f"lbs_dim_{dim}"] = dim_size
            for dim, dim_size in enumerate(voxel_w_correction_tensor.shape):
                meta_arr[f"corr_dim_{dim}"] = dim_size

            ply_elements.extend(
                [
                    PlyElement.describe(lbs_arr, "voxel_lbs_base"),
                    PlyElement.describe(corr_arr, "voxel_w_correction"),
                    PlyElement.describe(meta_arr, "voxel_meta"),
                ]
            )

        PlyData(ply_elements, text=False).write(path)

        print(f"[export_instance_to_ply] saved {arr.shape[0]} gaussians to {path}")
