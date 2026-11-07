import torch

import util.torch_util as torch_util

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_location_dist(root_pos, tar_pos):
    # type: (Tensor, Tensor) -> Tensor
    # horizontal distance between root and target
    diff = tar_pos[..., 0:2] - root_pos[..., 0:2]
    dist = torch.norm(diff, dim=-1)
    return dist

@torch.jit.script
def compute_steering_vel_err(root_pos, prev_root_pos, tar_dir, tar_speed, dt):
    # type: (Tensor, Tensor, Tensor, Tensor, float) -> Tensor
    # || tar_speed * tar_dir - horizontal root vel ||, root vel computed as in compute_steering_reward
    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt

    tar_vel = tar_speed.unsqueeze(-1) * tar_dir
    vel_err = torch.linalg.norm(tar_vel - root_vel[..., 0:2], dim=-1)
    return vel_err

@torch.jit.script
def compute_steering_facing_err(root_rot, face_dir):
    # type: (Tensor, Tensor) -> Tensor
    # absolute wrapped angle between character heading and commanded facing direction
    heading = torch_util.calc_heading(root_rot)
    tar_heading = torch.atan2(face_dir[..., 1], face_dir[..., 0])
    facing_err = torch.abs(torch_util.normalize_angle(heading - tar_heading))
    return facing_err
