"""
Template for an LLM-generated MimicKit task module.

A task module defines exactly two functions, always with these names:

    compute_reward(state)   -> (total, terms)       total [N], terms {name: [N]}
    check_success(state)    -> success              bool  [N]

State
-----
`state` is a dict whose tensors have leading dim N, the current batch.
N is NOT fixed: it may be 1, 7, 4096, or anything else.
Units are meters (m), meters per second (m/s), radians per second (rad/s)
and seconds (s). The world frame is z up; the ground is z = 0.

    root_pos       [N, 3]      m, world frame
    prev_root_pos  [N, 3]      m, world frame, root_pos one step earlier
    root_rot       [N, 4]      root rotation quaternion (x, y, z, w), world frame
    root_vel       [N, 3]      m/s, world frame
    root_ang_vel   [N, 3]      rad/s, world frame
    dt             float       s, time between prev_root_pos and root_pos
    key_body_names list[str]   ["head", "right_hand", "left_hand",
                                "right_foot", "left_foot", "sword"]
    key_body_pos   [N, K, 3]   m, world frame, in key_body_names order

Look up a key body by name, never by a hardcoded index:

    sword_pos = state["key_body_pos"][:, state["key_body_names"].index("sword"), :]

Each key body position is the origin of that body's frame:
    sword        the blade tip (the hilt is held in the right hand)
    head         the base of the head at the neck; the center of the head
                 is about 0.175 m higher
    right_hand,  the wrists
    left_hand
    right_foot,  the ankles
    left_foot

Task keys
---------
Some tasks add their own keys to `state`:

    Location   tar_pos    [N, 3]   m, world frame; only x and y are meaningful
               tar_speed  [N]      m/s, desired speed toward the target (1.0)
    Steering   tar_dir    [N, 2]   unit vector in the world's horizontal (x, y) plane
               face_dir   [N, 2]   unit vector in the world's horizontal (x, y) plane
               tar_speed  [N]      m/s, desired speed along tar_dir

Any other task gets only the keys above, plus the keys its task prompt lists.

Helpers
-------
Only these functions from `import util.torch_util as torch_util` may be used.
Quaternions are (x, y, z, w), and every input has a leading batch dim N.

    calc_heading(q)          -> [N]     yaw angle about z, rad
    calc_heading_quat(q)     -> [N, 4]  yaw-only rotation, heading frame -> world
    calc_heading_quat_inv(q) -> [N, 4]  inverse, world -> heading frame
    quat_rotate(q, v)        -> [N, 3]  rotates 3D vectors v by q
    normalize_angle(x)       -> same    wraps angles to [-pi, pi]

Rules
-----
1. Never hardcode N (e.g. 4096). Take shapes from the inputs and create new
   tensors with *_like ops so they share the device and dtype.
2. Never modify a tensor in `state`. In-place writes to tensors your own code
   created are fine, e.g. `d = torch.zeros_like(x); d[..., 2] = 1.0`.
   No Python loops over envs, no .item(), no .cpu().
3. Every term in `terms` and the total must stay in [0, 1]
   (REWARD_MIN..REWARD_MAX) every step. The HLC trains on
   0.9 * task + 0.1 * style, and the style reward is about 0.25 per step.
   A task reward on a 0-10 scale drowns out style and the motion becomes
   unnatural; a 0-0.01 scale drowns out the task. Build each term in [0, 1]
   (exp(-k * err), clamp(x, 0, 1)) and combine them with nonnegative weights
   that sum to 1. Do NOT clamp the total: a clamp hides an out-of-range term,
   which then goes flat with nothing to learn from. The execution check tests
   every term and the total, and rejects modules that break this rule.
4. No NaN or Inf: add an eps when normalizing or dividing by a distance.

check_success is a per-step condition: "is the task accomplished right now?"
It returns a torch.bool tensor [N] and keeps no memory across steps.

The functions below implement "raise the sword above the head" as a working
example. Lines marked `# EDIT:` are the parts to change for a new task.
"""

import torch

import util.torch_util as torch_util

REWARD_FN = "compute_reward"
SUCCESS_FN = "check_success"

REWARD_MIN = 0.0
REWARD_MAX = 1.0


def compute_reward(state):
    # EDIT: task inputs
    names = state["key_body_names"]
    key_body_pos = state["key_body_pos"]
    head_pos = key_body_pos[:, names.index("head"), :]
    hand_pos = key_body_pos[:, names.index("right_hand"), :]
    sword_pos = key_body_pos[:, names.index("sword"), :]
    root_vel = state["root_vel"]

    # EDIT: task parameters
    tar_height = 0.5
    height_err_scale = 10.0
    vel_err_scale = 2.0

    # EDIT: term weights, nonnegative and summing to 1
    height_w = 0.5
    upright_w = 0.3
    still_w = 0.2

    # EDIT: reward terms, each in [0, 1]
    sword_height = sword_pos[..., 2] - head_pos[..., 2]
    height_err = torch.clamp_min(tar_height - sword_height, 0.0)
    height_r = torch.exp(-height_err_scale * height_err * height_err)

    blade = sword_pos - hand_pos
    blade_dir = blade / (torch.norm(blade, dim=-1, keepdim=True) + 1e-6)
    upright_r = torch.clamp(blade_dir[..., 2], 0.0, 1.0)

    planar_speed_sq = torch.sum(root_vel[..., 0:2] * root_vel[..., 0:2], dim=-1)
    still_r = torch.exp(-vel_err_scale * planar_speed_sq)

    terms = {
        "height": height_r,
        "upright": upright_r,
        "still": still_r,
    }

    total = height_w * height_r + upright_w * upright_r + still_w * still_r
    return total, terms

def check_success(state):
    # EDIT: task inputs
    names = state["key_body_names"]
    key_body_pos = state["key_body_pos"]
    head_pos = key_body_pos[:, names.index("head"), :]
    hand_pos = key_body_pos[:, names.index("right_hand"), :]
    sword_pos = key_body_pos[:, names.index("sword"), :]

    # EDIT: success condition
    min_height = 0.3
    min_upright = 0.8

    sword_height = sword_pos[..., 2] - head_pos[..., 2]
    blade = sword_pos - hand_pos
    blade_dir = blade / (torch.norm(blade, dim=-1, keepdim=True) + 1e-6)

    success = (sword_height > min_height) & (blade_dir[..., 2] > min_upright)
    return success
