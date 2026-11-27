"""
Assembles the `state` dict handed to LLM-generated task modules.

The layout must match mimickit/llm/templates/task_template.py exactly: the
template is the prompt, so any difference in key names, shapes or key body order
means generated code reads keys that don't exist or the wrong body.
"""

import numbers

import torch

# key bodies in template order; must equal the env config's key_bodies
KEY_BODY_NAMES = ["head", "right_hand", "left_hand", "right_foot", "left_foot", "sword"]

# key -> trailing shape after the leading batch dim N
BASE_KEYS = {
    "root_pos": (3,),
    "prev_root_pos": (3,),
    "root_rot": (4,),
    "root_vel": (3,),
    "root_ang_vel": (3,),
    "key_body_pos": (len(KEY_BODY_NAMES), 3),
}

# task name -> {key: trailing shape}
TASK_KEYS = {
    "location": {
        "tar_pos": (3,),
        "tar_speed": (),
    },
    "steering": {
        "tar_dir": (2,),
        "face_dir": (2,),
        "tar_speed": (),
    },
}

DT_KEY = "dt"
KEY_BODY_NAMES_KEY = "key_body_names"


def check_key_bodies(key_bodies):
    """
    Call once when the env starts, with the config's key_bodies list.
    A reordered config would otherwise silently swap bodies (e.g. hand and sword).
    """
    if (list(key_bodies) != KEY_BODY_NAMES):
        raise ValueError("task state: config key_bodies {} must equal the template's "
                         "key_body_names {}, in the same order".format(list(key_bodies), KEY_BODY_NAMES))
    return


def assemble_task_state(raw):
    """
    raw: dict with the base keys, dt, and optionally one task group's keys.
         Location's tar_speed may be a single number; it is expanded to [N].
    Returns a new dict laid out as the template describes. Every tensor is a
    float32 copy, so generated code can't write into engine buffers.
    """
    task_keys = _check_keys(raw)

    root_pos = raw["root_pos"]
    if (not torch.is_tensor(root_pos)):
        raise ValueError("task state 'root_pos': expected a torch.Tensor, got {}".format(type(root_pos).__name__))
    if (root_pos.dim() < 1):
        raise ValueError("task state 'root_pos': expected [N, 3], got {}".format(list(root_pos.shape)))

    n = root_pos.shape[0]
    device = root_pos.device

    shapes = dict(BASE_KEYS)
    shapes.update(task_keys)

    state = dict()
    for key, trailing in shapes.items():
        val = raw[key]

        if (key == "tar_speed"):
            val = _expand_scalar(val, n, device)

        state[key] = _check_and_copy(key, val, n, trailing, device)

    try:
        dt = float(raw[DT_KEY])
    except (TypeError, ValueError, RuntimeError):
        raise ValueError("task state '{}': expected a positive float, got {}".format(DT_KEY, raw[DT_KEY]))
    if (not dt > 0.0):
        raise ValueError("task state '{}': expected a positive float, got {}".format(DT_KEY, dt))
    state[DT_KEY] = dt

    state[KEY_BODY_NAMES_KEY] = list(KEY_BODY_NAMES)
    return state


def _check_keys(raw):
    required = set(BASE_KEYS.keys()) | {DT_KEY}

    for key in sorted(required):
        if (key not in raw):
            raise ValueError("task state: missing key '{}'".format(key))

    extra = set(raw.keys()) - required
    if (len(extra) == 0):
        return dict()

    for task_keys in TASK_KEYS.values():
        if (extra == set(task_keys.keys())):
            return task_keys

    # no group matched: name the first key that can't be explained
    known = set()
    for task_keys in TASK_KEYS.values():
        known |= set(task_keys.keys())

    unknown = sorted(extra - known)
    if (len(unknown) > 0):
        raise ValueError("task state: unknown key '{}'".format(unknown[0]))

    for task_name, task_keys in TASK_KEYS.items():
        group = set(task_keys.keys())
        if (len(extra & (group - {"tar_speed"})) > 0):
            missing = sorted(group - extra)
            if (len(missing) > 0):
                raise ValueError("task state: {} task is missing key '{}'".format(task_name, missing[0]))
            stray = sorted(extra - group)
            raise ValueError("task state: key '{}' does not belong to the {} task".format(stray[0], task_name))

    raise ValueError("task state: keys {} do not form a known task ({})".format(sorted(extra), list(TASK_KEYS.keys())))


def _expand_scalar(val, n, device):
    # Location stores tar_speed as one number; the template promises [N]
    if (isinstance(val, numbers.Real) and not isinstance(val, bool)):
        val = torch.full([n], float(val), device=device, dtype=torch.float32)
    elif (torch.is_tensor(val) and val.dim() == 0):
        val = torch.full([n], float(val.item()), device=device, dtype=torch.float32)
    return val


def _check_and_copy(key, val, n, trailing, device):
    expected = [n] + list(trailing)
    expected_str = "[N" + "".join([", {}".format(d) for d in trailing]) + "]"

    if (not torch.is_tensor(val)):
        raise ValueError("task state '{}': expected a torch.Tensor {}, got {}".format(key, expected_str, type(val).__name__))

    if (list(val.shape) != expected):
        raise ValueError("task state '{}': expected {} with N={}, got {}".format(key, expected_str, n, list(val.shape)))

    if (val.device != device):
        raise ValueError("task state '{}': on device {}, but root_pos is on {}".format(key, val.device, device))

    # always a copy, even if val is already float32
    return val.detach().to(dtype=torch.float32, copy=True)
