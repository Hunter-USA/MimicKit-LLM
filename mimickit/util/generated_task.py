"""
Loads, checks and tracks one LLM-generated task module.

Its reward is checked on every call (structure, finite, within [reward_min, reward_max]), 
and per-term means plus the success fraction are tracked under fixed diagnostic keys.
"""

import hashlib
import importlib.util
import os
import re
import sys

import torch

import util.stats_tracker as stats_tracker
from util.logger import Logger

DEFAULT_REWARD_FN = "compute_reward"
DEFAULT_SUCCESS_FN = "check_success"
TERM_PREFIX = "rew_"
SUCC_KEY = "llm_succ_frac"
HASH_LEN = 12

class GeneratedTask():
    def __init__(self, path, reward_min=0.0, reward_max=1.0, tol=1e-4):
        # bounds come from the caller (code or env config), never from the generated
        # file, so a candidate can't widen its own allowed range
        self._reward_min = float(reward_min)
        self._reward_max = float(reward_max)
        self._tol = float(tol)
        assert(self._reward_max >= self._reward_min)

        self._path = os.path.abspath(path)
        with open(self._path, "rb") as f:
            src = f.read()
        self._hash = hashlib.sha256(src).hexdigest()[:HASH_LEN]

        self._module = self._load_module(src)
        self._reward_fn = self._find_fn("REWARD_FN", DEFAULT_REWARD_FN)
        self._success_fn = self._find_fn("SUCCESS_FN", DEFAULT_SUCCESS_FN)

        self._raw_names = None
        self._term_names = None
        self._term_tracker = None
        self._succ_tracker = None

        Logger.print("Loaded generated task {} (sha256 {})".format(self._path, self._hash))
        return

    def get_path(self):
        return self._path

    def get_hash(self):
        return self._hash

    def get_term_names(self):
        return list(self._term_names) if (self._term_names is not None) else None

    def init_terms(self, state):
        """
        Call once at the end of the env's __init__. Reads the term names and builds
        the trackers. Only the structure is checked: the env hasn't been reset yet,
        so a sensible reward may still produce odd values here.
        """
        if (self._raw_names is not None):
            raise RuntimeError("{}: init_terms was already called".format(self._path))

        n = self._get_batch_size(state)
        out = self._reward_fn(state)
        _, terms = self._check_reward_structure(out, n, expected_names=None)

        raw_names = list(terms.keys())
        term_names = []
        seen = dict()
        for raw_name in raw_names:
            name = self._clean_name(raw_name)
            if (name in seen):
                raise ValueError("{}: reward terms '{}' and '{}' both become '{}'".format(
                    self._path, seen[name], raw_name, name))
            seen[name] = raw_name
            term_names.append(name)

        device = state["root_pos"].device
        self._raw_names = raw_names
        self._term_names = term_names
        self._term_tracker = stats_tracker.StatsTracker(len(term_names), device)
        self._succ_tracker = stats_tracker.StatsTracker(1, device)

        Logger.print("Generated task reward terms: {}".format(term_names))
        return

    def compute_reward(self, state):
        self._check_initialized()

        n = self._get_batch_size(state)
        out = self._reward_fn(state)
        total, terms = self._check_reward_structure(out, n, expected_names=self._raw_names)
        self._check_reward_values(total, terms)
        return total, terms

    def check_success(self, state):
        n = self._get_batch_size(state)
        success = self._success_fn(state)

        if (not torch.is_tensor(success) or success.dtype != torch.bool or list(success.shape) != [n]):
            raise ValueError("{}: check_success must return a torch.bool tensor [N] with N={}, got {}".format(
                self._path, n, self._describe(success)))
        return success

    def update(self, terms, success, mask):
        """
        Track only the rows selected by mask (bool [N]).
        """
        self._check_initialized()

        cols = [terms[raw_name].float() for raw_name in self._raw_names]
        cols.append(success.float())
        stacked = torch.stack(cols, dim=-1)

        rows = stacked[mask]
        if (rows.shape[0] > 0):
            num_terms = len(self._raw_names)
            self._term_tracker.update(rows[:, :num_terms])
            self._succ_tracker.update(rows[:, num_terms:])
        return

    def reset(self):
        self._check_initialized()
        self._term_tracker.reset()
        self._succ_tracker.reset()
        return

    def get_diagnostics(self):
        self._check_initialized()

        means = torch.cat([self._term_tracker.get_mean(), self._succ_tracker.get_mean()]).tolist()

        diag = dict()
        for i, name in enumerate(self._term_names):
            diag[name] = means[i]
        diag[SUCC_KEY] = means[-1]
        return diag

    def _load_module(self, src):
        # a unique name per content keeps candidates loaded in one process from
        # overwriting each other in sys.modules
        name = "llm_task_{}".format(self._hash)
        spec = importlib.util.spec_from_file_location(name, self._path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module

        # execute the bytes that were hashed, rather than spec.loader.exec_module, which
        # re-reads the file and may reuse a stale __pycache__ .pyc (validated only by
        # mtime and size), so the logged hash is always the code that runs
        try:
            exec(compile(src, self._path, "exec"), module.__dict__)
        except BaseException:
            sys.modules.pop(name, None)
            raise

        return module

    def _find_fn(self, attr_name, default_name):
        fn_name = getattr(self._module, attr_name, default_name)
        if (not isinstance(fn_name, str)):
            raise ValueError("{}: {} must be a string, got {}".format(self._path, attr_name, type(fn_name).__name__))

        fn = getattr(self._module, fn_name, None)
        if (fn is None):
            raise ValueError("{}: function '{}' not found".format(self._path, fn_name))
        if (isinstance(fn, type) or not callable(fn)):
            raise ValueError("{}: '{}' is not a function ({})".format(self._path, fn_name, type(fn).__name__))
        return fn

    def _check_initialized(self):
        if (self._raw_names is None):
            raise RuntimeError("{}: call init_terms before using the reward".format(self._path))
        return

    def _get_batch_size(self, state):
        return state["root_pos"].shape[0]

    def _check_reward_structure(self, out, n, expected_names):
        if (not isinstance(out, tuple) or len(out) != 2):
            raise ValueError("{}: compute_reward must return a tuple (total, terms), got {}".format(
                self._path, self._describe(out)))

        total, terms = out
        self._check_reward_tensor("total", total, n)

        if (not isinstance(terms, dict)):
            raise ValueError("{}: reward terms must be a dict, got {}".format(self._path, type(terms).__name__))

        for name, val in terms.items():
            if (not isinstance(name, str)):
                raise ValueError("{}: reward term name {!r} is not a string".format(self._path, name))
            self._check_reward_tensor("term '{}'".format(name), val, n)

        if (expected_names is not None and set(terms.keys()) != set(expected_names)):
            missing = sorted(set(expected_names) - set(terms.keys()))
            extra = sorted(set(terms.keys()) - set(expected_names))
            raise ValueError("{}: reward terms changed since init (missing {}, extra {})".format(
                self._path, missing, extra))

        return total, terms

    def _check_reward_tensor(self, label, val, n):
        if (not torch.is_tensor(val) or not torch.is_floating_point(val) or list(val.shape) != [n]):
            raise ValueError("{}: reward {} must be a float tensor [N] with N={}, got {}".format(
                self._path, label, n, self._describe(val)))
        return

    def _check_reward_values(self, total, terms):
        lo = self._reward_min - self._tol
        hi = self._reward_max + self._tol

        vals = torch.stack([total.float()] + [terms[raw_name].float() for raw_name in self._raw_names])
        ok = torch.isfinite(vals) & (vals >= lo) & (vals <= hi)

        # a single GPU sync per call
        if (not bool(torch.all(ok))):
            row, env_id = (~ok).nonzero(as_tuple=False)[0].tolist()
            label = "total" if (row == 0) else "term '{}'".format(self._raw_names[row - 1])
            bad_val = vals[row, env_id].item()
            raise ValueError("{}: reward {} = {} at env {}, expected finite in [{}, {}]".format(
                self._path, label, bad_val, env_id, self._reward_min, self._reward_max))
        return

    def _clean_name(self, raw_name):
        # MimicKit title-cases diagnostic keys, so 'height' is logged as 'Rew_Height'
        name = raw_name.lower()
        name = re.sub(r"[^a-z0-9_]", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")
        if (len(name) == 0):
            raise ValueError("{}: reward term name {!r} has no letters or digits".format(self._path, raw_name))
        return TERM_PREFIX + name

    def _describe(self, val):
        if (torch.is_tensor(val)):
            return "tensor {} {}".format(val.dtype, list(val.shape))
        return type(val).__name__
