import numpy as np
import torch

import envs.base_env as base_env
import envs.task_steering_env as task_steering_env
import util.stats_tracker as stats_tracker
import util.task_metrics as task_metrics

class TaskSteeringMetricEnv(task_steering_env.TaskSteeringEnv):
    """
    TaskSteeringEnv with per-step velocity error, facing error, fraction of time
    within tolerance, and per-episode fall rate.

    Errors are sampled every step against the current command, skipping the first
    metric_skip_time seconds after each command change (including episode start)
    and any steps after the episode has ended. Training behavior (obs, reward, done)
    is unchanged.
    """

    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):
        self._metric_skip_time = env_config.get("metric_skip_time", 1.0)
        self._metric_vel_tol = env_config.get("metric_vel_tol", 0.3)
        self._metric_face_tol = np.deg2rad(env_config.get("metric_face_tol_deg", 30.0))

        super().__init__(env_config=env_config, engine_config=engine_config,
                         num_envs=num_envs, device=device, visualize=visualize,
                         record_video=record_video)
        return

    def set_mode(self, mode):
        super().set_mode(mode)

        if (self._mode == base_env.EnvMode.TEST):
            self._reset_metric_trackers()
        return

    def record_diagnostics(self):
        self._diagnostics["steer_vel_err"] = self._vel_err_tracker.get_mean()[0].clone()
        self._diagnostics["steer_face_err_deg"] = torch.rad2deg(self._face_err_tracker.get_mean()[0])
        self._diagnostics["steer_in_tol_frac"] = self._in_tol_tracker.get_mean()[0].clone()
        self._diagnostics["steer_eps"] = self._num_eps.clone()
        self._diagnostics["steer_fall_rate"] = self._num_eps_fell.float() / self._num_eps.clamp_min(1).float()

        diag = super().record_diagnostics()
        return diag

    def _build_sim_tensors(self, config):
        super()._build_sim_tensors(config)

        num_envs = self.get_num_envs()
        device = self._device

        # time at which the current command was issued
        self._cmd_start_time = torch.zeros([num_envs], device=device, dtype=torch.float)

        # sticky episode-end state, needed because the HLC steps the env several
        # times per action without resetting in between
        self._ep_ended = torch.zeros([num_envs], device=device, dtype=torch.bool)
        self._ep_end_reason = torch.full([num_envs], base_env.DoneFlags.NULL.value, device=device, dtype=torch.int)

        # trackers
        self._vel_err_tracker = stats_tracker.StatsTracker(1, device)
        self._face_err_tracker = stats_tracker.StatsTracker(1, device)
        self._in_tol_tracker = stats_tracker.StatsTracker(1, device)

        # running totals, kept on device to avoid per-step syncs
        self._num_eps = torch.zeros([], device=device, dtype=torch.long)
        self._num_eps_fell = torch.zeros([], device=device, dtype=torch.long)
        return

    def _reset_metric_trackers(self):
        self._vel_err_tracker.reset()
        self._face_err_tracker.reset()
        self._in_tol_tracker.reset()
        self._num_eps.zero_()
        self._num_eps_fell.zero_()
        return

    def _reset_task(self, env_ids):
        super()._reset_task(env_ids)
        self._cmd_start_time[env_ids] = self._time_buf[env_ids]
        return

    def _update_task(self):
        # measure against the current command, before the parent may replace it
        live = ~self._ep_ended & (self._time_buf >= self._cmd_start_time + self._metric_skip_time)
        live_ids = live.nonzero(as_tuple=False).flatten()
        if (len(live_ids) > 0):
            char_id = self._get_char_id()
            root_pos = self._engine.get_root_pos(char_id)[live_ids]
            root_rot = self._engine.get_root_rot(char_id)[live_ids]

            vel_err = task_metrics.compute_steering_vel_err(root_pos, self._prev_root_pos[live_ids],
                                                            self._tar_dir[live_ids], self._tar_speed[live_ids],
                                                            self._engine.get_timestep())
            face_err = task_metrics.compute_steering_facing_err(root_rot, self._face_dir[live_ids])
            in_tol = (vel_err <= self._metric_vel_tol) & (face_err <= self._metric_face_tol)

            self._vel_err_tracker.update(vel_err.unsqueeze(-1))
            self._face_err_tracker.update(face_err.unsqueeze(-1))
            self._in_tol_tracker.update(in_tol.float().unsqueeze(-1))

        super()._update_task()
        return

    def _update_done(self):
        super()._update_done()

        newly_done = (self._done_buf != base_env.DoneFlags.NULL.value) & ~self._ep_ended
        self._ep_ended |= newly_done
        self._ep_end_reason[:] = torch.where(newly_done, self._done_buf, self._ep_end_reason)
        return

    def _reset_envs(self, env_ids):
        if (len(env_ids) > 0):
            # forced resets (no end reason) are not counted
            reason = self._ep_end_reason[env_ids]
            self._num_eps += (reason != base_env.DoneFlags.NULL.value).sum()
            self._num_eps_fell += (reason == base_env.DoneFlags.FAIL.value).sum()

            self._ep_ended[env_ids] = False
            self._ep_end_reason[env_ids] = base_env.DoneFlags.NULL.value

        # parent calls _reset_task, which restarts the command timer
        super()._reset_envs(env_ids)
        return
