import torch

import envs.base_env as base_env
import envs.task_location_env as task_location_env
import util.stats_tracker as stats_tracker
import util.task_metrics as task_metrics

class TaskLocationMetricEnv(task_location_env.TaskLocationEnv):
    """
    TaskLocationEnv with a per-target-segment reach success rate.

    A segment opens whenever a new target is placed and closes when the target
    changes or the episode resets. Segments that spawn within reach, segments cut
    off by the episode time limit, and segments interrupted by forced resets are
    not counted. Training behavior (obs, reward, done) is unchanged.
    """

    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):
        # needed by _build_sim_tensors, which runs inside super().__init__
        self._metric_reach_dist = env_config.get("metric_reach_dist", 0.5)

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
        self._diagnostics["loc_succ_rate"] = self._succ_tracker.get_mean()[0].clone()
        self._diagnostics["loc_final_dist"] = self._dist_tracker.get_mean()[0].clone()
        self._diagnostics["loc_segs_counted"] = self._num_counted.clone()
        self._diagnostics["loc_segs_fell"] = self._num_fell.clone()
        self._diagnostics["loc_segs_excl_near"] = self._num_excl_near.clone()
        self._diagnostics["loc_segs_excl_time"] = self._num_excl_time.clone()
        self._diagnostics["loc_eps"] = self._num_eps.clone()
        self._diagnostics["loc_fall_rate"] = self._num_eps_fell.float() / self._num_eps.clamp_min(1).float()

        diag = super().record_diagnostics()
        return diag

    def _build_sim_tensors(self, config):
        super()._build_sim_tensors(config)

        num_envs = self.get_num_envs()
        device = self._device

        # per-env segment state
        self._seg_active = torch.zeros([num_envs], device=device, dtype=torch.bool)
        self._seg_reached = torch.zeros([num_envs], device=device, dtype=torch.bool)
        self._seg_start_dist = torch.zeros([num_envs], device=device, dtype=torch.float)

        # sticky episode-end state, needed because the HLC steps the env several
        # times per action without resetting in between
        self._ep_ended = torch.zeros([num_envs], device=device, dtype=torch.bool)
        self._ep_end_reason = torch.full([num_envs], base_env.DoneFlags.NULL.value, device=device, dtype=torch.int)

        # trackers
        self._succ_tracker = stats_tracker.StatsTracker(1, device)
        self._dist_tracker = stats_tracker.StatsTracker(1, device)

        # running totals, kept on device to avoid per-step syncs
        self._num_counted = torch.zeros([], device=device, dtype=torch.long)
        self._num_fell = torch.zeros([], device=device, dtype=torch.long)
        self._num_excl_near = torch.zeros([], device=device, dtype=torch.long)
        self._num_excl_time = torch.zeros([], device=device, dtype=torch.long)
        self._num_eps = torch.zeros([], device=device, dtype=torch.long)
        self._num_eps_fell = torch.zeros([], device=device, dtype=torch.long)
        return

    def _reset_metric_trackers(self):
        self._succ_tracker.reset()
        self._dist_tracker.reset()
        self._num_counted.zero_()
        self._num_fell.zero_()
        self._num_excl_near.zero_()
        self._num_excl_time.zero_()
        self._num_eps.zero_()
        self._num_eps_fell.zero_()
        return

    def _reset_task(self, env_ids):
        super()._reset_task(env_ids)

        # open a segment for the new target, unless the episode is already over
        # (targets still change during the leftover substeps after a done)
        open_ids = env_ids[~self._ep_ended[env_ids]]
        if (len(open_ids) > 0):
            char_id = self._get_char_id()
            root_pos = self._engine.get_root_pos(char_id)

            self._seg_active[open_ids] = True
            self._seg_reached[open_ids] = False
            self._seg_start_dist[open_ids] = task_metrics.compute_location_dist(root_pos[open_ids],
                                                                                self._tar_pos[open_ids])
        return

    def _update_task(self):
        live = self._seg_active & ~self._ep_ended

        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        dist = task_metrics.compute_location_dist(root_pos, self._tar_pos)
        self._seg_reached |= live & (dist <= self._metric_reach_dist)

        # close segments whose target is about to change (same test as the parent),
        # while the old target is still in place
        change = live & (self._time_buf >= self._tar_change_times)
        change_ids = change.nonzero(as_tuple=False).flatten()
        if (len(change_ids) > 0):
            counted = torch.ones_like(change_ids, dtype=torch.bool)
            keep = self._close_segments(change_ids, counted)

            final_dist = dist[change_ids][keep]
            if (final_dist.shape[0] > 0):
                self._dist_tracker.update(final_dist.unsqueeze(-1))

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
            # per-episode fall rate, forced resets (no end reason) are not counted
            ep_reason = self._ep_end_reason[env_ids]
            self._num_eps += (ep_reason != base_env.DoneFlags.NULL.value).sum()
            self._num_eps_fell += (ep_reason == base_env.DoneFlags.FAIL.value).sum()

            open_ids = env_ids[self._seg_active[env_ids]]
            if (len(open_ids) > 0):
                reason = self._ep_end_reason[open_ids]
                ended = reason != base_env.DoneFlags.NULL.value
                ended_ids = open_ids[ended]

                if (len(ended_ids) > 0):
                    ended_reason = reason[ended]
                    fell = ended_reason == base_env.DoneFlags.FAIL.value
                    near = self._seg_start_dist[ended_ids] <= self._metric_reach_dist
                    reached = self._seg_reached[ended_ids]

                    # falls are counted (success if already reached), anything else
                    # (time limit) is excluded
                    self._num_fell += (fell & ~near & ~reached).sum()
                    self._num_excl_time += (~fell & ~near).sum()
                    self._close_segments(ended_ids, fell)

                # forced resets (no end reason): discard without counting
                self._seg_active[open_ids] = False

            self._ep_ended[env_ids] = False
            self._ep_end_reason[env_ids] = base_env.DoneFlags.NULL.value

        # parent calls _reset_task, which opens the next segment
        super()._reset_envs(env_ids)
        return

    def _close_segments(self, env_ids, counted):
        # segments that started within reach would be free successes
        near = self._seg_start_dist[env_ids] <= self._metric_reach_dist
        self._num_excl_near += near.sum()

        keep = counted & ~near
        self._num_counted += keep.sum()

        reached = self._seg_reached[env_ids][keep]
        if (reached.shape[0] > 0):
            self._succ_tracker.update(reached.float().unsqueeze(-1))

        self._seg_active[env_ids] = False
        return keep
