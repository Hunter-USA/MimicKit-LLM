import learning.ppo_agent as ppo_agent
import learning.agent_builder as agent_builder
import gymnasium.spaces as spaces
import numpy as np
from learning.environment_view import EnvironmentView
import torch
import learning.ase_agent as ase_agent
from util.logger import Logger
import envs.base_env as base_env

class ASEHLCAgent(ppo_agent.PPOAgent):
    def __init__(self, config, env, device):
        # Load the LLC agent configuration
        llc_config = agent_builder.load_agent_file(config["llc_agent_config"])
        latent_dim = llc_config["model"]["latent_dim"]

        # Define the action space for the HLC agent based on the latent dimension of the LLC agent
        hlc_action_space = spaces.Box(low = -1.0, high = 1.0, shape = [latent_dim], dtype = np.float32)
        hlc_env = EnvironmentView(env, action_space=hlc_action_space)
        super().__init__(config, hlc_env, device)

        # Initialize the LLC agent based on the loaded configuration
        self._task_env = env
        self._latent_dim = latent_dim
        self._llc_config = llc_config
        self._build_llc()
        return

    def _load_params(self, config):
        super()._load_params(config)
        self._llc_model_file = config["llc_model_file"]
        self._llc_steps = config["llc_steps"]
        self._task_reward_weight = config["task_reward_weight"]
        self._style_reward_weight = config["style_reward_weight"]
        return

    def _build_llc(self):
        self._llc_agent = None

        llc_state = torch.load(self._llc_model_file, map_location=self._device)
        self._llc_obs_dim = llc_state["_obs_norm._mean"].shape[-1]

        llc_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=[self._llc_obs_dim], dtype=np.float32)
        llc_env = EnvironmentView(self._task_env, obs_space=llc_obs_space)
        
        llc_agent = ase_agent.ASEAgent(config=self._llc_config, env=llc_env, device=self._device)
        llc_agent.load(self._llc_model_file)

        llc_agent.eval()
        for p in llc_agent.parameters():
            p.requires_grad_(False)

        self.__dict__["_llc_agent"] = llc_agent
        self._check_llc_sizes(llc_state)
        return

    def _check_llc_sizes(self, llc_state):
        task_obs_dim = self._task_env.get_obs_space().shape[-1]
        task_a_dim = self._task_env.get_action_space().shape[-1]
        task_disc_dim = self._task_env.get_disc_obs_space().shape[-1]

        llc_a_dim = llc_state["_a_norm._mean"].shape[-1]
        llc_disc_dim = llc_state["_disc_obs_norm._mean"].shape[-1]

        assert self._llc_obs_dim < task_obs_dim, \
            "LLC obs size {} must be smaller than task obs size {}".format(self._llc_obs_dim, task_obs_dim)
        assert llc_a_dim == task_a_dim, \
            "LLC action size {} != env action size {}".format(llc_a_dim, task_a_dim)
        assert llc_disc_dim == task_disc_dim, \
            "LLC disc obs size {} != env disc obs size {}".format(llc_disc_dim, task_disc_dim)

        Logger.print("LLC loaded: obs {} + task obs {} = {}, actions {}, disc obs {}".format(
            self._llc_obs_dim, task_obs_dim - self._llc_obs_dim, task_obs_dim, llc_a_dim, llc_disc_dim))
        return

    def _extract_llc_obs(self, obs):
        # Task envs append the task observation after the character observation.
        return obs[..., :self._llc_obs_dim]

    def _normalize_latent(self, z_raw):
        # Normalize the latent vector to be in the range [-1, 1]
        z_norm = torch.clamp(z_raw, -1.0, 1.0)
        return torch.nn.functional.normalize(z_norm, dim=-1)

    def _compute_llc_action(self, obs, z):
        llc = self._llc_agent
        with torch.no_grad():
            llc_obs = self._extract_llc_obs(obs)
            norm_obs = llc._obs_norm.normalize(llc_obs)
            norm_a = llc._model.eval_actor(norm_obs, z).mode
            a = llc._a_norm.unnormalize(norm_a)
        return a

    def _calc_style_reward(self, disc_obs):
        llc = self._llc_agent
        norm_disc_obs = llc._disc_obs_norm.normalize(disc_obs)
        return llc._calc_disc_rewards(norm_disc_obs)

    def _record_data_post_step(self, next_obs, reward, done, next_info):
        super()._record_data_post_step(next_obs, reward, done, next_info)
        self._exp_buffer.record("style_reward", next_info["style_reward"])
        return

    def _step_env(self, action):
        z = self._normalize_latent(action)

        num_envs = self.get_num_envs()
        task_reward = torch.zeros(num_envs, device=self._device)
        style_reward = torch.zeros(num_envs, device=self._device)
        done = None

        obs = self._curr_obs
        for i in range(self._llc_steps):
            llc_action = self._compute_llc_action(obs, z)
            obs, reward, d, info = self._env.step(llc_action)

            if done is None:
                done = torch.full_like(d, base_env.DoneFlags.NULL.value)
            alive = (done == base_env.DoneFlags.NULL.value)

            task_reward = torch.where(alive, reward, torch.zeros_like(reward))
            style_reward = torch.where(alive, self._calc_style_reward(info["disc_obs"]), torch.zeros_like(reward))


            done = torch.where(alive, d, done)

        task_reward /= self._llc_steps
        style_reward /= self._llc_steps
        
        out_info = dict(info)
        out_info["style_reward"] = style_reward
        return obs, task_reward, done, out_info
    