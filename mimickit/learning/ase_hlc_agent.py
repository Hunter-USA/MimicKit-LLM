import learning.ppo_agent as ppo_agent
from mimickit.learning import agent_builder
import gymnasium.spaces as spaces
import numpy as np
from mimickit.learning.environment_view import EnvironmentView
import torch
import learning.ase_agent as ase_agent

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
        return