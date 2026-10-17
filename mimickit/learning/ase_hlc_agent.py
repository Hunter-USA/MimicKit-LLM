import learning.ppo_agent as ppo_agent
from mimickit.learning import agent_builder
import gymnasium.spaces as spaces
import numpy as np
from mimickit.learning.environment_view import EnvironmentView

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

    def _build_llc(self):
        self._llc_agent = None
        return