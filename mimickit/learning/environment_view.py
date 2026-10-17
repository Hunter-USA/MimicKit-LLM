class EnvironmentView:
    """A view of an environment that can override the observation and action spaces."""


    def __init__(self, env, obs_space=None, action_space=None):
        self._wrapped_env = env
        self._view_obs_space = obs_space
        self._view_action_space = action_space

    def get_obs_space(self):
        if self._view_obs_space is not None:
            return self._view_obs_space
        return self._wrapped_env.get_obs_space()

    def get_action_space(self):
        if self._view_action_space is not None:
            return self._view_action_space
        return self._wrapped_env.get_action_space()

    def __getattr__(self, name):
        if name == "_wrapped_env":
            raise AttributeError(name)
        return getattr(self._wrapped_env, name)