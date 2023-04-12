from typing import Union

import gym

from core.envs import VecNormalize

def get_render_func(venv: gym.Env):
    """
    Get render function for the environment.

    Args:
        venv (object): Environment in which

    Returns:
        Callable
    """
    if hasattr(venv, "envs"):
        return venv.envs[0].render
    elif hasattr(venv, "venv"):
        return get_render_func(venv.venv)
    elif hasattr(venv, "env"):
        return get_render_func(venv.env)

    return None

def get_vec_normalize(venv: gym.Env) -> Union[VecNormalize, None]:
    """
    Given an environment, wraps it in a normalized environment wrapper.

    Args:
        venv (gym.Env): Gym environment to normalize.

    Returns:
        gym.Env
    """
    if isinstance(venv, VecNormalize):
        return venv

    elif hasattr(venv, "venv"):
        return get_vec_normalize(venv.venv)

    return None
