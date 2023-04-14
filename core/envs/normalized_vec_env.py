import numpy as np
from stable_baselines3.common.vec_env import VecNormalize as VecNormalize_


class NormalizedVecEnv(VecNormalize_):

    def __init__(self, *args, **kwargs):
        super(NormalizedVecEnv, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        if self.obs_rms:
            if self.training and update:
                self.obs_rms.update(obs)
            obs = np.clip(
                (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon),
                -self.clip_obs,
                self.clip_obs,
            )
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False
