# Python wrapper that mirrors your checkers.py shape, but calls into Mojo.
import gymnasium as gym
import numpy as np

import os
import sys

# The Mojo importer module will handle compilation of the Mojo files.
import mojo.importer  # noqa: F401

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import the Mojo module named `checkers` (from checkers.mojo)
import checkers_mojo

def mojo_to_numpy(mojo_list, dtype):
    """
    Convert a Mojo list to a numpy array.
    """
    return np.array(mojo_list.split(","), dtype=dtype)

class CheckersMojoEnv:
    """
    A minimal, vectorless wrapper around the Mojo Checkers env.
    It exposes reset/step/render/close and the same buffer shapes as your original.
    """
    def __init__(self, size: int = 8):
        self.size = int(size)
        # Construct the Mojo env
        self._env = checkers_mojo.make_env(self.size)

        # Expose buffers as numpy arrays that view/copy from Mojo lists as needed
        self.single_observation_space = gym.spaces.Box(low=0, high=1, shape=(self.size * self.size,), dtype=np.uint8)
        self.single_action_space = gym.spaces.Discrete(self.size * self.size * 8)

    @property
    def observations(self) -> np.ndarray:
        return mojo_to_numpy(self._env.get_observations(), np.uint8)

    @property
    def rewards(self) -> np.ndarray:
        return mojo_to_numpy(self._env.get_rewards(), np.float32)

    @property
    def terminals(self) -> np.ndarray:
        return mojo_to_numpy(self._env.get_terminals(), np.uint8)

    def reset(self, seed: int | None = None):
        self._env.c_reset()
        return self.observations, {}

    def step(self, actions):
        # Write action into Mojo env
        self._env.c_set_actions(actions)
        self._env.c_step()
        return self.observations, self.rewards, self.terminals, np.array([0], dtype=np.uint8), {}

    def render(self):
        self._env.c_render()

    def close(self):
        self._env.c_close()


# Example usage
if __name__ == "__main__":
    env = CheckersMojoEnv(size=8)
    obs, _ = env.reset()
    done = False
    total = 0.0

    for _ in range(20000):
        action = np.random.randint(0, env.single_action_space.n)
        obs, rew, term, trunc, info = env.step(action)
        total += float(rew[0])
        if term[0]:
            obs, _ = env.reset()
    print("Total reward in 200 steps:", total)