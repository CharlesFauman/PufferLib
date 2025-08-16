# Python wrapper that mirrors your checkers.py shape, but calls into Mojo.
# Requires: `pip install mojo-importer` (or use the Modular toolchain that ships it)
import mojo.importer  # type: ignore

import gymnasium as gym
import numpy as np

try:
    import mojo.importer  # type: ignore
except Exception as e:
    raise ImportError(
        "Mojo importer not available. Make sure you're running within a Modular/Mojo environment "
        "or have `mojo` tooling installed."
    ) from e

# Ensure current dir is importable for the .mojo file
import sys
sys.path.insert(0, "")

# Import the Mojo module named `checkers` (from checkers.mojo)
import checkers

class CheckersMojoEnv:
    """
    A minimal, vectorless wrapper around the Mojo Checkers env.
    It exposes reset/step/render/close and the same buffer shapes as your original.
    """
    def __init__(self, size: int = 8):
        self.size = int(size)
        # Construct the Mojo env
        self._env = checkers.make_env(self.size)

        # Expose buffers as numpy arrays that view/copy from Mojo lists as needed
        self.single_observation_space = gym.spaces.Box(low=0, high=1, shape=(self.size * self.size,), dtype=np.uint8)
        self.single_action_space = gym.spaces.Discrete(self.size * self.size * 8)

    @property
    def observations(self) -> np.ndarray:
        # Convert Mojo List[UInt8] -> numpy (copy; Mojo lists aren't memoryview-compatible)
        return np.frombuffer(bytes(self._env.observations), dtype=np.uint8)

    @property
    def rewards(self) -> np.ndarray:
        return np.array([float(self._env.rewards[0])], dtype=np.float32)

    @property
    def terminals(self) -> np.ndarray:
        return np.array([int(self._env.terminals[0])], dtype=np.uint8)

    def reset(self, seed: int | None = None):
        self._env.c_reset()
        return self.observations, {}

    def step(self, action: int):
        # Write action into Mojo env
        self._env.actions[0] = int(action)
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

    for _ in range(200):
        action = np.random.randint(0, env.single_action_space.n)
        obs, rew, term, trunc, info = env.step(action)
        total += float(rew[0])
        if term[0]:
            obs, _ = env.reset()
    print("Total reward in 200 steps:", total)