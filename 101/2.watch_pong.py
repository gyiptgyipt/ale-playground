"""
Watch a Pong DQN model play. Point it at any checkpoint saved by train_pong.py,
or leave MODEL_PATH as "latest" to auto-pick the newest checkpoint in ./checkpoints/.

You can run this WHILE train_pong.py is still training in another terminal,
to check in on progress periodically -- just re-run this script to see the
latest checkpoint.

Run:
    python3 watch_pong.py
    python3 watch_pong.py ./checkpoints/dqn_pong_50000_steps.zip
    python3 watch_pong.py dqn_pong_final
"""

import sys
import glob
import os
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

CHECKPOINT_DIR = "./checkpoints/"
WATCH_EPISODES = 3


def find_latest_checkpoint():
    files = glob.glob(os.path.join(CHECKPOINT_DIR, "dqn_pong_*_steps.zip"))
    if not files:
        return None
    # sort by the step count embedded in the filename
    files.sort(key=lambda f: int(f.split("_")[-2]))
    return files[-1]


# Allow passing a specific model path as a command-line argument
if len(sys.argv) > 1:
    model_path = sys.argv[1]
else:
    model_path = find_latest_checkpoint()
    if model_path is None:
        print(f"No checkpoints found in {CHECKPOINT_DIR}. Is train_pong.py running yet?")
        print("Falling back to dqn_pong_final if it exists...")
        model_path = "dqn_pong_final"

print(f"Loading model: {model_path}")

watch_env = make_atari_env(
    "ALE/Pong-v5", n_envs=1, seed=1, env_kwargs={"render_mode": "human"}
)
watch_env = VecFrameStack(watch_env, n_stack=4)

model = DQN.load(model_path)

obs = watch_env.reset()
episodes_done = 0
while episodes_done < WATCH_EPISODES:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = watch_env.step(action)
    if done[0]:
        episodes_done += 1
        print(f"Episode {episodes_done} finished.")

watch_env.close()
