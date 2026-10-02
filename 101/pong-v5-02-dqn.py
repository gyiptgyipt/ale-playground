"""
Train a DQN agent on Pong (ALE) and then watch it play.

Install first:
    pip install "gymnasium[atari,accept-rom-license]" stable-baselines3[extra] ale-py

Run:
    python train_and_watch_pong.py
"""

import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

# ---------------------------------------------------------------------------
# SETTINGS — tweak these two numbers depending on how patient you're feeling
# ---------------------------------------------------------------------------
TOTAL_TIMESTEPS = 100_000   # start small (~10-20 min on CPU) to see the loop work.
                            # Bump to 1_000_000+ once you trust the pipeline, for
                            # a genuinely good player.
WATCH_EPISODES = 3          # how many episodes to render at the end


# ---------------------------------------------------------------------------
# STEP 1: TRAIN
# ---------------------------------------------------------------------------
print("Building training environment...")
train_env = make_atari_env("ALE/Pong-v5", n_envs=1, seed=0)
train_env = VecFrameStack(train_env, n_stack=4)  # stack 4 frames so the agent sees motion

model = DQN(
    "CnnPolicy",
    train_env,
    verbose=1,
    buffer_size=100_000,      # replay memory size
    learning_starts=10_000,   # random play before learning kicks in
    exploration_fraction=0.1,
    tensorboard_log="./pong_tensorboard/",
)

print(f"Training for {TOTAL_TIMESTEPS} timesteps...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
model.save("dqn_pong")
print("Training done. Model saved to dqn_pong.zip")

train_env.close()


# ---------------------------------------------------------------------------
# STEP 2: WATCH THE TRAINED AGENT PLAY
# ---------------------------------------------------------------------------
print("Loading model and opening render window...")
watch_env = make_atari_env(
    "ALE/Pong-v5", n_envs=1, seed=0, env_kwargs={"render_mode": "human"}
)
watch_env = VecFrameStack(watch_env, n_stack=4)

model = DQN.load("dqn_pong")

obs = watch_env.reset()
episodes_done = 0
while episodes_done < WATCH_EPISODES:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = watch_env.step(action)
    if done[0]:
        episodes_done += 1
        print(f"Episode {episodes_done} finished.")

watch_env.close()
print("Done watching. Re-run this script (or raise TOTAL_TIMESTEPS) to keep improving the agent.")