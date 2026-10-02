"""
Train a DQN agent on Pong (ALE). Saves checkpoints periodically so you can
watch progress with watch_pong.py WHILE this keeps training in another terminal.

Install first:
    pip install "gymnasium[atari,accept-rom-license]" stable-baselines3[extra] ale-py

Run:
    python3 1.train_pong.py
    tensorboard --logdir ./pong_tensorboard
"""

import os

import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)

gym.register_envs(ale_py)

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------
TOTAL_TIMESTEPS = 100_000   # start small to see the loop work; raise to 1_000_000+ later
CHECKPOINT_EVERY = 10_000   # save a checkpoint this often so you can watch progress
CHECKPOINT_DIR = "./checkpoints/"
BEST_MODEL_DIR = "./best_model/"


class RenderTrainingCallback(BaseCallback):
    def _on_step(self) -> bool:
        self.training_env.render()
        return True


# ---------------------------------------------------------------------------
# TRAIN
# ---------------------------------------------------------------------------
print("Building training environment...")
train_env = make_atari_env(
    "ALE/Pong-v5", n_envs=1, seed=0, env_kwargs={"render_mode": "human"}
)
train_env = VecFrameStack(train_env, n_stack=4)  # stack 4 frames so the agent sees motion
eval_env = make_atari_env("ALE/Pong-v5", n_envs=1, seed=1)
eval_env = VecFrameStack(eval_env, n_stack=4)

model = DQN(
    "CnnPolicy",
    train_env,
    verbose=1,
    buffer_size=100_000,
    learning_starts=10_000,
    exploration_fraction=0.1,
    tensorboard_log="./pong_tensorboard/",
)

checkpoint_callback = CheckpointCallback(
    save_freq=CHECKPOINT_EVERY,
    save_path=CHECKPOINT_DIR,
    name_prefix="dqn_pong",
)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=BEST_MODEL_DIR,
    log_path="./eval_logs/",
    eval_freq=CHECKPOINT_EVERY,
    n_eval_episodes=5,
    deterministic=True,
)

print(f"Training for {TOTAL_TIMESTEPS} timesteps. Checkpoints every {CHECKPOINT_EVERY} steps in {CHECKPOINT_DIR}")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    progress_bar=True,
    callback=CallbackList(
        [checkpoint_callback, eval_callback, RenderTrainingCallback()]
    ),
)

model.save("dqn_pong_final")
print("Training done. Final model saved to dqn_pong_final.zip")

best_model_path = os.path.join(BEST_MODEL_DIR, "best_model.zip")
if os.path.exists(best_model_path):
    model = DQN.load(best_model_path, env=train_env)
    print(f"Loaded best evaluated model from {best_model_path}")
else:
    print("No evaluation checkpoint was saved; playing with the final model.")

eval_env.close()
print("Playing continuously. Press Ctrl+C in this terminal to stop.")
obs = train_env.reset()
try:
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = train_env.step(action)
        train_env.render()
except KeyboardInterrupt:
    print("\nPlayback stopped.")
finally:
    train_env.close()
