import ale_py
import gymnasium as gym

isHuman = True
epochs = 20

gym.register_envs(ale_py)
env = gym.make("ALE/Breakout-v5", render_mode="human" if isHuman else "rgb_array")

for i in range(epochs):
    done = False
    obs, info = env.reset()

    while not done:
        action = 0

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

env.close()
