import gymnasium as gym
from Geneticore.loading import load_model

model = load_model()

env = gym.make("MountainCar-v0", 
               render_mode="human"
               )

while True:
    obs, info = env.reset()
    done = False

    while not done:
        action = model.run(obs)
        obs, reward, done, truncated, info = env.step(action)

        env.render()