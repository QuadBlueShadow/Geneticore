import gymnasium as gym
from Geneticore.loading import load_model

model, te, f = load_model("model_65_episodes.pickle")

env = gym.make("CartPole-v1", 
               render_mode="human"
               )

try:
    while True:
        obs, info = env.reset()
        done = False

        while not done:
            action = model.run(obs)
            obs, reward, done, truncated, info = env.step(action)

            env.render()
except KeyboardInterrupt:
    print("Exiting")