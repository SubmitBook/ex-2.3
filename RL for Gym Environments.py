
import gymnasium as gym
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")
from stable_baselines3 import DQN

# Create the CartPole environment
env = gym.make("CartPole-v1")

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)  # Train for 10,000 timesteps

def evaluate_agent(model, env, num_episodes=5):
    for episode in range(num_episodes):
        obs, _ = env.reset()  # Unpacking the tuple (obs, info)
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs)  # Ensure obs is correctly passed
            obs, reward, done, _, _ = env.step(action)  # Unpack step result
            total_reward += reward
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")


evaluate_agent(model, env)

model = DQN("MlpPolicy", env, learning_rate=0.001, gamma=0.98, verbose=1)
model.learn(total_timesteps=20000)  # Train for a longer duration

obs, _ = env.reset()
done = False

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)  # Unpack all 5 values
    done = done or truncated  # Consider episode finished if either is True
    env.render()  # Show the simulation

