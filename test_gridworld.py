# Run `pip install "gymnasium[classic-control]"` for this example.
import gymnasium as gym
from gymnasium.envs.registration import register

register(
    id="GridWorld-v0",
    entry_point="gridworld:GridWorld",
)

# Create our training environment.
env = gym.make(
    "GridWorld-v0",
    N=10,
    M=20,
    reward_type="dense",
    render_mode="human",
    max_steps=1000,
    visits=True,
)

# Reset environment to start a new episode
observation, info = env.reset()
# observation: what the agent can "see" - cart position, velocity, pole angle, etc.
# info: extra debugging information (usually not needed for basic learning)

print(f"Starting observation: {observation}")
# Example output: [ 0.01234567 -0.00987654  0.02345678  0.01456789]
# [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

episode_over = False
total_reward = 0

while not episode_over:
    # Choose an action: 0 = push cart left, 1 = push cart right
    action = env.action_space.sample()  # Random action for now - real agents will be smarter!

    # Take the action and see what happens
    observation, reward, terminated, truncated, info = env.step(action)

    # reward: +1 for each step the pole stays upright
    # terminated: True if pole falls too far (agent failed)
    # truncated: True if we hit the time limit (500 steps)

    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
env.close()
