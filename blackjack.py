from collections import defaultdict
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Progress bar
from time import sleep
import seaborn as sns
import pandas as pd


class BlackJackAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """
        Initialize a Q-Learning agent.

        Args:
            env: The training environment
            learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate (usually 0.1)
            discount_factor: How much to value future rewards (0-1)
        """

        self.env = env

        # Q-table: maps (state, action) to expected reward
        # defaultdict automatically creates entries with zeros for new states
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor  # How much we care about future rewards

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """Choose an action using epsilon-greedy strategy.

        Returns:
            action: 0 (stand) or 1 (hit)
        """
        # With probability epsilon: explore (random action)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # With probability (1-epsilon): exploit (best known action)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """Update Q-value based on experience.

        This is the heart of Q-learning: learn from (state, action, reward, next_state)
        """
        # What's the best we could do from the next state?
        # (Zero if episode terminated - no future rewards possible)
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])

        # What should the Q-value be? (Bellman equation)
        target = reward + self.discount_factor * future_q_value

        # How wrong was our current estimate?
        temporal_difference = target - self.q_values[obs][action]

        # Update our estimate in the direction of the error
        # Learning rate controls how big steps we take
        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )

        # Track learning progress (useful for debugging)
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

def train_agent(env, agent, n_episodes):
    for episode in tqdm(range(n_episodes)):

        #print(f"\nEpisode {episode}")
        # Start a new hand
        obs, info = env.reset()
        done = False

        # Play one complete hand
        while not done:
            # Agent chooses action (initially random, gradually more intelligent)
            action = agent.get_action(obs)

            # Take action and observe result
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Learn from this experience
            agent.update(obs, action, reward, terminated, next_obs)

            # Move to next state
            done = terminated or truncated
            obs = next_obs

        # Reduce exploration rate (agent becomes less random over time)
        agent.decay_epsilon()

def get_moving_avgs(arr, window, convolution_mode):
    """Compute moving average to smooth noisy data."""
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window

def visualize(env, agent):

    # Smooth over a 500-episode window
    rolling_length = 500
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

    # Episode rewards (win/loss performance)
    axs[0].set_title("Episode rewards")
    reward_moving_average = get_moving_avgs(
        env.return_queue,
        rolling_length,
        "valid"
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[0].set_ylabel("Average Reward")
    axs[0].set_xlabel("Episode")

    # Episode lengths (how many actions per hand)
    axs[1].set_title("Episode lengths")
    length_moving_average = get_moving_avgs(
        env.length_queue,
        rolling_length,
        "valid"
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)
    axs[1].set_ylabel("Average Episode Length")
    axs[1].set_xlabel("Episode")

    # Training error (how much we're still learning)
    axs[2].set_title("Training Error")
    training_error_moving_average = get_moving_avgs(
        agent.training_error,
        rolling_length,
        "same"
    )
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    axs[2].set_ylabel("Temporal Difference Error")
    axs[2].set_xlabel("Step")

    plt.tight_layout()
    plt.show()

    # Plot Q-table
    # print(agent.q_values)
    # q_table = pd.DataFrame(agent.q_values)
    # print(q_table)
    # sns.heatmap(q_table, annot=True,  fmt=".1f", linewidths=1)
    # plt.show()

# Test the trained agent
def test_agent(agent, env, num_episodes=10):
    """Test agent performance without learning or exploration."""
    total_rewards = []

    # Temporarily disable exploration for testing
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Pure exploitation
    q_table = defaultdict(lambda: np.zeros(env.action_space.n))
    for _ in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.get_action(obs)
            q_table[obs] = agent.q_values[obs]
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)

    # Restore original epsilon
    agent.epsilon = old_epsilon

    win_rate = np.mean(np.array(total_rewards) > 0)
    average_reward = np.mean(total_rewards)

    print(f"Test Results over {num_episodes} episodes:")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Average Reward: {average_reward:.3f}")
    print(f"Standard Deviation: {np.std(total_rewards):.3f}")

    # Plot Q-table
    q_table = pd.DataFrame(q_table)
    print(q_table)
    sns.heatmap(q_table, annot=True,  fmt=".1f", linewidths=0.5)
    plt.show()

def main():

    # Training hyperparameters
    """
    Train a Q-learning agent to play Blackjack using the gymnasium environment.

    The agent learns to take actions in the environment to maximize rewards.
    The agent uses an epsilon-greedy strategy to balance exploration and exploitation.
    After training, the agent is tested on 10 episodes to evaluate its win rate and average reward.

    Parameters:
        learning_rate (float): How fast to learn (higher = faster but less stable)
        n_episodes (int): Number of hands to practice
        start_epsilon (float): Start with 100% random actions
        epsilon_decay (float): Reduce exploration over time
        final_epsilon (float): Always keep some exploration

    Returns:
        None
    """
    learning_rate = 0.01       # How fast to learn (higher = faster but less stable)
    n_episodes = 100_000       # Number of hands to practice
    start_epsilon = 1.0         # Start with 100% random actions
    epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
    final_epsilon = 0.1         # Always keep some exploration

    env = gym.make("Blackjack-v1", sab=False) #render_mode="human")
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
    
    agent = BlackJackAgent(
        env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )

    # train_agent(env, agent, n_episodes)
    # print("Size of Q-table:", len(agent.q_values))
    # pd.DataFrame(agent.q_values.items()).to_csv("blackjack_q_table.csv")
    # visualize(env, agent)
    q_table = pd.read_csv("blackjack_q_table.csv").values[:,1:]
    agent.q_values = {eval(i) :np.fromstring(v.strip('[]'), sep=' ') for i,v in q_table}
    print(agent.q_values[(17, 7, 0)])
    test_agent(agent, env)


    
    # for episode in range(100):
    #     print(f"Episode {episode}")
    #     obs, info = env.reset()
    #     terminated = truncated = False
    #     print(f"Starting observation: {obs}")
    #     ep_reward = 0
    #     #sleep(5)
    #     while not (terminated or truncated):
    #         action = agent.get_action(obs)
    #         print(f"Action: {action}")
    #         next_obs, reward, terminated, truncated, info = env.step(action)
    #         print(f"Next observation: {next_obs}")
    #         print(f"Reward: {reward}")
    #         print(f"Terminated: {terminated}")
    #         print(f"Truncated: {truncated}")
    #         agent.update(obs, action, reward, terminated, next_obs)
    #         obs = next_obs
    #         print(f"Q-values: {agent.q_values[obs]}")
    #         print(f"Epsilon: {agent.epsilon}") 
    #         print("\n")
    #         ep_reward += reward
    #         q_values.append(agent.q_values[obs])
    #         #sleep(3)
    #     agent.decay_epsilon()
    #     rewards.append(ep_reward)
    # env.close()

    # plt.plot(rewards)
    # plt.show()
    # plt.plot(q_values)
    # plt.show()
    

if __name__ == "__main__":
    main()
