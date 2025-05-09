import numpy as np
import logging
# import sys
# import os

# Add the root directory of the project to the Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from exhibit.game.pong import Pong
from exhibit.game.player import PGAgent, BotPlayer
from exhibit.shared.config import Config
from exhibit.shared.utils import discount_rewards

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("continuous_training.log"),
        logging.StreamHandler()
    ]
)

class ContinuousTrainer:
    def __init__(self, config=None, learning_rate=0.001, dense_structure=(200,)):
        self.config = config or Config.instance()
        self.agent = PGAgent(
            state_size=self.config.CUSTOM_STATE_SIZE,
            action_size=self.config.CUSTOM_ACTION_SIZE,
            learning_rate=learning_rate,
            structure=dense_structure
        )
        self.env = Pong(config=self.config)
        self.opponent = BotPlayer(env=self.env, top=True)

    def train(self, episodes=10, batch_size=5):
        for episode in range(episodes):
            states, actions, rewards = [], [], []
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action, prob = self.agent.act(state)
                opponent_action = self.opponent.act()
                next_state, reward, done = self.env.step(action, opponent_action)

                states.append(state)
                actions.append(action)
                rewards.append(reward[0])  # Assuming reward[0] is for the agent

                state = next_state
                total_reward += reward[0]

            # Process rewards and train the agent
            discounted_rewards = discount_rewards(rewards, gamma=0.99)
            loss = self.agent.train(np.array(states), np.array(actions), np.array([discounted_rewards]))

            logging.info(f"Episode {episode + 1}/{episodes} completed. Total Reward: {total_reward}, Loss: {loss}")

if __name__ == "__main__":
    trainer = ContinuousTrainer()
    logging.info("Starting continuous training...")
    trainer.train(episodes=1000)
    logging.info("Continuous training completed.")