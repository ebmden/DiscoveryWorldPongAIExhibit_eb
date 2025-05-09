import numpy as np
import logging
from exhibit.game.pong import Pong
from exhibit.ai.model import PGAgent
from exhibit.game.player import BotPlayer
from exhibit.shared.config import Config
from exhibit.shared.utils import discount_rewards

#### added ####
from exhibit.shared import utils
import numpy as np
from exhibit.game.pong import Pong
from exhibit.game.player import BotPlayer
from exhibit.shared.config import Config
###############

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
        #self.env = Pong()
        self.opponent = BotPlayer(env=self.env, top=True)

    def train(self, episodes=10, batch_size=1):
        for episode in range(episodes):
            # states, actions, rewards = [], [], []
            # state = self.env.reset()
            # done = False
            # total_reward = 0
            
            states, left, right, meta = self.simulate_game(config=self.config, left=self.agent, right=self.opponent, batch=1, visualizer=None)

            # while not done:
            #     action, prob = self.agent.act(state)
            #     opponent_action = self.opponent.act()
            #     next_state, reward, done = self.env.step(action, opponent_action)

            #     states.append(state)
            #     actions.append(action)
            #     rewards.append(reward[0])  # Assuming reward[0] is for the agent

            #     state = next_state
            #     total_reward += reward[0]

            # Process rewards and train the agent
            #discounted_rewards = discount_rewards(np.array(left[2]), gamma=0.99)
            loss = self.agent.train(np.array(states), np.array(left[0]), np.array(left[1]), np.array([left[2]]))

            logging.info(f"Episode {episode + 1}/{episodes} completed. Total Reward: {total_reward}, Loss: {loss}")
            

    def simulate_game(self, config, env_type=Config.instance().CUSTOM, left=None, right=None, batch=1, visualizer=None):
        """
        Wraps both the OpenAI Gym Atari Pong environment and the custom
        Pong environment in a common interface, useful to test the same training setup
        against both environments
        """
        env = None
        state_size = None
        games_remaining = batch
        state_shape = self.config.CUSTOM_STATE_SHAPE
    
        if env_type == self.config.CUSTOM:
            env = Pong()
            state_size = self.config.CUSTOM_STATE_SIZE
            state_shape = self.config.CUSTOM_STATE_SHAPE
            if type(left) == BotPlayer: left.attach_env(env)
            if type(right) == BotPlayer: right.attach_env(env)
        elif env_type == self.config.HIT_PRACTICE:
            env = Pong(hit_practice=True)
            state_size = self.config.CUSTOM_STATE_SIZE
            state_shape = self.config.CUSTOM_STATE_SHAPE
            if type(right) == BotPlayer: right.attach_env(env)
    
        # Training data
        states = []
        states_flipped = []
        actions_l = []
        actions_r = []
        rewards_l = []
        rewards_r = []
        probs_l = []
        probs_r = []
    
        # Prepare to collect fun data for visualizations
        render_states = []
        model_states = []
        score_l = 0
        score_r = 0
        last_state = np.zeros(state_shape)
        state = env.reset()
        if visualizer is not None:
            visualizer.base_render(utils.preprocess_custom(state))
        i = 0
        while True:
            render_states.append(state.astype(np.uint8))
            current_state = utils.preprocess_custom(state)
            diff_state = current_state - last_state
            model_states.append(diff_state.astype(np.uint8))
            diff_state_rev = np.flip(diff_state, axis=1)
            last_state = current_state
            action_l, prob_l, action_r, prob_r = None, None, None, None
            x = diff_state.ravel()
            x_flip = diff_state_rev.ravel()
            if left is not None: action_l, prob_l = left.act(x_flip)
            if right is not None: action_r, prob_r = right.act(x)
            states.append(x)
    
            state, reward, done = None, None, None
            if env_type == self.config.HIT_PRACTICE:
                state, reward, done = env.step(None, self.config.ACTIONS[action_r], frames=self.config.AI_FRAME_INTERVAL)
            else:
                state, reward, done = env.step(self.config.ACTIONS[action_l], self.config.ACTIONS[action_r], frames=self.config.AI_FRAME_INTERVAL)
    
            reward_l = float(reward[0])
            reward_r = float(reward[1])
    
            # Save observations
            probs_l.append(prob_l)
            probs_r.append(prob_r)
            actions_l.append(action_l)
            actions_r.append(action_r)
            rewards_l.append(reward_l)
            rewards_r.append(reward_r)
    
            if reward_r < 0: score_l -= reward_r
            if reward_r > 0: score_r += reward_r
    
            if done:
                games_remaining -= 1
                print('Score: %f - %f.' % (score_l, score_r))
                utils.write(f'{score_l},{score_r}', f'/home/bassoe/srdes/DiscoveryWorldPongAIExhibit/scores.csv')
                if games_remaining == 0:
                    metadata = (render_states, model_states, (score_l, score_r))
                    return states, (actions_l, probs_l, rewards_l), (actions_r, probs_r, rewards_r), metadata
                else:
                    score_l, score_r = 0, 0
                    state = env.reset()
            i += 1

if __name__ == "__main__":
    trainer = ContinuousTrainer()
    logging.info("Starting continuous training...")
    trainer.train(episodes=10)
    logging.info("Continuous training completed.")
    
