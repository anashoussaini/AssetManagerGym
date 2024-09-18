# Asset Management Gym

Asset Management Gym is a custom OpenAI Gym environment designed for training reinforcement learning agents to manage portfolios and make trading decisions in the context of asset management. This environment focuses on stock price movements and uses financial metrics to help the agent make decisions about buying, selling, or holding assets over time.

## Features
- **StockEnv**: A gym-compatible environment that provides a vector of financial features about a stock each month and allows the agent to take actions (buy, sell, or hold).
- **Discounted Rewards**: Incorporates future rewards through a customizable discount factor.
- **Realistic Actions**: Continuous action space ranging from -1 (strong sell) to 1 (strong buy).
- **Historical Data**: The environment uses historical stock data to simulate market behavior.

## Installation
To install and run the environment, first clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/asset-management-gym.git
cd asset-management-gym
pip install -r requirements.txt
``` 
Usage

To use the StockEnv environment:

```bash
import gym
from asset_management_gym.stock_env import StockEnv

# Initialize the environment
env = StockEnv(json_file_path='stocks_dict.json')

# Set the horizon
env.set_horizon(12)

# Reset the environment
observation = env.reset()

# Take a random action
action = env.action_space.sample()
observation, reward, done, info = env.step(action)

```
Development

You can create and test your own agents using the provided Jupyter notebooks in the dev/ folder.

```bash
cd dev/
jupyter notebook experiments.ipynb
```
