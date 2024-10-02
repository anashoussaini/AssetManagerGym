# AssetManagerGym: A Custom Gym Environment for Stock Trading with RL

Welcome to **AssetManagerGym**, a custom OpenAI Gym environment designed for training reinforcement learning agents on stock trading tasks. This environment simulates stock market dynamics, allowing agents to learn trading strategies by interacting with historical stock data.

## Table of Contents

- [Introduction](#introduction)
- [Environment Description](#environment-description)
  - [Features](#features)
  - [Action Space](#action-space)
  - [Observation Space](#observation-space)
  - [Reward Function](#reward-function)
  - [Episode Termination](#episode-termination)
- [Installation](#installation)
- [Usage](#usage)
  - [Environment Initialization](#environment-initialization)
  - [Running an Episode](#running-an-episode)
- [Parameters](#parameters)
- [Agent Integration](#agent-integration)
- [Example](#example)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction

**AssetManagerGym** is a simulation environment that provides a platform for developing and testing reinforcement learning (RL) agents on stock trading tasks. It supports features like:

- Position management (long, short, flat)
- Stop loss and take profit mechanisms
- Customizable risk parameters
- Randomized stock selection for each episode
- Support for training and testing data splits

The environment is compatible with standard RL libraries and can be integrated with various deep RL algorithms.

---

## Environment Description

### Features

- **Random Stock Selection**: Each episode starts with a randomly selected stock from a provided dataset.
- **Position Management**: The agent can open long or short positions or choose to remain flat.
- **Risk Management**: Supports stop loss and take profit levels to simulate realistic trading conditions.
- **Customizable Horizon**: Set the maximum number of steps per episode.
- **Data Splitting**: Ability to split data into training and testing sets based on a specified year.

### Action Space

The action space is **continuous**, representing the agent's trading decisions:

- **Range**: `[-1.0, 1.0]`
  - **-1.0**: Strong sell (open a short position)
  - **0.0**: Hold (maintain current position)
  - **1.0**: Strong buy (open a long position)
- **Shape**: `(1,)`

### Observation Space

The observation space consists of a feature vector extracted from stock data:

- **Type**: `Box`
- **Shape**: `(n_features,)` where `n_features` is determined dynamically based on the dataset.
- **Data**: Includes features like price (`prc`) and other indicators relevant to trading.

### Reward Function

The reward is calculated based on the agent's actions and market movements:

- **Opening a Position**: No immediate reward upon opening a position.
- **Holding a Position**:
  - **Unrealized Profit/Loss**: Calculated as the percentage change in price since the position was opened, adjusted for position type (long or short).
- **Closing a Position**:
  - **Take Profit Hit**: Agent receives a reward of `+1` and the episode may terminate.
  - **Stop Loss Hit**: Agent receives a reward of `-1` (penalty) and the episode may terminate.
- **Choosing Not to Trade**:
  - Agent can choose to do nothing, and the episode continues unless specified otherwise.

### Episode Termination

An episode may terminate when:

- **Stop Loss or Take Profit is Hit**: Position is closed due to risk parameters being triggered.
- **Horizon is Reached**: The maximum number of steps per episode is reached.
- **No More Data**: No further data is available for the selected stock.
- **Custom Conditions**: Agent-specific logic can dictate episode termination.

---

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/anashoussaini/assetmanagergym.git
   cd assetmanagergym
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   *Ensure you have the following installed:*

   - `gym`
   - `numpy`
   - `matplotlib`
   - `torch` (for integrating with PyTorch agents)
   - `json` (standard library)
   - `os` (standard library)

3. **Prepare the Dataset**

   - Place your stock data in a JSON file named `feature_dict.json` in the working directory.
   - The data should be structured as:

     ```json
     {
       "AAPL": {
         "2008": {
           "1": {"prc": 150.0, "feature1": value, ...},
           "2": {"prc": 155.0, "feature1": value, ...},
           ...
         },
         "2009": { ... }
       },
       "GOOG": { ... },
       ...
     }
     ```

---

## Usage

### Environment Initialization

```python
from assetmanagergym import assetmanagergym

json_file = 'feature_dict.json'
risk_params = {'stop_loss': 0.05, 'take_profit': 0.1}  # 5% stop loss, 10% take profit
env = assetmanagergym(json_file_path=json_file, risk_params=risk_params, mode='train', train_test_split_year=2010)
env.set_horizon(12)  # Set the maximum number of steps per episode
```

### Running an Episode

```python
observation = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # Replace with agent's action
    observation, reward, done, info = env.step(action)
    env.render()  # Optional: Visualize the current state
```

---

## Parameters

| Parameter               | Type     | Description                                                                                  |
|-------------------------|----------|----------------------------------------------------------------------------------------------|
| `json_file_path`        | `str`    | Path to the JSON file containing stock data.                                                 |
| `risk_params`           | `dict`   | Dictionary with risk parameters: `stop_loss` and `take_profit`.                              |
| `mode`                  | `str`    | `'train'` or `'test'`, determines which data split to use.                                    |
| `train_test_split_year` | `int`    | The year used to split the data into training and testing sets.                              |
| `discount_factor`       | `float`  | Discount factor for calculating discounted rewards.                                          |
| `horizon`               | `int`    | Maximum number of steps per episode. Set using `env.set_horizon(horizon)`.                   |

---

## Agent Integration

**assetmanagergym** is designed to be compatible with deep reinforcement learning agents. Below is an example of how to integrate a custom agent.

### Example Agent Loop

```python
agent = YourCustomAgent(env.observation_space, env.action_space)

for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        agent.learn()  # Update agent's networks
        state = next_state

    agent.reset()  # Reset agent's state if necessary
```

---

## Example

Here's a full example combining environment setup, agent interaction, and visualization.

```python
import gym
import numpy as np
from assetmanagergym import assetmanagergym

def main():
    json_file = 'feature_dict.json'
    risk_params = {'stop_loss': 0.05, 'take_profit': 0.1}
    env = assetmanagergym(json_file_path=json_file, risk_params=risk_params, mode='train', train_test_split_year=2010)
    env.set_horizon(12)

    observation = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()  # Replace with agent's action
        observation, reward, done, info = env.step(action)
        env.render()

    env.close()

if __name__ == "__main__":
    main()
```

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch:

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. Commit your changes:

   ```bash
   git commit -am 'Add new feature'
   ```

4. Push to the branch:

   ```bash
   git push origin feature/your-feature-name
   ```

5. Open a pull request.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This README provides an overview of the **assetmanagergym** environment, including its setup and usage. For detailed documentation and advanced configurations, please refer to the code comments and additional documentation files in the repository.

# Appendix

## Key Classes and Methods

### `assetmanagergym` Class

#### Initialization

```python
env = assetmanagergym(
    json_file_path='feature_dict.json',
    risk_params={'stop_loss': 0.05, 'take_profit': 0.1},
    mode='train',
    train_test_split_year=2010,
    discount_factor=0.1
)
```

#### Methods

- `env.reset()`: Resets the environment to a new initial state.
- `env.step(action)`: Advances the environment by one step based on the action.
- `env.render()`: Renders the current state of the environment.
- `env.close()`: Performs any necessary cleanup.
- `env.set_horizon(horizon)`: Sets the maximum number of steps per episode.

### Action Interpretation

| Action Value    | Interpretation            |
|-----------------|---------------------------|
| `> 0.5`         | Open a long position      |
| `< -0.5`        | Open a short position     |
| `-0.5` to `0.5` | Hold / Do nothing         |

### Reward Calculation

- **Unrealized P&L**:

  \[
  \text{price\_change} = \frac{\text{next\_price} - \text{entry\_price}}{\text{entry\_price}}
  \]

  \[
  \text{unrealized\_pnl} = \text{price\_change} \times \text{position}
  \]

- **Stop Loss Triggered**:

  - If `unrealized_pnl <= -stop_loss`, reward is `-1`.

- **Take Profit Triggered**:

  - If `unrealized_pnl >= take_profit`, reward is `+1`.

---

## Visualization

The environment includes methods for visualizing rewards and actions.

### Reward Plot with Action Arrows

```python
env.render_with_arrows(rewards, discounted_rewards, actions, steps)
```

- **Arrows**:
  - **↑**: Buy action
  - **↓**: Sell action
  - **–**: Hold action

---

## Data Structure

### JSON Data Format

The environment expects the stock data in a specific JSON format.

#### Example Structure

```json
{
  "AAPL": {
    "2008": {
      "1": {
        "prc": 150.0,
        "feature1": value,
        "feature2": value,
        ...
      },
      "2": { ... },
      ...
    },
    "2009": { ... },
    ...
  },
  "GOOG": { ... },
  ...
}
```

- **Stock Ticker**: Top-level keys (e.g., `"AAPL"`, `"GOOG"`).
- **Year**: Second-level keys (e.g., `"2008"`, `"2009"`).
- **Month**: Third-level keys (e.g., `"1"` for January).
- **Features**: Dictionary containing `"prc"` and other features.

---

## Tables of Parameters

### Default Risk Parameters

| Parameter     | Default Value | Description               |
|---------------|---------------|---------------------------|
| `stop_loss`   | `0.05`        | Stop loss threshold (5%)  |
| `take_profit` | `0.10`        | Take profit threshold (10%)|

### Environment Parameters

| Parameter           | Type     | Default     | Description                            |
|---------------------|----------|-------------|----------------------------------------|
| `discount_factor`   | `float`  | `0.1`       | Discount factor for rewards            |
| `horizon`           | `int`    | `12`        | Max steps per episode                  |
| `mode`              | `str`    | `'train'`   | Mode: `'train'` or `'test'`            |
| `train_test_split_year` | `int`| `2010`      | Year to split training and testing data|

---

## Contact

For any questions or support, please open an issue on the repository or contact the maintainer at [email@example.com](mailto:email@example.com).

---

*Happy trading and reinforcement learning!*
