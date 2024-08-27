# gym-sailing: A sailing environment for OpenAI Gym / Gymnasium

---

This is a Gymnasium (OpenAI Gym) environment designed to train reinforcement learning (RL) agents to control a sailboat. The environment simulates the dynamics of a sailboat and allows the agent to learn tacking behavior to reach a target point.

![sailboat gif](https://github.com/Gabo-Tor/gym-sailing/raw/main/img/env.gif?raw=True "sailboat")

## Environments

| Environment | Description |
| --- | --- |
| **Sailboat-v0** | The main environment with a continuous action space. |
| **SailboatDiscrete-v0** | A variation of the environment with a discrete action space. |
| **Motorboat-v0** | An easy test environment with a motorboat instead of a sailboat. |

## Installation

You can install the latest release using pip:

```bash
pip install gym-sailing
```

Alternatively, if you prefer, you can clone the repository and install it locally.

## Usage

### Basic Usage

Bare minimum code to run the environment:

```python
import gymnasium as gym
import gym_sailing

env = gym.make("Sailboat-v0", render_mode="human")
observation, info = env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()  # this is where you would insert your policy
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

### Training an RL Agent

To train an RL agent using stable-baselines3:

```python
from stable_baselines3 import PPO
import gymnasium as gym
import gym_sailing

env = gym.make("Sailboat-v0")
model = PPO('MlpPolicy', env, verbose=1)

# Train the agent
model.learn(total_timesteps=1_000_000)

# Test the trained model
observation, info = env.reset()
for _ in range(1000):
    action, _ = model.predict(observation)
    observation, reward, terminated, truncated, info = env.step(action)

env.close()
```

## Environment Details

### Observation Space

The observation space includes:

- **Boat Speed:** The current speed of the boat.
- **Boat Heading:** The angle of the boat relative to the wind, ranging from -$\pi$ to $\pi$.
- **Heading Rate:** The rate of change of the boat's heading.
- **Course to Target:** The angle between the boat's heading and the target, ranging from -$\pi$ to $\pi$.
- **Distance to Target:** The normalized distance between the boat and the target.

### Action Space

The action space consists of:

- **Rudder Angle:** The angle of the rudder, ranging from -1 to 1 for *Sailboat-v0* and *Motorboat-v0*, and {-1, 0, 1} for *SailboatDiscrete-v0*.

### Reward

The default reward function includes:

- **Alive Penalty:** A penalty for each time step to encourage the agent to reach the target quickly.
- **Target Reward:** A reward for reaching the target.
- **Course Penalty:** A penalty for leaving the course area.
- **Progress Reward:** A reward for making progress towards the target, using the L8 norm, to encourage the agent to move upwind.

### Episode End

- The environment is **terminated** if the boat reaches the target or leaves the course area.
- The environment is **truncated** after 3000 steps.

## Benchmarks

Benchmarks using stable-baselines3 with default hyperparameters. Good policies that tack only once tend to achieve ~390 total reward for the sailboat environment. PPO seems to perform better, but SAC is also a good option, that even converging faster.

![benchmarks](https://github.com/Gabo-Tor/gym-sailing/raw/main/img/benchmarks.png?raw=True "benchmarks")

## Contributing

Contributions are welcome. Please fork the repository and submit a pull request with your changes. For any questions or suggestions, feel free to open an issue.

## Future Work

Here are some features I'd like to add in the future:

- Add currents of different intensities and directions.
- Add wind shifts.
- Add wind gusts and lulls.
- Make the polar diagram more accurate, using the data from this paper: *R. Binns, F. W. Bethwaite, and N. R. Saunders, “Development of A More Realistic Sailing Simulator,” High Performance Yacht Design Conference. RINA, pp. 243–250, Dec. 04, 2002. doi: 10.3940/rina.ya.2002.29.*

## Inspiration

This project was inspired by this fork: https://github.com/openai/gym/compare/master...JonAsbury:gym:Sailing-Simulator-Env

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/Gabo-Tor/gym-sailing/raw/main/LICENSE) file for details.
