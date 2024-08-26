from gymnasium.envs.registration import register

register(
    id="Sailboat-v0",
    entry_point="gym_sailing.envs:SailboatEnv",
    max_episode_steps=3000,
)

register(
    id="SailboatDiscrete-v0",
    entry_point="gym_sailing.envs:SailboatDiscreteEnv",
    max_episode_steps=3000,
)

register(
    id="Motorboat-v0",
    entry_point="gym_sailing.envs:MotorboatEnv",
    max_episode_steps=2000,
)
