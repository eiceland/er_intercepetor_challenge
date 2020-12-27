from gym.envs.registration import register

register(
    id='interceptor-v0',
    entry_point='gym_interceptor.envs:InterceptorEnv',
)