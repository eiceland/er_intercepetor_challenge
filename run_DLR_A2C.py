from stable_baselines3 import A2C
import gym_interceptor


def run_a2c(env_id='interceptor-v0'):
    model = A2C("MlpPolicy", env_id, device='cuda', seed=0, policy_kwargs=dict(net_arch=[16]), verbose=1, create_eval_env=True)
    # model.learn(total_timesteps=100, eval_freq=10)
    model.learn(total_timesteps=10, eval_freq=0)
    # model.save(path=r'E:\rafi\code\competition\er_challenge\checkpoints\net')


import cProfile
if __name__ == '__main__':
    # run_a2c(env_id='interceptor-v0')
    cProfile.run(r'run_a2c()')