import gym
import gym_interceptor
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import os
import numpy as np

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.Tanh())


    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

my_policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
)


def run_dlr(env_id='interceptor-v0', model_path_and_name=r'.\checkpoints\net', lr=0.0002, gamma=0.95):
    imitataion_learning = True # False #
    save_freq = 100000
    checkpoint_path = os.path.join(os.getcwd(), "checkpoints")
    checkpoint_prefix = 'cp_'
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=checkpoint_path, name_prefix=checkpoint_prefix)

    if imitataion_learning:
        gamma = 0.95001235512354532135 ## Rafi: HACK: this gamma enables imitation learning inside venv2/Lib/site-packages/stable_baselines3/common/off_policy_algorithm.py
        model = DQN("CnnPolicy", env_id, device='cuda', gamma=gamma, train_freq=1, tensorboard_log=r".\log\tensorboard\\", seed=0, policy_kwargs=my_policy_kwargs, verbose=1, create_eval_env=True, learning_rate=lr, learning_starts=0, exploration_final_eps=0.0, exploration_fraction=0.0)
        model.learn(total_timesteps=100000, eval_freq=0, tb_log_name=model_path_and_name[-23:], callback=checkpoint_callback)
    else:
        model = DQN("CnnPolicy", env_id, device='cuda', gamma=gamma, train_freq=1, tensorboard_log=r".\log\tensorboard\\", seed=0, policy_kwargs=my_policy_kwargs, verbose=1, create_eval_env=True, learning_rate=lr, learning_starts=25000, exploration_final_eps=0.1, exploration_fraction=0.2)
        model.learn(total_timesteps=5000000, eval_freq=0, tb_log_name=model_path_and_name[-23:], callback=checkpoint_callback)

    model.save(path=model_path_and_name)


def test_model(env_id, model_path_and_name, n_games, res_path):
    env = gym.make(env_id)
    model = DQN.load(model_path_and_name)
    game_scores = np.zeros(n_games, dtype = np.int)
    total_score = np.zeros(n_games, dtype = np.int)
    city_rockets = np.zeros(n_games, dtype = np.int)
    city_hits = np.zeros(n_games, dtype = np.int)
    city_interceptions = np.zeros(n_games, dtype = np.int)
    f_rockets = np.zeros(n_games, dtype=np.int)
    f_hits = np.zeros(n_games, dtype=np.int)
    f_interceptions = np.zeros(n_games, dtype=np.int)
    empty_shoots = np.zeros(n_games, dtype = np.int)
    for i in range(n_games):
        print ("game",i)
        obs = env.reset()
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)
            if dones:
                game_scores[i] = info["game score"]
                total_score[i] = info["total score"]
                city_rockets[i] = info["city rockets"]
                city_hits[i] = info["city hits"]
                city_interceptions[i] = info["city interceptions"]
                f_rockets[i] = info["field rockets"]
                f_hits[i] = info["field hits"]
                f_interceptions[i] = info["field interceptions"]
                empty_shoots[i] = info["empty shoots"]
                break
    stats  = "\n\nmodel {}. averaged over {} games\n".format(model_path_and_name, n_games)
    stats += "game score {} total score {} city rockets {} city hits {} city interceptions {} field rockets {} field hits {} field interceptions {} empty shoots {}".\
            format(game_scores.mean(), total_score.mean(),
                   city_rockets.mean(), city_hits.mean(), city_interceptions.mean(),
                   f_rockets.mean(), f_hits.mean(), f_interceptions.mean(), empty_shoots.mean())
    print (stats)
    with open(os.path.join(res_path, "res.txt"), 'a') as f:
        f.write(stats)
    return game_scores


if __name__ == '__main__':
    model_path = os.path.join(os.getcwd(), "trained_models")
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    res_path = os.path.join(os.getcwd(), "log", "res")
    if not os.path.exists(res_path):
        os.mkdir(res_path)
    for gamma in [0.95001235512354532135]: ## Rafi: HACK: this gamma enables imitation learning inside venv2/Lib/site-packages/stable_baselines3/common/off_policy_algorithm.py
        for lr in [0.002]: #[0.0001, 0.0002, 0.0004]:
            model_path_and_name = os.path.join(model_path, "il_{}_gamma_{}.zip".format(lr, gamma))
            run_dlr(env_id='interceptor-v0', model_path_and_name=model_path_and_name, lr=lr, gamma=gamma)
            print("\ntesting model {}\n".format(model_path_and_name))
            test_model(env_id='interceptor-v0',model_path_and_name=model_path_and_name, n_games=500, res_path=res_path)