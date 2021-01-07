from typing import Any, Dict
import gym
import gym_interceptor
from stable_baselines3 import A2C, DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
# from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.logger import SummaryWriter
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

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

my_policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
)



def run_dlr(env_id='interceptor-v0', model_path_and_name = r'.\checkpoints\net', lr =0.00005, gamma = 0.95):
    save_freq = 5000
    checkpoint_path = os.path.join(os.getcwd(), "checkpoints")
    checkpoint_prefix = 'cp_'
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=checkpoint_path, name_prefix=checkpoint_prefix)

   # model = A2C("CnnPolicy", env_id, device='cuda', gamma=0.9, tensorboard_log=r".\log\tensorboard\\", seed=0, policy_kwargs=my_policy_kwargs, verbose=1, create_eval_env=True, vf_coef=0.1, learning_rate=0.00005)
    model = DQN("CnnPolicy", env_id, device='cuda', gamma=gamma, train_freq=1, tensorboard_log=r".\log\tensorboard\\", seed=0, policy_kwargs=my_policy_kwargs, verbose=1, create_eval_env=True, learning_rate=lr, learning_starts=25000, exploration_final_eps=0.1, exploration_fraction=0.1)
    score_recorder = ScoreRecorderCallback(gym.make(env_id), render_freq=1)
    #model.learn(total_timesteps=10000000, eval_freq=0, tb_log_name="first_train",  callback=score_recorder)
    model.learn(total_timesteps=1000000, eval_freq=0, tb_log_name="first_train", callback=checkpoint_callback)
    model.save(path=model_path_and_name)


class ScoreRecorderCallback(BaseCallback):
    def __init__(self, eval_env: gym.Env, render_freq: int, n_eval_episodes: int = 1, deterministic: bool = True):
        super().__init__()
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic
        self.tb = SummaryWriter()

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            game_scores = []

            def grab_score(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                game_score = self._eval_env.game_score
                game_scores.append(game_score)

            evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_score,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )
            self.logger.record(
                "trajectory/game_scores",
                self.tb.add_scalars(game_scores),
            )
        return True

def test_model(env_id, model_path_and_name, n_games, res_path):
    env = gym.make(env_id)
    model = DQN.load(model_path_and_name)
    game_scores = np.zeros(n_games, dtype = np.int)
    for i in range(n_games):
        print ("game",i)
        obs = env.reset()
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)
            if dones:
                game_scores[i] = info["game score"]
                break

    print("model {}. averaged game score {}".format(model_path_and_name, game_scores.mean()))
    with open(os.path.join(res_path, "res.txt"), 'a') as f:
        f.write("\n\nmodel {}. averaged game score {}\n".format(model_path_and_name, game_scores.mean()))
        f.write(np.array2string(game_scores))
    return game_scores

# import cProfile
if __name__ == '__main__':
    model_path = os.path.join(os.getcwd(), "trained_models")
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    res_path = os.path.join(os.getcwd(), "log", "res")
    if not os.path.exists(res_path):
        os.mkdir(res_path)
    for gamma in [0.93, 0.95, 0.97]:
        for lr in [0.00002, 0.00005, 0.0001]:
            model_path_and_name = os.path.join(model_path, "lr_{}_gamma_{}.zip".format(lr, gamma))
            run_dlr(env_id='interceptor-v0', model_path_and_name= model_path_and_name, lr= lr, gamma= gamma)
            print("\ntesting model {}\n".format(model_path_and_name))
            test_model(env_id='interceptor-v0',model_path_and_name=  model_path_and_name, n_games = 25, res_path= res_path)
    #cProfile.run(r'run_a2c()')