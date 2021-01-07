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



def run_dlr(env_id='interceptor-v0'):
    save_freq = 500000
    checkpoint_path = r'E:\rafi\code\competition\er_challenge\checkpoints'
    checkpoint_prefix = 'cp_'
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=checkpoint_path, name_prefix=checkpoint_prefix)

   # model = A2C("CnnPolicy", env_id, device='cuda', gamma=0.9, tensorboard_log=r".\log\tensorboard\\", seed=0, policy_kwargs=my_policy_kwargs, verbose=1, create_eval_env=True, vf_coef=0.1, learning_rate=0.00005)
    model = DQN("CnnPolicy", env_id, device='cuda', gamma=0.95, train_freq=1, tensorboard_log=r".\log\tensorboard\\", seed=0, policy_kwargs=my_policy_kwargs, verbose=1, create_eval_env=True, learning_rate=0.00005, learning_starts=25000, exploration_final_eps=0.1, exploration_fraction=0.1)
    score_recorder = ScoreRecorderCallback(gym.make(env_id), render_freq=1)
    #model.learn(total_timesteps=10000000, eval_freq=0, tb_log_name="first_train",  callback=score_recorder)
    model.learn(total_timesteps=10000000, eval_freq=0, tb_log_name="first_train", callback=checkpoint_callback)
    model.save(path=r'.\checkpoints\net')


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

# import cProfile
if __name__ == '__main__':
    run_dlr(env_id='interceptor-v0')
    #cProfile.run(r'run_a2c()')