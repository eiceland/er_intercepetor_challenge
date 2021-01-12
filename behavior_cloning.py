import numpy as np
import gym_interceptor
import os
# from stable_baselines import DQN
from stable_baselines.gail import generate_expert_traj#, ExpertDataset
# from stable_baselines.common.callbacks import CheckpointCallback
from deterministic_player import GreedyPlayer

def collect():
    greedy_player = GreedyPlayer()
    generate_expert_traj(greedy_player, 'greedy', n_timesteps=int(1e5), n_episodes=10)
#
#
# def train():
#     env_id = 'interceptor-v0'
#     save_freq = 100000
#     gamma = 0.95
#     lr = 0.0002
#     checkpoint_path = os.path.join(os.getcwd(), "checkpoints")
#     checkpoint_prefix = 'cp_'
#     checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=checkpoint_path, name_prefix=checkpoint_prefix)
#
#     model = DQN("CnnPolicy", env_id, device='cuda', gamma=gamma, train_freq=1, tensorboard_log=r".\log\tensorboard\\",
#                 seed=0, policy_kwargs=my_policy_kwargs, verbose=1, create_eval_env=True, learning_rate=lr,
#                 learning_starts=25000, exploration_final_eps=0.1, exploration_fraction=0.2)
#
#     dataset = ExpertDataset(expert_path='greedy.npz', traj_limitation=1, batch_size=128)
#     model.pretrain(dataset, n_epochs=1000)
#
#     log_dir = r".\log\tensorboard"
#     model.learn(total_timesteps=5000000, eval_freq=0, tb_log_name=log_dir, callback=checkpoint_callback)
#     model_path_and_name = r".\checkpoints\cp_final.zip"
#     model.save(path=model_path_and_name)
#
#
#
#
# class CustomCNN(BaseFeaturesExtractor):
#     """
#     :param observation_space: (gym.Space)
#     :param features_dim: (int) Number of features extracted.
#         This corresponds to the number of unit for the last layer.
#     """
#
#     def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
#         super(CustomCNN, self).__init__(observation_space, features_dim)
#         # We assume CxHxW images (channels first)
#         # Re-ordering will be done by pre-preprocessing or wrapper
#         n_input_channels = observation_space.shape[0]
#         self.cnn = nn.Sequential(
#             nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
#             nn.ReLU(),
#             nn.Flatten(),
#         )
#
#         # Compute shape by doing one forward pass
#         with th.no_grad():
#             n_flatten = self.cnn(
#                 th.as_tensor(observation_space.sample()[None]).float()
#             ).shape[1]
#
#         self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
#
#     def forward(self, observations: th.Tensor) -> th.Tensor:
#         return self.linear(self.cnn(observations))
#
# my_policy_kwargs = dict(
#     features_extractor_class=CustomCNN,
#     features_extractor_kwargs=dict(features_dim=128),
# )

if __name__ == '__main__':
    collect()
    # train()