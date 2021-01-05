from typing import Any, Dict
import gym
import gym_interceptor
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
# from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.logger import SummaryWriter
import torch as th


def run_a2c(env_id='interceptor-v0'):
    model = A2C("MlpPolicy", env_id, device='cuda', gamma=0.9, tensorboard_log=r".\log\tensorboard\\", seed=0, policy_kwargs=dict(net_arch=[16]), verbose=1, create_eval_env=True)
    score_recorder = ScoreRecorderCallback(gym.make(env_id), render_freq=1)
    # model.learn(total_timesteps=10000000, eval_freq=0, tb_log_name="first_train",  callback=score_recorder)
    model.learn(total_timesteps=10000000, eval_freq=0, tb_log_name="first_train")
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
    run_a2c(env_id='interceptor-v0')
    #cProfile.run(r'run_a2c()')