import gym
import gym_interceptor
from stable_baselines3 import DQN
import os
import numpy as np
import time
import sys

def test_model(env_id, model_path_and_name, n_games = 500):
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
    #print (stats)
    return game_scores


if __name__ == '__main__':
    if len(sys.argv) >1:
        n_games = int(sys.argv[1])
    else:
        n_games = 500

    t_start = time.time()
    model_path_and_name = os.path.join(os.getcwd(), "trained_models", "rl_wide_0.0003_gamma_0.95.zip")
    game_scores = test_model(env_id='interceptor-v0', model_path_and_name=model_path_and_name, n_games=n_games)
    t_end = time.time()
    print ("{} games. Averaged_score {},   std {},    time {} minutes".format(n_games, game_scores.mean(), game_scores.std(), (t_end - t_start)/60))