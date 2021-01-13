import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import gym
import gym_interceptor


##Enums:
LEFT = 0
STRAIGHT = 1
RIGHT = 2
SHOOT = 3
QUIT = 4

actions = {52: LEFT, 56: STRAIGHT, 53: STRAIGHT, 54: RIGHT, 48: SHOOT, 113: QUIT}
actions_names = {LEFT: 'LEFT', STRAIGHT: 'STRAIGHT', RIGHT: 'RIGHT', SHOOT: 'SHOOT'}


class GreedyPlayer:
    def __init__(self):
        self.last_shoot = -8
        self.step = 0

    def act(self, state):
        player_column = state[0, 0, 0].argmax()
        action = act(state, player_column, self.step - self.last_shoot)
        if action == SHOOT:
            self.last_shoot = self.step
        self.step += 1
        return action

    def reset(self):
        self.player_column = 15
        self.last_shoot = -8
        self.step = 0


def act(state, player_column, time_from_shoot):
    disp = False # True #
    if len(state.shape) == 3:
        state = state[:, :, 0]
    elif len(state.shape) == 4:
        state = state[0, 0, :, :]

    roi = state[1:9, :]

    ## make max choose closer and center places:
    fraction_mask = np.zeros_like(roi).astype(float)
    fraction_mask[-1, 20] = 0.1
    for i in range(fraction_mask.shape[0]):
        fraction_mask[i] = 0.1 - i*0.01
    roi = roi + fraction_mask

    ## remove unreachable areas:
    # for i in range(fraction_mask.shape[0]):
    #     roi[i, :player_column-i] = 0
    #     roi[i, player_column+i+1:] = 0

    if time_from_shoot <= 8:
        roi[:8-time_from_shoot, :] = 0
    mx_x, mx_y = np.unravel_index(roi.argmax(), roi.shape)
    if mx_y < player_column:
        action = LEFT
    elif mx_y > player_column:
        action = RIGHT
    else:
        if mx_x == 0:
            action = SHOOT
        else:
            action = STRAIGHT
    if disp:
        #plt.imshow(state[:32, :][::-1])
        plt.imshow(state[::-1])
        plt.title(actions_names[action])
        plt.pause(0.001)
    return action


def play():
    env = gym.make('interceptor-v0')
    env.reset()
    rewards = []
    state = env.render('rgb_array')
    player_column = 15
    last_shoot = -8
    for i in range(1000):
        action = act(state, player_column, i-last_shoot)
        state, reward, done, info = env.step(action)
        rewards.append(reward)
        action_name = actions_names[action]
        if action==LEFT:
            player_column -= 1
        elif action==RIGHT:
            player_column += 1
        elif action==SHOOT:
            last_shoot = i
        #print("Step: " + str(i) + ", action: " + action_name + ", step reward: " + str(reward) + ", game score: " + str(info["game score"]) + ", rockets: " + str(info["rockets"]) )
        if done:
            break
    print(np.mean(rewards))
    return info["game score"]

if __name__ == '__main__':
    scores = []
    for i in range(250):
        print ("game", i)
        score = play()
        scores.append(score)
    print(str(np.mean(scores)) +" std: " + str(np.std(scores)) )
    a=1
