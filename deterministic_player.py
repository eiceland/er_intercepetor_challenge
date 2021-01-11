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

def play(state, player_column, time_from_shoot):
    if len(state.shape) > 2:
        state = state[:, :, 0]
    roi = state[1:9, :]

    ## make max choose closer and center places:
    fraction_mask = np.zeros_like(roi)
    if not roi.any():
        fraction_mask[-1, 20] = 0.5
    for i in range(fraction_mask.shape[0]):
        fraction_mask[i] += 1-i*0.01
    roi += fraction_mask

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
    #plt.imshow(state[:32, :][::-1])
    plt.imshow(state[::-1])
    plt.title(actions_names[action])
    plt.pause(0.1)
    return action

def main():
    env = gym.make('interceptor-v0')
    env.reset()
    rewards = []
    state = env.render('rgb_array')
    player_column = 15
    last_shoot = -8
    for i in range(1000):
        action = play(state, player_column, i-last_shoot)
        state, reward, done, info = env.step(action)
        rewards.append(reward)
        action_name = actions_names[action]
        if action==LEFT:
            player_column -= 1
        elif action==RIGHT:
            player_column += 1
        elif action==SHOOT:
            last_shoot = i
        print("Step: " + str(i) + ", action: " + action_name + ", step reward: " + str(reward) + ", game score: " + str(info["game score"]) + ", rockets: " + str(info["rockets"]) )
        if done:
            break
    print(np.mean(rewards))

if __name__ == '__main__':
    main()
