import cv2 as cv
import numpy as np
import gym
import gym_interceptor


##Enums:
LEFT = 0
STRAIGHT = 1
RIGHT = 2
SHOOT = 3
QUIT = 4

actions = {52: LEFT, 56: STRAIGHT, 53: STRAIGHT, 54: RIGHT, 48: SHOOT, 113: QUIT}
actions_names = {52: 'LEFT', 56: 'STRAIGHT', 53: 'STRAIGHT', 54: 'RIGHT', 48: 'SHOOT', 113: "QUIT"}


def main():
    env = gym.make('interceptor-v0')
    env.reset()
    rewards = []
    for i in range(1000):
        im = env.render('human')
        im = cv.resize(im.astype("uint8"), (int(im.shape[0] / 5), int(im.shape[1] / 5)))
        cv.imshow("4-left, 5-straight, 6-right, 0-shoot", im)
        key = cv.waitKey()
        action = STRAIGHT
        if key in actions.keys():
            action = actions[key]
            action_name = actions_names[key]
            if action == QUIT:
                print("GOODBYE!")
                break
        state, reward, done, info = env.step(action)
        rewards.append(reward)
        print("Step: " + str(i) + ", action: " + action_name + ", step reward: " + str(reward) + ", game score: " + str(info["game score"]) + ", rockets: " + str(info["rockets"]) )
        if done:
            break
    print(np.mean(rewards))

if __name__ == '__main__':
    main()
