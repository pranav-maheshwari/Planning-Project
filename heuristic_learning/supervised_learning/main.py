#!/usr/bin/env python
import sys
import os
import pickle
import time

sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('..'))

from environment_database import *
from graphs.HeuristicFunctions import *
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import neural_network
import numpy as np
import math

print("Packs loaded")
import operator


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

# Set the learning parameters

batch_size = 32
seed = 1234
learning_rate = 0.01
training_epochs = 1000

# Set the base heuristic value
base_heuristic = Manhattan

# Set training parameters


visualize = True
graph_connectivity = "four_connected"
num_env_to_load = 100
swamp_cost = 100
load_from_pickle = False
save_to_pickle = True
need_additional_features = False
need_normalized_features = False
preloaded = False
dijkstra = True
# Get database of environments to run experiments on
# if not load_from_pickle:
test_env_database = getEnvironmentDatabase(graph_connectivity, "soft", swamp_cost, num_env_to_load, preloaded, dijkstra,
                                           need_additional_features, need_normalized_features)

NUM_TEST = 122880


# NUM_TEST = 12000

# X, Y = getData(test_env_database, NUM_TRAIN)



# Linear regression learner
def linearRegressionLearner(X, Y, batch_size, ita, training_epochs, num_test):
    train_X = X[:-num_test]
    train_Y = Y[:-num_test]
    test_X = X[-num_test:]
    test_Y = Y[-num_test:]
    # train_X, test_X = preprocessing(train_X, test_X)
    regr = linear_model.LinearRegression()
    regr.fit(train_X, train_Y)

    print('Coefficients: \n', regr.coef_)
    print('Bias: \n', regr.intercept_)
    # The mean squared error
    print("Mean squared error: %.2f"
          % np.mean((regr.predict(test_X) - test_Y) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(test_X, test_Y))
    # Plot outputs

    """
    plt.scatter(test_X, test_Y, color='black')
    plt.plot(test_X, regr.predict(test_X), color='blue',
             linewidth=3)
    # plt.scatter(test_X, test_Y, color='black')
    # plt.plot(test_X, regr.predict(test_X), color='blue',
    #          linewidth=3)

    # plt.xticks(())
    # plt.yticks(())

    plt.show()
    """
    return regr.coef_, regr.intercept_


# Stochastic Gradient Descent Learner
def sgdLearner(X, Y, batch_size, ita, training_epochs, num_test):
    train_X = X[:-num_test]
    train_Y = Y[:-num_test]
    test_X = X[-num_test:]
    test_Y = Y[-num_test:]

    # train_X, test_X = preprocessing(train_X, test_X)
    regr = linear_model.SGDRegressor(loss='huber', alpha=0.01, verbose=2, n_iter=1000,
                                     fit_intercept=True)  # , average=True)
    regr.fit(train_X, train_Y)
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f"
          % np.mean((regr.predict(test_X) - test_Y) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(test_X, test_Y))


# Plot outputs
# plt.scatter(test_X, test_Y,  color='black')
# plt.plot(test_X, regr.predict(test_X), color='blue',
#         linewidth=3)

# plt.xticks(())
# plt.yticks(())

# plt.show()

# Non linear regression learner

def mlpRegressionLearner(X, Y, batch_size, ita, training_epochs, num_test):
    train_X = np.asarray(X[:-num_test])
    train_Y = np.asarray(Y[:-num_test])
    test_X = np.asarray(X[-num_test:])
    test_Y = np.asarray(Y[-num_test:])
    # train_X, test_X = preprocessing(train_X, test_X)

    # regr = neural_network.MLPRegressor(hidden_layer_sizes=(20), activation='relu', solver='adam', alpha='0.0001',
    #                                    learning_rate='adaptive', learning_rate_init=0.1, warm_start=True)
    regr = neural_network.MLPRegressor(hidden_layer_sizes=(30), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant',\
                                        learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=True,\
                                         warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    print train_X.shape
    print train_Y.shape
    regr.fit(train_X, train_Y)
    # print('Coefficients: \n', regr.coef_)
    print("Mean squared error: %.2f"
          % np.mean((regr.predict(test_X) - test_Y) ** 2))
    print('Variance score: %.2f' % regr.score(test_X, test_Y))

    return regr


def preprocessing(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)  # Don't cheat - fit only on training data
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)  # apply same transformation to test data
    return X_train, X_test


def getData(env_database):
    feature_array = []
    label_array = []
    for env in env_database:
        for key in env[3].iterkeys():
            feature_array.append(env[3][key])
            label_array.append(env[4][key])
    return feature_array, label_array

img = np.ones([64, 64, 3]) * 255

def display():
    cv2.namedWindow('Planning', cv2.WINDOW_NORMAL)
    while True:
        cv2.imshow('Planning', img)
        cv2.waitKey(30)

t1 = Thread(target=display)
t1.start()

def initialize_image(planning_prob):
    global img
    graph = planning_prob[0]
    start_list = planning_prob[1]
    goal_list = planning_prob[2]
    
    img = np.ones([graph.width, graph.height, 3]) * 255
    for start in start_list:
        img[start[0], start[1]] = (0, 255, 0)
    for goal in goal_list:
        img[goal[0], goal[1]] = (0, 0, 255)
    for i in graph.walls:
        img[i[0], i[1]] = (0, 0, 0)

def run_weights_in_astar(planning_problem, weights, bias, heuristic_fn, a_star = True):
    global img
    initialize_image(planning_problem)
    graph = planning_problem[0]
    start = planning_problem[1][0]
    goal = planning_problem[2][0]
    initialize_image
    feature_map = planning_problem[3]
    frontier = PriorityQueue()
    came_from = {}
    cost_so_far = {}
    frontier.put(start, 0, 0, 0)
    came_from[start] = None
    cost_so_far[start] = 0
    num_expansions = 0
    closed = set()
    while not frontier.empty():
        num_expansions += 1
        current, current_priority = frontier.get()
        if current == goal:
            break
        img[current[0]][current[1]] = (255, 0, 0)
        closed.add(current)
        time.sleep(0.01)
        neighbors = graph.neighbors(current)
        for next in neighbors:
            new_cost = cost_so_far[current] + graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                if not a_star:
                    priority = new_cost + 5*(np.dot(weights, feature_map[next]) + bias)
                    if next not in closed: 
                        frontier.put(next, priority, 5*(np.dot(weights, feature_map[next]) + bias), new_cost)
                else:
                    priority = new_cost + heuristic_fn(next, goal)
                    if next not in closed: 
                        frontier.put(next, priority, (np.dot(weights, feature_map[next]) + bias), new_cost)   
                came_from[next] = current
                cost_so_far[next] = new_cost
                
    return came_from, cost_so_far, num_expansions

def run_weights_in_astar_mlp(planning_problem, regr, Manhattan, a_star = False):
    global img
    initialize_image(planning_problem)
    graph = planning_problem[0]
    start = planning_problem[1][0]
    goal = planning_problem[2][0]
    feature_map = planning_problem[3]
    frontier = PriorityQueue()
    came_from = {}
    cost_so_far = {}
    if a_star:
        frontier.put(start, 0 + Manhattan(start, goal), Manhattan(start, goal), 0)
    else:
        frontier.put(start, 0 + regr.predict(feature_map[start]), regr.predict(feature_map[start]), 0)
    came_from[start] = None
    cost_so_far[start] = 0
    num_expansions = 0
    while not frontier.empty():
        current, current_priority = frontier.get()
        num_expansions += 1
        if current == goal:
            break
        img[current[0]][current[1]] = (255, 0, 0)
        neighbors = graph.neighbors(current)
        for next in neighbors:
            new_cost = cost_so_far[current] + graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                feature_vec = feature_map[next]
                if not a_star:
                    # print "Here"
                    priority = new_cost + regr.predict(feature_vec)
                else:
                    # print "in a star"
                    priority = new_cost + Manhattan(next, goal)
                cost_so_far[next] = new_cost
                frontier.put(next, priority, regr.predict(feature_vec), new_cost)
                came_from[next] = current
        time.sleep(0.01)
    return came_from, cost_so_far, num_expansions

           
X, Y = getData(test_env_database)

# print X, Y

# weights, bias = linearRegressionLearner(X, Y, batch_size, learning_rate, training_epochs, NUM_TEST)
# sgdLearner(X, Y, batch_size, learning_rate, training_epochs, NUM_TEST)
# sum_of_errors = 0
# for i in range(len(X) - 122880, len(X)):
#     sum_of_errors += math.pow(X[i][1] - Y[i], 2)

# print "Mean squared error: ", sum_of_errors / 122880

# _, _, num_expansions = run_weights_in_astar(test_env_database[67], weights, bias, Manhattan, True)

# print "A-Star: ", num_expansions

# print time.sleep(10)

# _, _, num_expansions = run_weights_in_astar(test_env_database[67], weights, bias, Manhattan, False)

# print "Chooo: ", num_expansions
regr = mlpRegressionLearner(X, Y, batch_size, learning_rate, training_epochs, NUM_TEST)
sum_of_errors = 0
for i in range(len(X) - 122880, len(X)):
    sum_of_errors += math.pow(X[i][1] - Y[i], 2)

print "Mean squared error Manhattan: ", sum_of_errors / 122880
# _, _, num_expansions = run_weights_in_astar(test_env_database[67], weights, bias, Manhattan, True)

# print "A-Star: ", num_expansions
# time.sleep(10)
for i in xrange(71, len(test_env_database)):
    _, _, num_expansions = run_weights_in_astar_mlp(test_env_database[i], regr, Manhattan, True)
    print "Num Expansions (A-star)", num_expansions
    _, _, num_expansions = run_weights_in_astar_mlp(test_env_database[i], regr, Manhattan, False)
    print "Num Exmapnsions (Learned Heuristic)", num_expansions