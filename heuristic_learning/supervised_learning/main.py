#!/usr/bin/env python
import sys
import os
import pickle

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
<<<<<<< HEAD
    # print('Coefficients: \n', regr.coef_)
=======
    print('Coefficients: \n', regr.coef_)
    print('Bias: \n', regr.intercept_)
>>>>>>> 16a049cf58cd3325dbc79d29ad70214501110745
    # The mean squared error
    print("Mean squared error: %.2f"
          % np.mean((regr.predict(test_X) - test_Y) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(test_X, test_Y))
    # Plot outputs
<<<<<<< HEAD

    """
    plt.scatter(test_X, test_Y, color='black')
    plt.plot(test_X, regr.predict(test_X), color='blue',
             linewidth=3)
=======
>>>>>>> 16a049cf58cd3325dbc79d29ad70214501110745
    # plt.scatter(test_X, test_Y, color='black')
    # plt.plot(test_X, regr.predict(test_X), color='blue',
    #          linewidth=3)

    # plt.xticks(())
    # plt.yticks(())

<<<<<<< HEAD
    plt.show()
    """
    return regr.coef_, regr.intercept_

=======
    # plt.show()
    return regr.coef_, regr.intercept_
>>>>>>> 16a049cf58cd3325dbc79d29ad70214501110745


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
    train_X = X[:-num_test]
    train_Y = Y[:-num_test]
    test_X = X[-num_test:]
    test_Y = Y[-num_test:]
    # train_X, test_X = preprocessing(train_X, test_X)

    regr = neural_network.MLPRegressor(hidden_layer_sizes=(20,), activation='relu', solver='adam', alpha='0.0001',
                                       learning_rate='adaptive', learning_rate_init=0.1, warm_start=True)
    regr.fit(train_X, train_Y)
    print('Coefficients: \n', regr.coef_)
    print("Mean squared error: %.2f"
          % np.mean((regr.predict(test_X) - test_Y) ** 2))
    print('Variance score: %.2f' % regr.score(test_X, test_Y))



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


<<<<<<< HEAD

def run_weights_in_astar(planning_problem, weights, bias, heuristic_fn, a_star = True):
    graph = planning_problem[0]
    start = planning_problem[1][0]
    goal = planning_problem[2][0]
    feature_map = planning_problem[3]
    frontier = PriorityQueue()
    came_from = {}
    cost_so_far = {}
    frontier.put(start, 0, 0, 0)
    came_from[start] = None
    cost_so_far[start] = 0
    num_expansions = 0
    while not frontier.empty():
        num_expansions += 1
        current, current_priority = frontier.get()
        if current == goal:
            break
        neighbors = graph.neighbors(current)
        for next in neighbors:
            new_cost = cost_so_far[current] + graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                if not a_star:
                    priority = new_cost + (np.dot(weights, feature_map[next]) + bias)
                    frontier.put(next, priority, (np.dot(weights, feature_map[next]) + bias), new_cost)
                else:
                    priority = new_cost + heuristic_fn(next, goal)
                    frontier.put(next, priority, heuristic_fn(next, goal), new_cost)
               
                came_from[next] = current
                cost_so_far[next] = new_cost
                
    return came_from, cost_so_far, num_expansions
  
           
X, Y = getData(test_env_database)

# print X, Y

weights, bias = linearRegressionLearner(X, Y, batch_size, learning_rate, training_epochs, NUM_TEST)

sum_of_errors = 0
for i in range(122880, len(X)):
    sum_of_errors += math.pow(X[i][3] - Y[i], 2)

print "Mean squared error: ", sum_of_errors / (len(X) - 122880)

_, _, num_expansions = run_weights_in_astar(test_env_database[98], weights, bias, Manhattan, False)

print num_expansions

