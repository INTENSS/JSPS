# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 11:21:31 2021

@author: Fede Laptop
"""

########## IMPORTS ########### {{{1

from gym.envs.box2d import BipedalWalker

from MLP_fede import MLP
from individual_ga_fede import Individual
from container_fede import Container

import random
import torch
import numpy as np

import time

file_name = str(int(time.time()))

SEED = 1
RENDER = False

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
env = BipedalWalker()
env.seed(SEED)

N_INPUTS = 24
#print(N_INPUTS)
N_HIDDENS = [40, 40]
N_OUTPUTS = 4
IS_MASK = [True, True]
DISCONNECTABLE_UNITS = 0
for i in range(len(N_HIDDENS)):
    if IS_MASK[i]:
        DISCONNECTABLE_UNITS += N_HIDDENS[i]
        
INITIAL_POPULATION = 10000
BATCH_SIZE = 1000
GENERATIONS = 100
CONTAINER_X = 20
CONTAINER_Y = 20

'''fully connected or beta tresholded'''
FULLY_CONNECTED = False
ALPHA = 1.
BETA = 1.

TRIALS = 5
WORLD_ITERATIONS = 3000

PROBABILITY_INIT_VS_SELECT = 0.3
PROBABILITY_MUTATION = 0.01

'''mutation poly bounded'''
ETA = 20.
'''mutation gauss'''
MEAN = 0.
STD = 0.001

PROBABILITY_DISCONNECT = 0.01
PROBABILITY_CONNECT = 0.005



saved_parameters = {"seed": SEED, "initial_population": INITIAL_POPULATION, "batch_size": BATCH_SIZE, "generations": GENERATIONS, 
                    "container_x": CONTAINER_X, "container_y": CONTAINER_Y, "beta_dist_alpha": ALPHA, "beta_dist_beta": BETA,
                    "p_disconnect": PROBABILITY_DISCONNECT, "p_connect": PROBABILITY_CONNECT, "p_select": PROBABILITY_INIT_VS_SELECT,
                    "p_mutation": PROBABILITY_MUTATION, "mutation_mean": MEAN, "mutation_std": STD}

mlp_test = MLP(N_INPUTS, N_HIDDENS, N_OUTPUTS, IS_MASK)
container_test = Container(CONTAINER_X, CONTAINER_Y, mlp_test)

# individual = Individual(mlp_test.param_weights_count(), mlp_test.param_masks_count(), init_fully_connected=False)
# mlp_test.set_model_params(individual.genome_weights, individual.genome_masks)

'''save fitness'''
fitness_grid = np.empty((GENERATIONS+1, CONTAINER_X, CONTAINER_Y))
fitness_grid[:] = np.nan
    

'''generation 0'''
for p in range(INITIAL_POPULATION):
    
    if p%1000 == 0:
        print(p)
    
    env.reset()
    
    individual = Individual(mlp_test, beta_distribution_alpha=ALPHA, beta_distribution_beta=BETA, init_fully_connected=FULLY_CONNECTED)   
    individual.reset_fitness()
    
    mlp_test.set_model_params(individual.genome_weights, individual.genome_masks)
    '''we do the first step with X=0 for now'''
    output = mlp_test.forward(torch.zeros(N_INPUTS))
    
    individual_fitness = np.zeros(TRIALS)
    for t in range(TRIALS):        
        for _ in range(WORLD_ITERATIONS):
            # if RENDER:
            #     env.render('human')
            obs, reward, done, _ = env.step(output)
            output = mlp_test.forward(obs)
            individual_fitness[t] += reward
            #individual.set_fitness(reward)
            #print(output)
            #print(max(obs), "   ", np.argmax(np.array(obs)))
            #print(done)
            if done:
                #print('DEAD')
                break
    
    #print('fit: ', individual.get_fitness())
    individual.set_fitness(np.mean(individual_fitness))
    individual.set_evolutionary_history(0, DISCONNECTABLE_UNITS)
    container_test.place_individual_in_cell(individual)
    

#print(container_test.grid)    
'''save fitness'''
for i in range(CONTAINER_X):
    for j in range(CONTAINER_Y):
        if container_test.grid[i][j] is not None:
            fitness_grid[0][i][j] = container_test.grid[i][j].get_fitness()  
        
print(fitness_grid[0])
'''evolve'''
for g in range(1, GENERATIONS):
    #print(g)
    for i in range(BATCH_SIZE):
        env.reset()
        '''init a new genome or select'''
        if torch.rand(1) < PROBABILITY_INIT_VS_SELECT:
            individual = Individual(mlp_test, beta_distribution_alpha=ALPHA, beta_distribution_beta=BETA, init_fully_connected=FULLY_CONNECTED)  
        else:
            #individual = container_test.get_random_individual()
            individual = container_test.get_roulette_individual()
            #individual.mutation_polynomial_bounded(PROBABILITY_MUTATION, ETA, lower=-1., upper=1.)
            '''mutation mask first'''
            individual.mutation_pruning(PROBABILITY_DISCONNECT, PROBABILITY_CONNECT)   
            individual.mutation_gaussian(PROBABILITY_MUTATION, MEAN, STD)
        individual.reset_fitness()
            
        mlp_test.set_model_params(individual.genome_weights, individual.genome_masks)
        '''we do the first step with X=0 for now'''
        output = mlp_test.forward(torch.zeros(N_INPUTS))
        
        individual_fitness = np.zeros(TRIALS)
        for t in range(TRIALS): 
            for _ in range(WORLD_ITERATIONS):
                # if RENDER:
                #     env.render('human')
                obs, reward, done, _ = env.step(output)
                output = mlp_test.forward(obs)
                individual_fitness[t] += reward
                #individual.set_fitness(reward)
                #print(output)
                #print(max(obs), "   ", np.argmax(np.array(obs)))
                #print(done)
                if done:
                    #print('DEAD')
                    break
        
        # print('ind: ', i, '  ', individual.get_fitness(), '  features: ', 
        #       individual.get_feature_sparsity(), '  ', 
        #       individual.get_feature_disconnected_units(DISCONNECTABLE_UNITS))
        #print('fit: ', individual.get_fitness())
        individual.set_fitness(np.mean(individual_fitness))
        individual.set_evolutionary_history(g, DISCONNECTABLE_UNITS)
        container_test.place_individual_in_cell(individual)
        
    
    '''save fitness'''    
    for i in range(CONTAINER_X):
        for j in range(CONTAINER_Y):
            if container_test.grid[i][j] is not None:
                fitness_grid[g+1][i][j] = container_test.grid[i][j].get_fitness()  
    
    #print(container_test.grid)  
    if g%10 == 0:
        print(fitness_grid[g])          
        
grid = container_test.get_grid()   
env.close()

np.savez(file_name, fitness_grid, grid, saved_parameters)

for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        if grid[i][j] is not None:
            print(grid[i][j].evolutionary_history)

#mlp_test.set_model_params(individual.genome_weights, individual.genome_masks)

# print(individual.get_feature_sparsity())
# print(individual.get_feature_disconnected_units(DISCONNECTABLE_UNITS))

#env = gym.make('BipedalWalkerHardcore-v3')
#env = gym.make('BipedalWalker-v3')

'''
env = BipedalWalker()
env.reset()


for _ in range(3000):
    env.render('human')
    obs, reward, done, _ = env.step(env.action_space.sample())
    #print(max(obs), "   ", np.argmax(np.array(obs)))
    #print(done)
    if done:
        #print('DEAD')
        break
env.close()
'''