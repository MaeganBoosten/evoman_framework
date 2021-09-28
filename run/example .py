import sys
import os
import numpy as np
import pickle
import logging
import re

sys.path.insert(0, 'evoman')
sys.path.insert(0, 'agents')

from environment import Environment
from controller import Controller

experiment_name = 'test'

if not os.path.exists(experiment_name):
	os.makedirs(experiment_name)

if __name__=='__main__':
    # Defining controller
    with open('models/experiment_pop_100_gen_100/best_model', 'rb') as fp:
        population = pickle.load(fp)

    population_ordered = dict(sorted(population.items(), key=lambda x: x[1], reverse=True))

    env = Environment(speed="normal",
                      enemymode="static",
                      player_controller=list(population_ordered.keys())[0],
                      enemies=[1, 2, 3, 4, 5, 6, 7, 8],
                      multiplemode="yes",
                      level=2,
                      logs='on')

    env.update_parameter('contacthurt', 'player')

    f, p, e, t = env.play()

##Collecting  Logs

regex = re.compile(r'(\d+)')

def open_file(filepath, generation):
    with open(os.path.join(filepath, generation), 'rb') as fp:
        population = pickle.load(fp)
    return population

if __name__=='__main__':
    # Defining controller
    trained = [1, 5, 7, 8]
    tested  = [3, 4, 2, 6]

    experiment_name = 'test'

    filepath = os.path.join('models', experiment_name)
    files = os.listdir(filepath)

    generations = {int(regex.findall(file)[0]): file for file in files}
    generations =  sorted(generations.items(), key=lambda x: x[0])
    
    results = dict()

    for generation in generations:
        population = open_file(filepath, generation[1])
        networks = list(population.keys())

        for network in networks:
            
            env = Environment(speed="fastest",
                            enemymode="static",
                            player_controller=network,
                            enemies=tested,
                            multiplemode="yes",
                            level=2,
                            logs='off')

            env.update_parameter('contacthurt', 'player')

            f, p, e, t = env.play()
            results[(generation[0], network)] = [f, p, e, t]
    
    with open(os.path.join(filepath, 'all_results_tested.pkl'), 'wb') as fp:
            pickle.dump(results, fp)
	
