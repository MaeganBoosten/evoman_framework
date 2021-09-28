import numpy as np
import itertools
import os
import pickle
from copy import deepcopy 
from keras.models import Sequential
from keras.layers import Dense, Activation
import sys
sys.path.insert(0, 'evoman')

from environment import Environment
from controller import Controller



sigmoid = lambda x: 1 / (1 + np.exp(-x))
relu    = lambda x: np.maximum(x, 0, x)

class NeuralNetwork(object):

    def __init__(self, 
                input_shape=20, 
                n_hidden=20,
                output_shape=5):


        self.input_shape = input_shape
        self._n_hidden = n_hidden
        self.output_shape = output_shape

        self._build_model()

    def _build_model(self):

        model = Sequential()
        model.add(Dense(units=self._n_hidden, input_dim=self.input_shape))
        model.add(Activation("relu"))
        model.add(Dense(units=self.output_shape))
        model.add(Activation("sigmoid"))

        self._model = model

    def get_weights(self):
        return self._model.get_weights()

    def set_weights(self, weights):
        return self._model.set_weights(weights)

    def return_model(self):
        return self._model

    def return_action(self, inputs):
        inputs = np.array([inputs])
        probabilities = np.round(self._model.predict(inputs))[0]

        left = probabilities[0]
        right = probabilities[1]
        jump = probabilities[2]
        shoot = probabilities[3]
        release = probabilities[4]

        return [left, right, jump, shoot, release]

    def load_model(self, saved_model):
        self._model.load_model(saved_model)

    def control(self, params, cont=None):
        params = (params-min(params))/float((max(params)-min(params)))
        action1, action2, action3, action4, action5 = self.return_action(params)

        return [action1, action2, action3, action4, action5]

class SimpleNeuralNetwork(object):

    def __init__(self, 
                input_shape=20, 
                n_hidden=10,
                output_shape=5):


        self.input_shape = input_shape
        self.n_hidden = n_hidden
        self.output_shape = output_shape

        self.W1 = self.initialize_weights((self.input_shape, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = self.initialize_weights((self.n_hidden, self.output_shape))
        self.b2 = np.zeros(self.output_shape)

        self.weights = [self.W1, self.b1, self.W2, self.b2]

    def predict(self, input):

        y1 = np.dot(input, self.W1) + self.b1
        h1 = relu(y1)
        y2 = np.dot(h1, self.W2) + self.b2
        pred = sigmoid(y2)

        return pred
    
    def get_weights(self):
        return self.weights

    def initialize_weights(self, shape):
        limit = np.sqrt( 6 / (shape[0] + shape[1]))
        return np.random.uniform(-limit, limit, size=shape)

    def set_weights(self, weights):
        self.weights = deepcopy(weights)

        self.W1 = self.weights[0]
        self.b1 = self.weights[1]
        self.W2 = self.weights[2]
        self.b2 = self.weights[3]
    
    def return_action(self, inputs):
        # inputs = np.array([inputs])
        
        probabilities = np.round(self.predict(inputs))

        left = probabilities[0]
        right = probabilities[1]
        jump = probabilities[2]
        shoot = probabilities[3]
        release = probabilities[4]

        return [left, right, jump, shoot, release]

    def control(self, params, cont=None):

        params = (params-min(params))/float((max(params)-min(params)))
        action1, action2, action3, action4, action5 = self.return_action(params)

        return [action1, action2, action3, action4, action5]
  
class GeneticAlgorithm(object):

    def __init__(self,
                savepath,
                population_size=10,
                number_of_generations=10,
                mutation_rate=0.1,
                es_strategy=False,
                load_model=False,
                model=None,
                state=0
                ):
        
        self.savepath = savepath
        self.population_size = population_size
        self.number_of_generations = number_of_generations
        self.mutation_rate = mutation_rate
        self.load_model = load_model
        self.state = state
        self.es_strategy = es_strategy
        

        if self.load_model==True:
            self.model = model
            with open(os.path.join(self.savepath, self.model), "rb") as fp:
                self.population = pickle.load(fp)
        else:
            self.population = None

    def set_env(self, env):
        self.env = env

    def create_population(self):
        population = dict()
        for i in range(self.population_size):
            # Initialize a neural model
            model = SimpleNeuralNetwork()

            # Calculate fitness
            f, p, e, t = self.fitness(model)
            
            population[model] = f
        
        self.population = population

    def fitness(self, model):
        f, p, e, t = self.env.play(pcont=model)
        print("Fitness: {}\t Player Life: {}\t Enemy Life: {}\n".format(f, p, e))

        return f, p, e, t

    def mutate(self, model):

        # Get model's weights
        weights = model.get_weights()
        n_hidden_layers = len(weights)

        mutate_layer = np.random.choice(n_hidden_layers)
        mutate_value = np.random.uniform(-1, 1)

        if weights[mutate_layer].ndim == 1:
            row_value = np.random.choice(weights[mutate_layer].shape[0])
            weights[mutate_layer][row_value] = mutate_value
        else:
            row_value = np.random.choice(weights[mutate_layer].shape[0])
            col_value = np.random.choice(weights[mutate_layer].shape[1])

            weights[mutate_layer][row_value][col_value] = mutate_value

        model.set_weights(weights)

    def crossover(self, parent, mother):
        # Getting parent and mother weights
        p_weights = parent.get_weights()
        m_weights = mother.get_weights()

        n_hidden_layers = len(p_weights)
        crossover_layer = np.random.choice(n_hidden_layers)

        original_shape = p_weights[crossover_layer].shape
        
        p_stacked_weight = np.hstack(p_weights[crossover_layer])
        m_stacked_weight = np.hstack(m_weights[crossover_layer])

        crossover_point = np.random.choice(p_stacked_weight.shape[0])

        tmp_weight = p_stacked_weight[:crossover_point].copy()
        p_stacked_weight[:crossover_point] = m_stacked_weight[:crossover_point]
        m_stacked_weight[:crossover_point] = tmp_weight

        p_weights[crossover_layer] = p_stacked_weight.reshape(original_shape)
        m_weights[crossover_layer] = m_stacked_weight.reshape(original_shape)

        parent.set_weights(p_weights)
        mother.set_weights(m_weights)

        return parent, mother

    def es_crossover(self, parent, mother):
        # Defining alpha
        alpha = np.random.uniform(0, 1)

        # Getting parent and mother weights
        p_weights = parent.get_weights()
        m_weights = mother.get_weights()

        n_hidden_layers = len(p_weights)
        crossover_layer = np.random.choice(n_hidden_layers)

        original_shape = p_weights[crossover_layer].shape
        
        p_stacked_weight = np.hstack(p_weights[crossover_layer])
        m_stacked_weight = np.hstack(m_weights[crossover_layer])

        crossover_weights = p_stacked_weight*alpha + (1-alpha)*m_stacked_weight

        p_weights[crossover_layer] = crossover_weights.reshape(original_shape)
        parent.set_weights(p_weights)

        return parent


    def select_contestant(self, population):
        return np.random.choice(list(population.keys()))

    def single_tournament(self, parent, mother):
        f_p, p, e, t = self.fitness(parent)
        f_m, p, e, t = self.fitness(mother)

        if f_p > f_m:
            return parent, f_p
        else:
            return mother, f_m


    def multiple_tournament(self, parents, offspring):
        new_population = {}

        while len(new_population) != self.population_size:
            contestant_1 = self.select_contestant(parents)
            contestant_2 = self.select_contestant(offspring)

            if parents[contestant_1] > offspring[contestant_2]:
                new_population[contestant_1] = parents[contestant_1]
            else:
                new_population[contestant_2] = offspring[contestant_2]

        return new_population


    def evolve(self):
        if not self.population:
            self.create_population()

        print("Starting Evolution:\n")
        for i in range(self.state, self.number_of_generations):
        
            # Sort population according to fitness value
            print("Best fit for generation: {} is {}".format(i, 
                                    np.max(list(self.population.values()))))

            offspring = {}
            while len(offspring) != self.population_size:

                parents = np.random.choice(list(self.population.keys()), 
                                          size=2,
                                          replace=True)

                new_parent_weight = parents[0].get_weights()
                new_mother_weight = parents[1].get_weights()

                parent = SimpleNeuralNetwork()
                mother = SimpleNeuralNetwork()

                parent.set_weights(new_parent_weight)
                mother.set_weights(new_mother_weight)

                if self.es_strategy==False:

                    parent, mother = self.crossover(parent, mother)
                    children, f_o = self.single_tournament(parent, mother)
                else:
                    children = self.es_crossover(parent, mother)
                    f_o, _, _, _ = self.fitness(children)
                
                offspring[children] = f_o

                # Check for mutation
                if self.mutation_rate > np.random.random():
                    self.mutate(children)
                    print("Mutation happened!\n")

            new_population = self.multiple_tournament(self.population, 
                                                          offspring)

                
            print("Population size: {}".format(len(new_population)))
            self.population = new_population

            self.save_state(i)
        

    def save_state(self, last_generation):
        filename = 'state_generation_' + str(last_generation) + '.pkl'

        with open(os.path.join(self.savepath, filename), 'wb') as fp:
            pickle.dump(self.population, fp)


    def control(self, params, controller):
        params = (params-min(params))/float((max(params)-min(params)))
        action1, action2, action3, action4, action5 = controller.return_action(params)

        return [action1, action2, action3, action4, action5]
      
      
##neuroevolution    

experiment_name = 'models/experiment_pop_100_gen_100'

if not os.path.exists(experiment_name):
	os.makedirs(experiment_name)

if __name__=='__main__':
    ga = GeneticAlgorithm(savepath=experiment_name,
                          population_size=100,
                          number_of_generations=100,
                          mutation_rate=0.3,
                          load_model=True,
                          es_strategy=True,
                          model='best_model',
                          state=120)

    env = Environment(speed="fastest",
                      enemymode="static",
                      player_controller=ga,
                      enemies=[1, 2, 4, 8],
                      multiplemode="yes",
                      level=2,
                      logs="off")

    env.update_parameter('contacthurt', 'player')

    ga.set_env(env)
    ga.evolve()
      
