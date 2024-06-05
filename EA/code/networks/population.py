from sklearn.preprocessing import StandardScaler, Normalizer
import numpy as np
from . import MLP

import random

class Population:
    def __init__(self, layers, X_train, y_train, X_test, y_test, problem_type = "regression"):
        self.individuals = []
        self.n_individuals = 0
        self.layers = layers
        self.problem_type = problem_type
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        self.norm = StandardScaler()
        self.ynorm = StandardScaler()
        
        self.norm.fit(self.X_train)
        self.ynorm.fit(self.y_train)
        
        self.y_train_norm = self.ynorm.transform(self.y_train, copy=True)
        self.X_train_norm = self.norm.transform(self.X_train, copy=True)
        
        
        self.y_test_norm = self.ynorm.transform(self.y_test, copy=True)
        self.X_test_norm = self.norm.transform(self.X_test, copy=True)
        
    def __str__(self) -> str:
        return(f"Population with {self.n_individuals} individuals")

        
    def generate_individuals(self, n_individuals = 10):
        self.n_individuals = n_individuals
        for i in range(n_individuals):
            self.individuals.append(MLP(self.layers, input=self.X_train.T, output_type=self.problem_type))
    
    def mutate_individual(self, individual, weights = True, biases = True, magnitude = 1):
        assert biases or weights, "Mutation has to do something, specify if mutation should be done on biases or weights"
        new_individual = MLP(self.layers, input=self.X_train.T, output_type=self.problem_type)
        for new_layer, layer in zip(new_individual.layers, individual.layers):
            
            if weights:
                new_layer.weights = layer.weights + np.random.normal(loc = 0, scale = magnitude, size = layer.weights.shape)
            if biases:
                new_layer.bias = layer.bias+  np.random.normal(loc = 0, scale = magnitude, size = layer.bias.shape)
                
        return new_individual
                
    def mutate_population(self, how_many_individuals_mutate = 0.7, weights = True, biases = True, magnitude = 1):
        
        individuals_to_mutate = random.sample(self.individuals, int(how_many_individuals_mutate * self.n_individuals))
        
        for individual in individuals_to_mutate:
            self.individuals.append(self.mutate_individual(individual, weights, biases, magnitude))

    def crossover(self, individual1, individual2):
        new_individual1 = MLP(self.layers, input=self.X_train.T, output_type=self.problem_type)
        new_individual2 = MLP(self.layers, input=self.X_train.T, output_type=self.problem_type)
        
        for i, (layer1, layer2) in enumerate(zip(new_individual1.layers, new_individual2.layers)):
            x_weights = random.randint(0, len(individual1.layers[i].weights) - 1)
            x_bias = random.randint(0, len(individual1.layers[i].bias) - 1)
            layer1.weights[:x_weights] = individual1.layers[i].weights[:x_weights]
            layer1.weights[x_weights:] = individual2.layers[i].weights[x_weights:]
            
            layer2.weights[:x_weights] = individual2.layers[i].weights[:x_weights]
            layer2.weights[x_weights:] = individual1.layers[i].weights[x_weights:]
            
            
            layer1.bias[:x_bias] = individual1.layers[i].bias[:x_bias]
            layer1.bias[x_bias:] = individual2.layers[i].bias[x_bias:]
            
            layer2.bias[:x_bias] = individual2.layers[i].bias[:x_bias]
            layer2.bias[x_bias:] = individual1.layers[i].bias[x_bias:]
            
        return new_individual1, new_individual2
    
    def crossover_population(self, how_many_individuals_crossover = 0.2):
        individuals_to_crossover = random.sample(self.individuals, int(how_many_individuals_crossover * self.n_individuals))
        
        for i in range(0, len(individuals_to_crossover), 2):
            new_ind1, new_ind2 = self.crossover(individuals_to_crossover[i], individuals_to_crossover[i+1])
            self.individuals.append(new_ind1)
            self.individuals.append(new_ind2)
            
    def select_individuals(self, elite = 0.1, temp = None):
        self.individuals = sorted(self.individuals, key = lambda x: self.score(x))        
        scores = self.get_scores()
        new_individuals = self.individuals[:int(elite * self.n_individuals)]
        
        if temp is None:
            temp = np.max(scores)
            
        scores = np.exp(-scores / temp)
        scores = scores / np.sum(scores)
        
        
        new_individuals += np.random.choice(self.individuals, p = scores, size = self.n_individuals - len(new_individuals), replace = True).tolist()
        
        self.individuals = new_individuals     
        
        
    def run(self, n_generations = 100, elite = 0.1, temp = None, how_many_individuals_crossover = 0.2, how_many_individuals_mutate = 0.7, weights = True, biases = True, magnitude = 1, verbose = False):
        
        if not hasattr(self, "best_scores"):
            self.best_scores = []
        if not hasattr(self, "avg_scores"):
            self.avg_scores = []
        
        for i in range(n_generations):
            self.best_scores.append(self.best_score())
            self.avg_scores.append(np.mean(self.get_scores()))
            if verbose:
                print(f"Generation {i}, best score: {self.best_scores[-1]}, avg score: {self.avg_scores[-1]}")  
            
            self.select_individuals(elite = elite, temp = temp)
            self.crossover_population(how_many_individuals_crossover)
            self.mutate_population(how_many_individuals_mutate, weights, biases, magnitude)
            
        return self.best_scores, self.avg_scores
            
    
    def predict(self, individual, train = True):
        if train:
            y_hat = individual.predict(self.X_train_norm.T)
        else:
            y_hat = individual.predict(self.X_test_norm.T)
        if self.problem_type == "classification":
            return y_hat.T
        return self.ynorm.inverse_transform(y_hat.T, copy=True)

    
    def score(self, individual, train = True):
        if train:        
            return individual.calculate_loss(self.X_train_norm.T, self.y_train_norm.T)
        return  individual.calculate_loss(self.X_test_norm.T, self.y_test_norm.T)

    
    def get_scores(self, train = True):
        return np.array([self.score(individual, train) for individual in self.individuals])
    
    def best_score(self, train = True):
        return min(self.get_scores(train))
        
        