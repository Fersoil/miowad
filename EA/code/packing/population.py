import numpy as np
import matplotlib.pyplot as plt
import random

from . import Individual


from time import time

class Population:
    def __init__(self, r, rect_data, timeout=1000, elite = 0.1, temperature=0.1, p = None, rect_temp=0.1):
        self.individuals = []
        self.r = r
        self.rect_data = rect_data
        if p is None:
            self.p = np.zeros((len(rect_data)))
            for i in range(len(rect_data)):
                self.p[i] = rect_data.iloc[i]["value"] / (rect_data.iloc[i]["width"]* rect_data.iloc[i]["height"]) 
                self.p[i] = np.exp(self.p[i]/ rect_temp)
            self.p = self.p / np.sum(self.p)
        else:
            self.p = p
        self.timeout = timeout
        self.elite = elite
        self.temperature = temperature
        self.population_size = -1
    
    def get_n_elite(self):
        return int(self.population_size * self.elite)
        
    def generate_population(self, n_individuals=10, iterations=40):
        self.population_size = n_individuals
        for _ in range(n_individuals):
            individual = Individual([], self.r)
            for _ in range(iterations):
                rect = individual.generate_rect_from_data(self.rect_data, self.timeout)
            self.individuals.append(individual)
        
    def generate_individual(self, iterations=40):
        individual = Individual([], self.r)
        for _ in range(iterations):
            rect = individual.generate_rect_from_data(self.rect_data, self.timeout)
        self.individuals.append(individual)
        
    def scores(self):
        return np.array([individual.score() for individual in self.individuals])
    
    def best_individual(self):
        return max(self.individuals, key=lambda individual: individual.score())
    
    def mutate(self, type=["move"], axis="both", mutation_probs = None, how_many_individuals_mutate = 0.2, how_many_rects_move=0.1, magnitude_move=0.1, collide_strategy="skip", minus_sign_prob=0.2):
        if not isinstance(type, list):
            type = [type]
        if mutation_probs is None:
            mutation_probs = [1/len(type)] * len(type)
        
        assert len(mutation_probs) == len(type), "Mutation probabilities must have the same length as the mutation types"
        assert sum(mutation_probs) == 1, "Mutation probabilities must sum to 1"
        
        for individual in self.individuals:
            if random.uniform(0, 1) < how_many_individuals_mutate:
                t = np.random.choice(type, p=mutation_probs)
                individual.mutate(t, axis, how_many_move=how_many_rects_move, magnitude_move=magnitude_move, collide_strategy=collide_strategy, rect_data=self.rect_data, timeout=self.timeout, minus_sign_prob=minus_sign_prob, p=self.p)
                  
    def crossover(self, how_many_individuals_crossover=0.7, axis=["x", "y"], add_new_rects=True, add_new_rects_iterations=20):
        n_individuals_to_crossover = int(len(self.individuals) * how_many_individuals_crossover)
        if axis == "both":
            axis = ["x", "y"]
        if axis == "x":
            axis = ["x"]
        if axis == "y":
            axis = ["y"]
        
        
        for _ in range(n_individuals_to_crossover):
            i1 = random.choice(self.individuals)
            i2 = random.choice(self.individuals)
            
            new_i1, new_i2 = Individual.crossover(i1, i2, random.choice(axis), add_new_rects, add_new_rects_iterations, self.rect_data)
            self.individuals.append(new_i1)
            self.individuals.append(new_i2)
        
    
    def selection(self, elite=None, temperature=None):
        if elite is None:
            n_elite = self.get_n_elite()
        else:
            n_elite = int(elite * self.population_size)
        assert self.population_size > n_elite > 0, "Number of elite individuals must be greater than 0 and less than the population size"
        if temperature is None:
            temperature = self.temperature
        self.individuals = sorted(self.individuals, key=lambda individual: individual.score(), reverse=True)
        
        scores = self.scores()
        scores = scores[n_elite:]

        scores = np.exp(scores/temperature)
        
        scores = scores / np.sum(scores) 
        new_individuals = [0] * self.population_size
        new_individuals[:n_elite] = self.individuals[:n_elite]
        
        #new_population[elite:] = population[elite + np.random.choice(range(len(population) - elite), size = population_size - elite, p = scores)]
        
        # include the elite twice
        individuals_to_add = np.random.choice(range(n_elite, len(self.individuals)), size = self.population_size - n_elite, p = scores)
        for i, ind in enumerate(individuals_to_add):
            new_individuals[n_elite + i] = self.individuals[ind]
        self.individuals = new_individuals
        
    def run(self, n_generations=100, n_individuals=10, iterations=40, elite=None, temperature=None, type=["move"], axis="both", 
            mutation_probs = None, how_many_individuals_mutate = 0.2, how_many_rects_move=0.1, magnitude_move=0.1, collide_strategy="skip", 
            how_many_individuals_crossover=0.7, add_new_rects=True, add_new_rects_iterations=20, resume=False, minus_sign_prob=0.2):
        if resume:
            n_individuals = self.population_size
        else:
            self.generate_population(n_individuals, iterations)
        best_individual_scores = []
        avg_individual_scores = []
        times = []
        
        best_individual_scores.append(self.best_individual().score())
        avg_individual_scores.append(np.mean(self.scores()))
        time_start = time()
        times.append(time_start)
        
        for _ in range(n_generations):
            print(f"Generation: {_}")
            self.mutate(type, axis, how_many_individuals_mutate=how_many_individuals_mutate, how_many_rects_move=how_many_rects_move, magnitude_move=magnitude_move, collide_strategy=collide_strategy, mutation_probs=mutation_probs, minus_sign_prob=minus_sign_prob)
            self.crossover(how_many_individuals_crossover, axis, add_new_rects, add_new_rects_iterations=add_new_rects_iterations)
            self.selection(elite, temperature)
            
            best_individual_scores.append(self.best_individual().score())
            avg_individual_scores.append(np.mean(self.scores()))
            
            print(f"Best individual score in generation {_}: {self.best_individual().score()}")
            print(f"Average individual score in generation {_}: {np.mean(self.scores())}")
            times.append(time())
            print(f"Generation time: {times[-1] - times[-2]}")
        
        self.best_individual_scores = best_individual_scores
        self.avg_individual_scores = avg_individual_scores
        self.times = times
        
        return best_individual_scores, avg_individual_scores, times
        