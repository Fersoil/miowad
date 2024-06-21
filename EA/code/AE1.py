import numpy as np 

class Individual:
    def __init__(self, x, y, z, function):
        self.params = [x, y, z]
                
        self.function = function
        
    def gaussian_mutation(self, i = 0, sigma = 1, low = -10, high = 10):
        new_param = self.params[i] + np.sqrt(2) * sigma * (high - low) * 
        
        return Individual(x, y, z, self.function)
        
        
def gaussian_mutation(individual, i = 0, mu = 0, sigma_0 = 1, low = -10, high = 10):
    # https://www.iue.tuwien.ac.at/phd/heitzinger/node27.html
    
    sigma = sigma_0 * (high - low)
    new_individual = Individual(individual.params[0], individual.params[1], individual.params[2], individual.function)
    
    new_individual.params[i] = np.min(np.max(individual.params[i] + np.random.normal(mu, sigma), low), high)
    
    return new_individual

def crossover(individual1, individual2):
    # https://www.iue.tuwien.ac.at/phd/heitzinger/node26.html
    i = np.random.randint(0, 2)
    j = np.random.randint(i, 3)
    
    new_individual_1 = Individual(individual1.params[0], individual1.params[1], individual1.params[2], individual1.function)
    new_individual_2 = Individual(individual2.params[0], individual2.params[1], individual2.params[2], individual2.function)
    
    new_individual_1.params[i:j] = individual2.params[i:j]
    new_individual_2.params[i:j] = individual1.params[i:j]
    
    return new_individual_1, new_individual_2


def modify_population(population, mutation_rate = 0.1, sigma = 1):
    new_population = Population(population.low, population.high)
    
    for individual in population.individuals:
        if np.random.uniform() < mutation_rate:
            new_population.add_individual(gaussian_mutation(individual, sigma = sigma))
        else:
            new_population.add_individual(individual)
    return new_population

def cross_populations(population1, population2):
    new_population = Population(population1.low, population1.high)
    
    for individual1, individual2 in zip(population1.individuals, population2.individuals):
        new_individual1, new_individual2 = crossover(individual1, individual2)
        new_population.add_individual(new_individual1)
        new_population.add_individual(new_individual2)
        
    return new_population
    
class Population:
    def __init__(self, low = -10, high = 10):
        self.individuals = []
        self.low = low
        self.high = high
        
    def add_individual(self, individual):
        self.individuals.append(individual)
        
    def create_uniform_individual(self):
        x = np.random.uniform(self.low, self.high)
        y = np.random.uniform(self.low, self.high)
        z = np.random.uniform(self.low, self.high)
        return Individual(x, y, z, self.function)
    
def create_initial_population(population_size, function, low = -10, high = 10):
    
    population = Population(low=low, high=high)
    
    for _ in range(population_size):
        population.create_uniform_individual()
        
    return population



    
square_func = lambda x, y, z: x**2 + y**2 + 2 * z**2
print(create_initial_population(1000, square_func))


