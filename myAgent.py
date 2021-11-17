import numpy as np

# myAgent.py
# Author: Jada Mataroa
# Date: 20/05/2021
# Implements a genetic algorithm to optimise the fitness of a species of creatures in a 2D grid-based game

playerName = "myAgent"
nPercepts = 75  # This is the number of percepts
nActions = 5  # This is the number of actions
maxVal = 1  # Maximum value to be used in chromosome initialisation
minVal = -1  # Minimum value to be used in chromosome initialisation
p_cross = 0.8  # Probability of crossover
p_mut = 0.01  # Probability of mutation
p_elitism = 0.9  # Probability of elitism
sample = 10  # Sample size for tournament selection
elite = 4  # Number of elite creatures from old population to be used in elitism

# Train against random for 500 generations, then against self for 100 generations
trainingSchedule = [("random", 500), ("self", 100)]


# Class lass that initialises with random values chromosome and contains the AgentFunction that implements a model
# parametrised by the chromosome mapping percepts to returned vector of actions
class MyCreature:
    # Initialises chromosome with random values between minVal and maxVal
    def __init__(self):
        self.chromosome = np.random.uniform(minVal, maxVal, (nPercepts * nActions))

    # Maps percepts to actions using weights in the creature's chromosome
    def AgentFunction(self, percepts):
        actions = np.zeros(nActions)
        inputs = percepts.flatten()
        weights = self.chromosome
        gene = 0
        for a in range(len(actions)):
            score = 0
            for p in range(len(inputs)):
                score += weights[gene] * inputs[p]
                gene += 1
            actions[a] = score

        high = np.argwhere(actions == np.amax(actions))  # array of highest value(s) in actions

        # eliminates first occurrence found always being used if more than one high value in actions
        if len(high) > 1:
            index = high[np.random.randint(0, len(high))]
            actions[index] += 1

        return actions


# Fitness function that calculates and returns the fitness of each creature in the given population
def fitness_function(pop):
    fitness = np.zeros(len(pop))
    for n, cr in enumerate(pop):
        fitness[n] = 10 * (cr.size + cr.enemy_eats + cr.strawb_eats) * cr.alive
    return fitness


# Orders the population by fitness in descending order and returns that population of a given final size
def order(pop, fitness, final_size):
    N = len(pop)
    fit_comp = np.copy(fitness)
    ordered_index = np.zeros(N)
    temp = list()

    for i in range(0, N):
        highF = np.argwhere(fit_comp == np.amax(fit_comp))[0]  # finds first instance of highest fitness index
        ordered_index[i] = highF  # fills array with ordered fitness index in order
        fit_comp[highF] = -1  # takes fitness out of consideration by making it lowest fitness

    for i in range(N):
        index = int(ordered_index[i])
        temp.append(pop[index])  # appends creatures in order of fitness using ordered index array

    return temp[:final_size]


# Selects 2 fittest parents within a sample of 10 from a given population and returns them
# Implements tournament selection where k = 10
def parent_select(pop):
    numParents = 2

    # create sample of a random 10 parents from the population then calculate and order their fitness
    parents = np.random.permutation(pop)[:sample]
    fitness = fitness_function(parents)
    parents = order(parents, fitness, numParents)

    # take 2 fittest parents from the sample by selecting the first 2 parents in the ordered fitness list
    p1 = parents[0]
    p2 = parents[1]
    return p1.chromosome, p2.chromosome


# Crosses over parents chromosomes to create 2 children then returns them
def crossover(p1, p2):
    # randomly select cross over point
    point = np.random.randint(1, (len(p1)-2))

    # initialise new crossed over chromosomes from parents
    cross1 = [p1[:point], p2[point:]]
    cross2 = [p2[:point], p1[point:]]

    # initialise new children with newly crossed over chromosomes
    c1 = MyCreature()
    c1.chromosome = np.concatenate(cross1)

    c2 = MyCreature()
    c2.chromosome = np.concatenate(cross2)

    return c1, c2


# Selects random gene(s) in a creatures chromosome to mutate
def mutation(child):
    # randomly select how many mutations (up to 1/20 of chromosome)
    num_mutations = np.random.randint(1, int(len(child.chromosome)/20))

    index = np.random.randint(0, len(child.chromosome))
    mut = np.random.uniform(minVal, maxVal, 1)
    while child.chromosome[index] == mut:  # ensures mutation is not the same as original value
        mut = np.random.uniform(minVal, maxVal)
    child.chromosome[index] = mut

    # if more than one mutation, carry out each mutation based on number of mutations value
    if num_mutations > 1:
        mutated = list()
        mutated.append(index)  # appends first mutation index to list use for double up checking
        for i in range(num_mutations-1):
            index = np.random.randint(0, len(child.chromosome))
            while index in mutated:  # ensures no index is mutated twice
                index = np.random.randint(0, len(child.chromosome))
            mutated.append(index)
            mut = np.random.uniform(minVal, maxVal)
            while child.chromosome[index] == mut:  # ensures mutation is not the same as original value
                mut = np.random.uniform(minVal, maxVal)
            child.chromosome[index] = mut
    return child


# Function that takes a list of MyCreature objects and returns a new list of MyCreature that constitutes a new
# generation of creatures and also returns the average fitness of the old population
def newGeneration(old_population):
    N = len(old_population)

    # Fitness for all agents
    fitness = fitness_function(old_population)

    new_population = list()

    for n in range(N):
        # Here you should modify the new_creature's chromosome by selecting two parents (based on their
        # fitness) and crossing their chromosome to overwrite new_creature.chromosome

        if (np.random.randint(0, 101) / 100) <= p_cross:
            p1, p2 = parent_select(old_population)
            c1, c2 = crossover(p1, p2)
            if (np.random.randint(0, 101) / 100) <= p_mut:
                c1 = mutation(c1)
            if (np.random.randint(0, 101) / 100) <= p_mut:
                c2 = mutation(c2)
            new_population.append(c1)
            new_population.append(c2)

    N2 = len(new_population)

    # ensures new population is the correct size just in case of error
    if N2 < N:
        old_population = order(old_population, fitness, N - N2)
        new_population += old_population  # fittest from old population are added to make up correct population size
    if N2 >= N:
        # if elitism, 4 from new population are replaced with 4 fittest from old population
        # else random 34 children are taken for new population
        if (np.random.randint(0, 101) / 100) <= p_elitism:
            old_population = order(old_population, fitness, elite)
            new_population = new_population[:N - elite]
            new_population += old_population

        else:
            new_population = new_population[:N]

    # At the end you need to compute average fitness and return it along with your new population
    avg_fitness = np.mean(fitness)

    return new_population, avg_fitness
