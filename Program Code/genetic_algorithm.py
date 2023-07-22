import matplotlib.pyplot as plt
import pymysql
from random import random
import time
import pandas as pd

start_time = time.time()
# Run the algorithm here

class Services():
    def __init__(self,responsetime,availability,throughput,successability,reliability,compliance,bestpractices,latency,servicename):
        self.responsetime= responsetime
        self.availability=availability
        self.throughput=throughput
        self.successability=successability
        self.reliability=reliability
        self.compliance=compliance
        self.bestpractices=bestpractices
        self.latency=latency
        self.servicename=servicename

class Individual():
    def __init__(self, responsetime, availability, throughput, successability, reliability, compliance, bestpractices,
                 latency, size_limit, generation=0):
        self.responsetime = responsetime
        self.availability = availability
        self.throughput = throughput
        self.successability = successability
        self.reliability = reliability
        self.compliance = compliance
        self.bestpractices = bestpractices
        self.latency = latency
        self.size_limit = size_limit
        self.score_evaluation = 0
        self.used_size = 0
        self.generation = generation
        self.chromosome1 = []
        chromosome1 = []
        self.chromosome = []

        for i in range(len(responsetime)):
            if responsetime[i] < 500 and bestpractices[i] > 85 and latency[i] < 10 and availability[i] > 80 and reliability[i] and random() > 0.5 and reliability[i] > 80:
                self.chromosome.append('1')
            else:
                self.chromosome.append('0')

    def fitness(self):
        score = 0
        sum_size = 0
        for i in range(len(self.chromosome)):
            if self.chromosome[i] == '1':
                score += (0.4 * self.responsetime[i]) + (0.2 * self.availability[i]) + (
                            0.05 * self.bestpractices[i]) + (0.1 * self.throughput[i]) + (0.05 * self.latency[i]) + (
                                     0.05 * self.compliance[i]) + (0.1 * self.successability[i]) + (
                                     0.1 * self.reliability[i])
                sum_size += throughput[i]
        if sum_size > self.size_limit:
            score = 1
        self.score_evaluation = score
        self.used_size = sum_size

    def crossover(self, other_individual):
        cutoff = round(random() * len(self.chromosome))
        # print("cutoff:",cutoff)
        child1 = other_individual.chromosome[0:cutoff] + self.chromosome[cutoff::]
        child2 = self.chromosome[0:cutoff] + other_individual.chromosome[cutoff::]
        # print(child1)
        # print(child2)
        children = [
            Individual(self.responsetime, self.availability, self.throughput, self.successability, self.reliability,
                       self.compliance, self.bestpractices, self.latency, self.size_limit, self.generation + 1),
            Individual(self.responsetime, self.availability, self.throughput, self.successability, self.reliability,
                       self.compliance, self.bestpractices, self.latency, self.size_limit, self.generation + 1)]
        children[0].chromosome = child1
        children[1].chromosome = child2
        return children

    def mutation(self, rate):
        # print('Before: ', self.chromosome)
        for i in range(len(self.chromosome)):
            if random() < rate:
                if self.chromosome[i] == '1':
                    self.chromosome[i] = '0'
                else:
                    self.chromosome[i] = '1'
        # print('After : ', self.chromosome)
        return self
class GeneticAlgorithm():
    def __init__(self, population_size):
        self.population_size = population_size
        self.population = []
        self.generation = 0
        self.best_solution = None
        self.list_of_solutions = []

    def initialize_population(self, responsetime, availability, throughput, successability, reliability, compliance,
                              bestpractices, latency, size_limit):
        for i in range(self.population_size):
            self.population.append(
                Individual(responsetime, availability, throughput, successability, reliability, compliance,
                           bestpractices, latency, size_limit))
        self.best_solution = self.population[0]

    def order_population(self):
        self.population = sorted(self.population, key=lambda population: population.score_evaluation, reverse=True)

    def best_individual(self, individual):
        if individual.score_evaluation > self.best_solution.score_evaluation:
            self.best_solution = individual

    def sum_evaluations(self):
        sum = 0
        for individual in self.population:
            sum += individual.score_evaluation
        return sum

    def select_parent(self, sum_evaluation):
        parent = -1
        random_value = random() * sum_evaluation
        sum = 0
        i = 0
        # print("random value:",random_value)
        while i < len(self.population) and sum < random_value:
            # print("i",i,"-sum",sum)
            sum += self.population[i].score_evaluation
            parent += 1
            i += 1
        return parent

    def visualize_generation(self):  # best population in each individual of generation
        best = self.population[0]
        print('Generation', self.population[0].generation,
              'Total:', best.score_evaluation, 'Space: ', best.used_size,
              'chromosome:', best.chromosome)

    def solve(self, mutation_probability, number_of_generation, responsetime, availability, throughput, successability,
              reliability, compliance, bestpractices, latency, size_limit):
        self.initialize_population(responsetime, availability, throughput, successability, reliability, compliance,
                                   bestpractices, latency, size_limit)
        for individual in self.population:
            individual.fitness()
        self.order_population()
        self.best_solution = self.population[0]
        self.list_of_solutions.append(self.best_solution.score_evaluation)
        self.visualize_generation()

        for genertion in range(number_of_generations):
            sum = self.sum_evaluations()
            new_population = []
            for new_individuals in range(0, self.population_size, 2):
                parent1 = self.select_parent(sum)
                parent2 = self.select_parent(sum)
                # print('\n',parent1,parent2)
                children = self.population[parent1].crossover(self.population[parent2])
                # print(ga.population[parent1].chromosome)
                # print(ga.population[parent2].chromosome)
                # print(children[0].chromosome)
                # print(children[1].chromosome)

                new_population.append(children[0].mutation(mutation_probability))
                new_population.append(children[1].mutation(mutation_probability))
            self.population = list(new_population)

            for individual in self.population:
                individual.fitness()
            self.visualize_generation()
            best = self.population[0]
            self.list_of_solutions.append(best.score_evaluation)
            self.best_individual(best)
        print('Best Solution -Generation', self.best_solution.generation,
              'Total:', self.best_solution.score_evaluation, 'Space: ', self.best_solution.used_size,
              'chromosome:', self.best_solution.chromosome)
        return self.best_solution.chromosome
if __name__ == '__main__':
    services_list =[]
    connection = pymysql.connect(host='localhost',user='root',passwd='Yuvaraj@2002',db='services')
    cursor =connection.cursor()
    cursor.execute('select servicename,responsetime,availability,throughput,successability,reliability,compliance,bestpractices,latency from services')
    for service in cursor :
            services_list.append(Services(service[1],service[2],service[3],service[4],service[5],service[6],service[7],service[8],service[0]))
    cursor.close()
    connection.close()

    # for service in services_list:
    #     print(service.servicename)

    servicename = []
    responsetime = []
    availability = []
    throughput = []
    successability = []
    reliability = []
    compliance = []
    bestpractices = []
    latency = []

    for services in services_list:
        servicename.append(services.servicename)
        responsetime.append(services.responsetime)
        availability.append(services.availability)
        throughput.append(services.throughput)
        successability.append(services.successability)
        reliability.append(services.reliability)
        compliance.append(services.compliance)
        bestpractices.append(services.bestpractices)
        latency.append(services.latency)

    size_limit = 2800
    population_size = 100
    mutation_probability = 0.1
    number_of_generations =100
    ga1 = GeneticAlgorithm(population_size)
    # ga2 = GeneticAlgorithm(population_size)

    result1= ga1.solve(mutation_probability, number_of_generations, responsetime, availability, throughput,
                        successability, reliability, compliance, bestpractices, latency, size_limit)

    for i in range(len(ga1.best_solution.chromosome)):
        if result1[i] == '1':
            print('Services Name', servicename[i])
    # result2= ga2.solve(mutation_probability, number_of_generations, responsetime, availability, throughput,
    #                     successability, reliability, compliance, bestpractices, latency, size_limit)
    #
    # for i in range(len(ga2.best_solution.chromosome)):
    #     if result2[i] == '1':
    #         print('Services Name', servicename[i])

    end_time = time.time()

    time_consumed = end_time - start_time
    print("Time consumed: ", time_consumed)

    plt.plot(ga1.list_of_solutions,c="blue")
    # plt.plot(ga2.list_of_solutions,c="black")
    plt.title("Genetic Algorithm Results")
    plt.xlabel("Fitness Value")
    plt.ylabel("best Solution")
    plt.show()

    df = pd.read_csv('ConvergenceRate.csv')
    x = df["populationsize"]
    y = df["NoOfgeneration"]
    z = df["Time"]

    plt.figure(figsize=(4, 5))
    plt.plot(x, y)
    plt.title("Over populationsize")
    plt.xlabel("Fitness Value")
    plt.ylabel("Executed Time")
    # Display the plot
    plt.show()

    plt.figure(figsize=(4, 5))
    plt.plot(x, z)
    plt.title("Over NoOfgeneration")
    plt.xlabel("Fitness Value")
    plt.ylabel("Executed Time")
    plt.show()


