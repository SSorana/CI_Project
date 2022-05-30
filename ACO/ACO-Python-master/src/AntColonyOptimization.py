import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import time
from Maze import Maze
from Ant import Ant
from Coordinate import Coordinate
from Route import Route
from PathSpecification import PathSpecification

# Class representing the first assignment. Finds shortest path between two points in a maze according to a specific
# path specification.
class AntColonyOptimization:

    # Constructs a new optimization object using ants.
    # @param maze the maze .
    # @param antsPerGen the amount of ants per generation.
    # @param generations the amount of generations.
    # @param Q normalization factor for the amount of dropped pheromone
    # @param evaporation the evaporation factor.
    def __init__(self, maze, ants_per_gen, generations, q, evaporation):
        self.maze = maze
        self.ants_per_gen = ants_per_gen
        self.generations = generations
        self.q = q
        self.evaporation = evaporation

     # Loop that starts the shortest path process
     # @param spec Spefication of the route we wish to optimize
     # @return ACO optimized route
    def find_shortest_route(self, path_specification):
        # initialize the shortest route
        shortest_route = Route(Coordinate(-1, -1))
        for j in range(0, self.generations):
            # create an array to add all routes from a generation
            routes = []
            for i in range(0, self.ants_per_gen):
                ant = Ant(self.maze, path_specification)
                route = ant.find_route()
                routes.append(route)
                # if it is our first route we update it to be the shortest
                if shortest_route.get_start() == Coordinate(-1, -1):
                    shortest_route = route
                # here we update the shortest route
                if route.shorter_than(shortest_route):
                    shortest_route = route

            # here we update the pheromone with formulas after a generation has passed and evaporate
            self.maze.evaporate(self.evaporation)
            self.maze.add_pheromone_routes(routes, self.q)

        self.maze.reset()
        return shortest_route

# Driver function for Assignment 1
# in the find_shortest_path we run the ant find_route for every generation on every ant
# after every generation we evaporate and add_pheromone_routes (methods from maze class)
# to deposit the pheromone for next generation
if __name__ == "__main__":
    #parameters
    gen = 10
    no_gen = 100
    q = 1000
    evap = 0.7

    #construct the optimization objects
    maze = Maze.create_maze("./../data/hard maze.txt")
    spec = PathSpecification.read_coordinates("./../data/hard coordinates.txt")
    aco = AntColonyOptimization(maze, gen, no_gen, q, evap)
    #save starting time
    start_time = int(round(time.time() * 1000))

    #run optimization
    shortest_route = aco.find_shortest_route(spec)

    #print time taken
    print("Time taken: " + str((int(round(time.time() * 1000)) - start_time) / 1000.0))

    #save solution
    shortest_route.write_to_file("./../data/hard_solution.txt")

    #print route size
    print("Route size: " + str(shortest_route.size()))