import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import random
from Route import Route
from Direction import Direction


# Class that represents the ants functionality.
class Ant:

    # Constructor for ant taking a Maze and PathSpecification.
    # @param maze Maze the ant will be running in.
    # @param spec The path specification consisting of a start coordinate and an end coordinate.
    def __init__(self, maze, path_specification):
        self.maze = maze
        self.start = path_specification.get_start()
        self.end = path_specification.get_end()
        self.current_position = self.start
        self.rand = random

    # Method that performs a single run through the maze by the ant.
    # @return The route the ant found through the maze.

    def find_route(self):
        route = Route(self.start)
        # this array will keep all elements that have been visited
        visited = []
        element = self.start
        # while we did not find a valid route -> to the end we try to search
        while element != self.end:
            visited.append(element)
            sum = 0
            probabilities = []
            directions = []
            # we try all neighbours to see if they are valid in order to create the probabilities
            east = element.add_direction(Direction.east)
            if self.is_valid(east, visited):
                sum += self.maze.walls[east.x][east.y]
                directions.append(0)
            north = element.add_direction(Direction.north)
            if self.is_valid(north, visited):
                sum += self.maze.walls[north.x][north.y]
                directions.append(1)
            west = element.add_direction(Direction.west)
            if self.is_valid(west, visited):
                sum += self.maze.walls[west.x][west.y]
                directions.append(2)
            south = element.add_direction(Direction.south)
            if self.is_valid(south, visited):
                sum += self.maze.walls[south.x][south.y]
                directions.append(3)
            # we calculate the probabilities
            if 0 in directions:
                if sum != 0:
                    probabilities.append(self.maze.walls[east.x][east.y] / sum)
                else:
                    probabilities.append(0)

            if 1 in directions:
                if sum != 0:
                    probabilities.append(self.maze.walls[north.x][north.y] / sum)
                else:
                    probabilities.append(0)

            if 2 in directions:
                if sum != 0:
                    probabilities.append(self.maze.walls[west.x][west.y] / sum)
                else:
                    probabilities.append(0)

            if 3 in directions:
                if sum != 0:
                    probabilities.append(self.maze.walls[south.x][south.y] / sum)
                else:
                    probabilities.append(0)
            # if we hit a dead end then we cannot move and we need to go back
            if len(directions) == 0:
                element = element.subtract_direction(route.remove_last())
                continue
            dir = self.rand.choices(directions, probabilities)
            if dir == [1]:
                element = element.add_direction(Direction.north)
            if dir == [2]:
                element = element.add_direction(Direction.west)
            if dir == [0]:
                element = element.add_direction(Direction.east)
            if dir == [3]:
                element = element.add_direction(Direction.south)
            # we finally add the step to the route
            route.add(Direction(dir[0]))
        return route

    # this method checks if a position is valid (still in the maze, not visited and not a wall)
    def is_valid(self, element, visited):
        if not self.maze.in_bounds(element):
            return False
        if self.maze.walls[element.x][element.y] == 0:
            return False
        if element in visited:
            return False
        return True
