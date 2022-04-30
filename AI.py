

# importing needed packages
from datetime import datetime
from collections import deque
import random
import numpy as np
# import copy

""""-----------------------------------------------------------------------------------------------------"""
# heap functions:


# Push item onto heap, maintaining the heap invariant.
def heappush(heap, item):
    heap.append(item)
    _siftdown(heap, 0, len(heap)-1)


# Pop the smallest item off the heap, maintaining the heap invariant.
def heappop(heap):
    lastelt = heap.pop()    # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        _siftup(heap, 0)
        return returnitem
    return lastelt


# 'heap' is a heap at all indices >= startpos, except possibly for pos.  pos
# is the index of a leaf with a possibly out-of-order value.  Restore the
# heap invariant.
def _siftdown(heap, startpos, pos):
    newitem = heap[pos]
    # Follow the path to the root, moving parents down until finding a place
    # newitem fits.
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        if newitem[1] < parent[1]:
            heap[pos] = parent
            pos = parentpos
            continue
        break
    heap[pos] = newitem


def _siftup(heap, pos):
    endpos = len(heap)
    startpos = pos
    newitem = heap[pos]
    # Bubble up the smaller child until hitting a leaf.
    childpos = 2*pos + 1    # leftmost child position
    while childpos < endpos:
        # Set childpos to index of smaller child.
        rightpos = childpos + 1
        if rightpos < endpos and not heap[childpos][1] < heap[rightpos][1]:
            childpos = rightpos
        # Move the smaller child up.
        heap[pos] = heap[childpos]
        pos = childpos
        childpos = 2*pos + 1
    # The leaf at pos is empty now. Put new item there, and bubble it up
    # to its final resting place (by sifting its parents down).
    heap[pos] = newitem
    _siftdown(heap, startpos, pos)


""""-----------------------------------------------------------------------------------------------------"""

"""class Node is used as nodes for the Goal tree class, in addition to having the expansion function and any other
cost functions."""


class Node:
    """constructor of the Node class."""
    def __init__(self, state, g, parent=None, action=None, children=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = children
        self.g = g

    """"return if self.state = state, false otherwise."""
    def equal(self, state):
        for i in range(len(state)):
            for j in range(len(state)):
                if self.state[i][j] != state[i][j]:
                    return False
        return True

    """calculate the distance between every square and its goal position measured along axes at right angles"""
    def manhattan_distance(self):
        distance = 0
        for i in range(len(self.state)):
            for j in range(len(self.state)):
                if self.state[i][j] != 0:
                    x, y = divmod(self.state[i][j] - 1, len(self.state))
                    distance += abs(x - i) + abs(y - j)
        return distance

    """return the estimated cost to reach the goal state."""
    def get_h(self):
        return self.manhattan_distance(self)

    def get_f(self):
        return self.get_h(self) + self.g

    """return the list of possible states after exactly one move."""
    def expand(self):
        x, y = self.find(self.state, 0)
        """ Direc contains position values for moving the blank space in either of
            the 4 directions [up,down,left,right] . """
        direc = {"Left": [x, y - 1], "Right": [x, y + 1], "Up": [x - 1, y], "Down": [x + 1, y]}
        childs = []
        for i in direc:
            child = self.move(self.state, x, y, direc[i][0], direc[i][1])
            if child is not None:
                child_node = Node(child, self.g + 1, self, i)
                childs.append(child_node)
        self.children = childs
        return childs

    """move the blank space in the given direction and if the position value are out
                of limits then return None """
    def move(self, state, x1, y1, x2, y2):
        if (0 <= x2 < len(self.state)) and (0 <= y2 < len(self.state)):
            # You can also use { copy.deepcopy(state) } by importing copy library, but it's a little bit slower than
            # the explicit one down there.
            temp_state = self.copy(state)

            temp = temp_state[x2][y2]
            temp_state[x2][y2] = temp_state[x1][y1]
            temp_state[x1][y1] = temp
            return temp_state
        else:
            return None

    """specifically used to find the position of the blank space """
    def find(self, state, x):
        for i in range(len(self.state)):
            for j in range(len(self.state)):
                if state[i][j] == x:
                    return i, j

    """ Copy function to create a similar matrix of the given node"""
    def copy(self, state):
        temp = []
        for row in state:
            temp_row = []
            for element in row:
                temp_row.append(element)
            temp.append(temp_row)
        return temp


""""-----------------------------------------------------------------------------------------------------"""

""" Goal tree is the main class of the software, 
having the search methods and the closed list, in addition to other things """


class GoalTree:
    # constructor for the class goal tree.
    def __init__(self, initial_state):
        self.root = Node(initial_state, 0)

    # Goal test function.
    @staticmethod
    def is_goal(state):
        counter = 1
        last = len(state) * len(state)

        for x in range(len(state)):
            for y in range(len(state)):
                # we check if the matrix in ascending order, but exclude the last element because it is the empty square
                if counter != last:
                    if state[x][y] != counter:
                        return False
                counter = counter + 1
        return True

    # the interactive method of the class
    def solve(self, strategy):
        if strategy.lower() == 'breadth first':
            start = datetime.now()
            sol_state, sol, g, processed_nodes, max_stored_nodes, flag = self.breadth_first()
        elif strategy.lower() == 'depth first':
            start = datetime.now()
            sol_state, sol, g, processed_nodes, max_stored_nodes, flag = self.depth_first()
        elif strategy.lower() == 'uniform cost':
            start = datetime.now()
            sol_state, sol, g, processed_nodes, max_stored_nodes, flag = self.uniform_cost()
        elif strategy.lower() == 'depth limited':
            limit = int(input('Enter a limit -> '))
            start = datetime.now()
            sol_state, sol, g, processed_nodes, max_stored_nodes, flag = self.depth_limited(limit)
        elif strategy.lower() == 'iterative deepening':
            start = datetime.now()
            sol_state, sol, g, processed_nodes, max_stored_nodes, flag = self.iterative_deepening()
        elif strategy.lower() == 'a*' or strategy.lower() == 'a star':
            start = datetime.now()
            sol_state, sol, g, processed_nodes, max_stored_nodes, flag = self.a_star()
        elif strategy.lower() == 'greedy' or 'best first':
            start = datetime.now()
            sol_state, sol, g, processed_nodes, max_stored_nodes, flag = self.greedy()

        return sol_state, sol, g, processed_nodes, max_stored_nodes, flag, start

    """"-------------------------------------------------------------------------------------------"""
    """ Search methods:
            uninformed search methods: (breadth-first, depth-first, uniformed cost, depth limited, iterative deepening) """

    # find the shallowest solution.
    def breadth_first(self):
        node = Node(self.root.state, 0)
        self.root = node
        processed_nodes = 1
        max_stored_nodes = 1
        dim = len(node.state) * len(node.state)

        # case 1: if the initial state is the goal state
        if self.is_goal(node.state):
            sol = self.solution(node)
            return node.state, sol, node.g, processed_nodes, max_stored_nodes, True

        # case 2: searching for the goal state
        frontier = deque([node])
        explored = set()
        while True:
            # Failed outcome, (i.e. didn't find the goal state)
            if len(frontier) == 0:
                sol = self.solution(self.root)
                # will return the root state instead of node state to indicate that we don't find the solution
                return self.root.state, sol, node.g, processed_nodes, max_stored_nodes, False

            stored = len(frontier)
            if stored > max_stored_nodes:
                max_stored_nodes = stored

            # checking for the goal state
            node = frontier.popleft()
            if self.is_goal(node.state):
                sol = self.solution(node)
                return node.state, sol, node.g, processed_nodes, max_stored_nodes, True

            # recording visited states.
            temp = tuple(np.reshape(node.state, dim))
            explored.add(temp)
            children = node.expand()
            for child in children:
                child1 = tuple(np.reshape(child.state, dim))
                if child1 not in explored:
                    processed_nodes += 1
                    frontier.append(child)

    # find the deepest solution
    def depth_first(self):
        node = Node(self.root.state, 0)
        self.root = node
        processed_nodes = 1
        max_stored_nodes = 1
        dim = len(node.state) * len(node.state)

        # case 1: if the initial state is the goal state
        if self.is_goal(node.state):
            sol = self.solution(node)
            return node.state, sol, node.g, processed_nodes, max_stored_nodes, True

        # case 2: searching for the goal state
        frontier = [node]
        explored = set()
        while True:
            # Failed outcome, (i.e. didn't find the goal state)
            if len(frontier) == 0:
                sol = self.solution(self.root)
                # will return the root state instead of node state to indicate that we don't find the solution
                return self.root.state, sol, node.g, processed_nodes, max_stored_nodes, False

            stored = len(frontier)
            if stored > max_stored_nodes:
                max_stored_nodes = stored

            # checking for the goal state
            node = frontier.pop()
            if self.is_goal(node.state):
                sol = self.solution(node)
                return node.state, sol, node.g, processed_nodes, max_stored_nodes, True

            # recording visited states.
            temp = tuple(np.reshape(node.state, dim))
            explored.add(temp)
            children = node.expand()
            for child in children:
                child1 = tuple(np.reshape(child.state, dim))
                if child1 not in explored:
                    processed_nodes += 1
                    frontier.append(child)

    def uniform_cost(self):
        node = Node(self.root.state, 0)
        self.root = node
        processed_nodes = 1
        max_stored_nodes = 1
        dim = len(node.state) * len(node.state)

        # case 1: if the initial state is the goal state
        if self.is_goal(node.state):
            sol = self.solution(node)
            return node.state, sol, node.g, processed_nodes, max_stored_nodes, True

        # case 2: searching for the goal state
        frontier = []
        heappush(frontier, (node, node.g))
        explored = set()
        while True:
            # Failed outcome, (i.e. didn't find the goal state)
            if len(frontier) == 0:
                sol = self.solution(self.root)
                # will return the root state instead of node state to indicate that we don't find the solution
                return self.root.state, sol, node.g, processed_nodes, max_stored_nodes, False

            stored = len(frontier)
            if stored > max_stored_nodes:
                max_stored_nodes = stored

            # checking for the goal state
            node = heappop(frontier)[0]
            if self.is_goal(node.state):
                sol = self.solution(node)
                return node.state, sol, node.g, processed_nodes, max_stored_nodes, True

            # recording visited states.
            temp = tuple(np.reshape(node.state, dim))
            explored.add(temp)
            children = node.expand()
            # add unexplored states to frontier
            for child in children:
                child1 = tuple(np.reshape(child.state, dim))
                if child1 not in explored:
                    processed_nodes += 1
                    heappush(frontier, (child, child.g))

    # without explored set
    def depth_limited(self, limit):
        node = Node(self.root.state, 0)
        self.root = node
        processed_nodes = 1
        max_stored_nodes = 1

        # case 1: if the initial state is the goal state
        if self.is_goal(node.state):
            sol = self.solution(node)
            return node.state, sol, node.g, processed_nodes, max_stored_nodes, True

        # case 2: searching for the goal state
        frontier = [node]
        # explored = set()
        while True:
            # Failed outcome, (i.e. didn't find the goal state)
            if len(frontier) == 0:
                sol = self.solution(self.root)
                # will return the root state instead of node state to indicate that we don't find the solution
                return self.root.state, sol, node.g, processed_nodes, max_stored_nodes, False

            stored = len(frontier)
            if stored > max_stored_nodes:
                max_stored_nodes = stored

            # checking for the goal state
            node = frontier.pop()
            if self.is_goal(node.state):
                sol = self.solution(node)
                return node.state, sol, node.g, processed_nodes, max_stored_nodes, True

            # recording visited states.
            if limit >= node.g + 1:
                children = node.expand()
                for child in children:
                    processed_nodes += 1
                    frontier.append(child)

    def iterative_deepening(self):
        level = 0
        flag = False
        total_processed_nodes = 0
        final_max_stored_nodes = 0
        while not flag:
            sol_state, sol, g, processed_nodes, max_stored_nodes, flag = self.depth_limited(level)
            total_processed_nodes += processed_nodes
            if final_max_stored_nodes < max_stored_nodes:
                final_max_stored_nodes = max_stored_nodes
            level += 1
        return sol_state, sol, g, total_processed_nodes, final_max_stored_nodes, flag

    """-------------------------------------------------------------------------------------------"""
    """Informed search methods: (greedy search(best-first search), A*) """
    def greedy(self):
        node = Node(self.root.state, 0)
        self.root = node
        max_stored_nodes = 1
        processed_nodes = 1
        dim = len(node.state) * len(node.state)

        # case 1: if the initial state is the goal state
        if self.is_goal(node.state):
            sol = self.solution(node)
            return node.state, sol, node.g, processed_nodes, max_stored_nodes, True

        # case 2: searching for the goal state
        frontier = []
        heappush(frontier, (node, node.manhattan_distance()))
        explored = set()
        while True:
            # Failed outcome, (i.e. didn't find the goal state)
            if len(frontier) == 0:
                sol = self.solution(node)
                # will return the root state instead of node state to indicate that we don't find the solution
                return self.root.state, sol, node.g, processed_nodes, max_stored_nodes, False

            stored = len(frontier)
            if stored > max_stored_nodes:
                max_stored_nodes = stored
            # checking for the goal state
            node = heappop(frontier)[0]
            if self.is_goal(node.state):
                sol = self.solution(node)
                return node.state, sol, node.g, processed_nodes, max_stored_nodes, True

            # recording visited states.
            temp = tuple(np.reshape(node.state, dim))
            explored.add(temp)
            children = node.expand()
            # add unexplored states to frontier
            for child in children:
                child1 = tuple(np.reshape(child.state, dim))
                if child1 not in explored:
                    processed_nodes += 1
                    heappush(frontier, (child, child.manhattan_distance()))

    def a_star(self):
        node = Node(self.root.state, 0)
        self.root = node
        max_stored_nodes = 1
        processed_nodes = 1
        dim = len(node.state)*len(node.state)

        # case 1: if the initial state is the goal state
        if self.is_goal(node.state):
            sol = self.solution(node)
            return node.state, sol, node.g, processed_nodes, max_stored_nodes, True

        # case 2: searching for the goal state
        frontier = []
        heappush(frontier, (node, node.g + node.manhattan_distance()))
        explored = set()
        while True:

            # Failed outcome, (i.e. didn't find the goal state)
            if len(frontier) == 0:
                sol = self.solution(self.root)
                # will return the root state instead of node state to indicate that we don't find the solution
                return self.root.state, sol, node.g, processed_nodes, max_stored_nodes, False

            stored = len(frontier)
            if stored > max_stored_nodes:
                max_stored_nodes = stored

            # checking for the goal state
            node = heappop(frontier)[0]
            if self.is_goal(node.state):
                sol = self.solution(node)
                return node.state, sol, node.g, processed_nodes, max_stored_nodes, True

            # recording visited states.
            temp = tuple(np.reshape(node.state, dim))
            explored.add(temp)
            children = node.expand()
            # add unexplored states to frontier
            for child in children:
                child1 = tuple(np.reshape(child.state, dim))
                if child1 not in explored:
                    processed_nodes += 1
                    heappush(frontier, (child, child.g + child.manhattan_distance()))

    @staticmethod
    def solution(node):
        sol = []
        current = node
        while current.parent is not None:
            sol.append(current.action)
            current = current.parent

        sol.reverse()
        return sol


""""-----------------------------------------------------------------------------------------------------"""
# Functions.


# check if a given state is solvable or not
def solvable(state):
    noi = number_of_inversion(state)
    n = len(state)
    rob = row_of_blank_from_bottom(state)
    if n % 2 == 1 and noi % 2 == 0:
        return True
    elif n % 2 == 0 and ((rob % 2 == 0 and noi % 2 == 1) or (rob % 2 == 1 and noi % 2 == 0)):
        return True
    else:
        return False


# generate random solvable state
def random_state(n):
    state1 = np.array(random.sample(range(n*n), n*n))
    state = np.reshape(state1, (n, n))
    while not solvable(state):
        state1 = np.array(random.sample(range(n * n), n * n))
        state = np.reshape(state1, (n, n))

    return state


"""-----------------------------------------------------------------------------------------------------"""


# Helper functions:
def number_of_inversion(state):
    row = []
    # turning the matrix into a raw
    n = len(state)
    for x in range(n):
        for y in range(n):
            row.append(state[x][y])
    noi = 0  # number of inversion (n.o.i)
    for i in range(len(row)):
        for j in row[i + 1:]:  # go through all elements after i
            if row[i] > j != 0:  # j is bigger than i and j isn't the empty element
                noi = noi + 1
    return noi


def row_of_blank_from_bottom(state):
    rob = 0
    for i in state:
        if 0 in i:
            return len(state) - rob
        rob = rob + 1


""""-----------------------------------------------------------------------------------------------------"""


""" ------- Just for testing --------"""

""" Random state """
# dim = int(input("Enter dimension -> "))
# initial = random_state(dim)
# print("The initial random state is:")
# for row in initial:
#     print(row)

""" Custom state """
initial = [[14, 15, 3, 4],
[10, 6, 7, 13],
[9, 0, 5, 11],
[12, 8, 2, 1]]



""" Choosing an algorithm """
algorithm = input("Choose an algorithm [Breadth first, Depth first, Uniform cost, Depth limited, Iterative deepening, \
Greedy, A*] -> ")

gt = GoalTree(initial)
info = gt.solve(algorithm)
stop_time = datetime.now()
sol_state, sol, g, processed_nodes, max_stored_nodes, find_sol, start_time = info

print("\n---------- Output Information ----------")
print("Time taken:", stop_time-start_time)
print(f'G-value (level solution found in goal tree): {g}')
print(f'Processed nodes: {processed_nodes}')
print(f'Max stored nodes: {max_stored_nodes}')
print(f'Do we find solution: {find_sol}')
print("Solution:\n" + str(sol))
print("Solution state:")
for row in sol_state:
    print(row)