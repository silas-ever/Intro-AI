# Silas Ever - using 1 late day
# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
# 
# Supplementary Resources I used: https://www.educative.io/edpresso/how-to-implement-depth-first-search-in-python

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

from sys import int_info
import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest currents in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    
    closedList = [] # Visited currents
    plan = [] # History of actions to the goal 
    openList = util.Stack() # Currents we know exist but have not visited
    current = problem.getStartState()

    if (problem.isGoalState(current)) : 
        return plan
    
    openList.push((current, plan)) # Add the initial state and metadata to openList

    while openList.isEmpty() == False :
        state = openList.pop() # Pop the last state pushed onto the openList
        curr = state[0]
        actions = state[1]

        if curr not in closedList : # Means curr has not been visited
            if problem.isGoalState(curr) == True : # Stop when we reach goal state
                for action in actions :
                    plan.append(action) # Reconstruct the path to the goal
                break

            closedList.append(curr) # Add successor to list of visited nodes
            currSuccessors = problem.getSuccessors(curr)
    
            for successor in currSuccessors :
                newSuccessors = successor[0]
                newAction = successor[1]

                path = actions + [newAction] # Construct new metadata
                openList.push((newSuccessors, path)) # Add new states to openList
    
    return plan


def breadthFirstSearch(problem):
    """
    Search the shallowest currents in the search tree first.
    """
    closedList = [] # Visited currents
    plan = [] # History of actions to the goal 
    openList = util.Queue() # Currents we know exist but have not visited
    current = problem.getStartState()

    if (problem.isGoalState(current)) : 
        return plan
    
    openList.push((current, plan)) # Add the initial state and metadata to openList

    while openList.isEmpty() == False :
        state = openList.pop() # Pop the last state pushed onto the openList
        curr = state[0]
        actions = state[1]

        if curr not in closedList : # Means curr has not been visited
            if problem.isGoalState(curr) == True : # Stop when we reach goal state
                for action in actions :
                    plan.append(action) # Reconstruct the path to the goal
                break

            closedList.append(curr) # Add successor to list of visited nodes
            currSuccessors = problem.getSuccessors(curr) # Get successors of curr
    
            for successor in currSuccessors : # Add each of the successors to openList
                newSuccessors = successor[0]
                newAction = successor[1]

                path = actions + [newAction] # Construct new metadata
                openList.push((newSuccessors, path)) # Push new states to openList
    return plan


def uniformCostSearch(problem):
    """
    Search the current of least total cost first.
    """
    closedList = [] # Visited currents
    plan = [] # History of actions to the goal 
    openList = util.PriorityQueue() # Currents we know exist but have not visited
    current = problem.getStartState()

    if (problem.isGoalState(current)) : 
        return plan
    
    openList.push((current, plan), 0) # Add the initial state and metadata to openList

    while openList.isEmpty() == False :
        state = openList.pop() # Pop the last state pushed onto the openList
        curr = state[0]
        actions = state[1]

        if curr not in closedList : # Means curr has not been visited
            if problem.isGoalState(curr) == True : # Stop when we reach goal state
                for action in actions :
                    plan.append(action) # Reconstruct the path to the goal
                break # Leave the loop once we've found the goal

            closedList.append(curr) # Add successor to list of visited nodes
            currSuccessors = problem.getSuccessors(curr)
    
            for successor in currSuccessors :
                newSuccessors = successor[0]
                newAction = successor[1]

                path = actions + [newAction] # Construct new metadata
                cost = problem.getCostOfActions(path) # Get new cost

                openList.push((newSuccessors, path), cost) # Add new states to openList
    return plan


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic = nullHeuristic):
    """
    Search the current that has the lowest combined cost and heuristic first.
    """
    closedList = [] # Visited currents
    plan = [] # History of actions to the goal 
    openList = util.PriorityQueue() # Currents we know exist but have not visited
    current = problem.getStartState()

    if (problem.isGoalState(current)) : 
        return plan
    
    openList.push((current, plan), 0) # Add the initial state and metadata to openList

    while openList.isEmpty() == False :
        state = openList.pop() # Pop the last state pushed onto the openList
        curr = state[0]
        actions = state[1]

        if curr not in closedList : # Means curr has not been visited
            if problem.isGoalState(curr) == True : # Stop when we reach goal state
                for action in actions :
                    plan.append(action) # Reconstruct the path to the goal
                break # Leave the loop once we've found the goal

            closedList.append(curr) # Add successor to list of visited nodes
            currSuccessors = problem.getSuccessors(curr)
    
            for successor in currSuccessors :
                newSuccessors = successor[0]
                newAction = successor[1]

                path = actions + [newAction] # Construct new metadata
                cost = problem.getCostOfActions(path) # Get new cost
                hValue = heuristic(newSuccessors, problem)

                openList.push((newSuccessors, path), cost + hValue) # Add new states to openList
    return plan

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch