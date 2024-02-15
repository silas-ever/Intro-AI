# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState) - don't have to provide nextState
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # Essentially implementing the Bellman Update Equation
        # Loop through every state in the MDP environment
        # do it self.iterations times
            # for each state in the MDP, compute max q value at that state
            # store all values at that iteration in the copy
            # notice there is a self.values
            # make a copy of self.values at the beginning of each iteration
            # at the end, set self.values to the copied version that you are actually updating

        copy = util.Counter() # Initialize a copy counter
        i = 0 # Iteration Counter
        qvalue = 0 # Initialize Q value

        for i in range(self.iterations) : # Loop through MDP environment self.iterations times
            copy = self.values.copy() # Make a copy of self.values at the beginning of each iteration

            for state in self.mdp.getStates() : # For each state in MDP

                if not self.mdp.isTerminal(state) : # If not a terminal state
                    action = self.getAction(state) 
                    qvalue = self.getQValue(state, action) # Compute max Q value at that state
                    copy[state] = qvalue # Store all values at that iteration into the copy

            self.values = copy # Making sure new computed values aren't being based upon the values you're re-computing at each iteration


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # A part of the Bellman Update Equation - everything inside of the parentheses
        # For that specific state and action pair, compute the reward for all of the next states
        # Look at the transition probabilities
        # Look up in self.values what the next state has a value of
        # We essentially want to compute everything inside of the parentheses using methods from self.mdp
        # Ex: transition probability fn - getTransitionStatesAndProbs
        # reward: self.mdp.getReward
        # looping through the actions self.mdp.getPossibleActions
        # Tells the value of the summation itself

        qvalue = 0 # Initialize Q value
        
        for transition in self.mdp.getTransitionStatesAndProbs(state, action) : # For all of the next states
            nextState, probability = transition # Unload the tuple
            nextValue = self.values[nextState] # Get the next value of the next state for our calculation

            reward = self.mdp.getReward(state, action, nextState) # Calculate reward for our state
            qvalue += probability * (reward + self.discount * nextValue) # Calculate Q value

        return qvalue



    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Does the same thing as QValueFromValues, but returns the action isntead of the Q value
        # Think of it as an argmax over the Q values for that action
        # This will be our policy - decides what action to take 

        possibleActions = util.Counter() # Making a counter for our actions

        for action in self.mdp.getPossibleActions(state) : # For each possible action
            possibleActions[action] = self.getQValue(state, action) # Get Q value and store it at that action
    
        policy = possibleActions.argMax() # Get the argmax of the actions - that is our policy
        return policy # Return the action we want to take

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
