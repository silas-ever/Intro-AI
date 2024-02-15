# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        # Create some sort of Q table
        # Remember a Q table can be indexed by both state and action - to get the Q value
        # Can think of it as a 2D array or a 2D dictionary
        
        self.qtable = util.Counter() # Q table

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        # Look up the Q value in the Q table
        # Should return 0.0 if we have never seen a state before (make sure to implement properly)
        # Or return the Q-value otherwise

        if (state, action) not in self.qtable : # If we have never seen a state before
          self.qtable[(state, action)] == 0.0 # Set value at that pair as 0.0
          return 0.0 
        
        else : # If we have seen the state before
          return self.qtable[(state, action)] # Return Q value otherwise



    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # Loop through all possible actions at a given state
        # Note there is a self.getLegalActions(state) which returns legal actions for a state
        # Loop through those and figure out which one gives us the highest Q value just by using getQValue()
        #   for that state and action pair
        # Return the highest Q value

        maxAction = None # Initializing max action
        tempQValues = util.Counter() # Making a counter to store Q value info
        
        legalActions = self.getLegalActions(state) # Get legal actions

        if not (len(legalActions) == 0) : # If list of legal action is not empty
          for action in legalActions : # Loop through each legal action
            tempQValues[action] = self.getQValue(state, action) # Get Q value for that pair, store in the temp
          
          maxAction = tempQValues.argMax() # Get argMax of 
          qvalue = tempQValues[maxAction] # Use maxAction as an index into temp to get the Q value
          return qvalue
        return 0.0 # If legalActions is empty, return 0


    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Find the best action to take in a state
        # Do the same thing where you look at the legal actions, loop through them, and find the one that has the highest value
        #   Instead of returning the value, return the action

        bestAction = None # Initializing bestAction
        tempQValues = util.Counter() # Making a counter to store Q value info

        legalActions = self.getLegalActions(state) # Getting legal actions

        if not (len(legalActions) == 0) : # Making sure legalActions not empty

          for action in legalActions :  # Loop through all the legal actions
            tempQValues[action] = self.getQValue(state, action) # Get the Q value for that pair

        bestAction = tempQValues.argMax() # Find the argMax
        return bestAction # Return the highest value action


    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        # Can really just call computeActionFromQValues() here
        # In question 5, we will change getAction

        # Instead of just returning computeActionFromQValues, 
        #   use util.fipCoin and provide episilon to it as the probability
        # When you've gotten a problem that is less than epsilon, choose a random action 
        #   Use random.choice method
        # Otherwise, return highest action from Q values - acting on policy

        if util.flipCoin(self.epsilon) : # Choose a random action to proceed
          action = random.choice(legalActions) # Randomly choose from legal actions
        else : # Act on-policy
          action = self.computeActionFromQValues(state) # Get action from Q values

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # Implement Q-Learning Equation
        # self.learningRate, self.discount
        # Can find any of the Q(st, at) in the Q table 
        # Reward is provided in parameters
        # Not too many methods from other classes you should have to look for
        # Compute new Q value and update it in the Q table accordingly
        # This is the method where you actually change a Q value

        qvalue = self.getQValue(state, action) # Get Q value using state, action pair
        nextValue = self.getValue(nextState) # Get value of next State

        qvalue += self.alpha * (reward + self.discount * nextValue - qvalue) # Q-Learning Equation
        # Compute new Q value

        self.qtable[(state, action)] = qvalue # Update Q table with the new Q value


    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
