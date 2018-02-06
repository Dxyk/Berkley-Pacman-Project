# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # if near ghost, run away
        for gp in successorGameState.getGhostPositions():
            possible = [gp, (gp[0] + 1, gp[1]), (gp[0] - 1, gp[1]), (gp[0], gp[1] + 1), (gp[0], gp[1] - 1)]
            if any(newPos == pgp for pgp in possible):
                return -float("inf")
        # find min distance and return
        old_food_list = currentGameState.getFood().asList()
        distance = []
        for food in old_food_list:
            distance.append(manhattanDistance(food, newPos))
        return -min(distance)




def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        return self.minimax(0, 0, gameState)[0]

    def minimax(self, player, depth, state):
        # first decide whether the current node is the terminal node
        if depth == self.depth or state.isWin() or state.isLose():
            return None, self.evaluationFunction(state)
        # only increment depth when the last ghost has made a decision
        if player == state.getNumAgents() - 1:
            depth += 1
        best_move = None
        value = -float("inf") if player == 0 else float("inf")
        for move in state.getLegalActions(player):
            next_state = state.generateSuccessor(player, move)
            next_player = (player+1)%state.getNumAgents()
            next_move, next_val = self.minimax(next_player, depth, next_state)
            if player == 0 and value < next_val:
                best_move, value = move, next_val
            elif player != 0 and value > next_val:
                best_move, value = move, next_val
        return best_move, value


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.alphaBeta(0, 0, gameState, -float("inf"), float("inf"))[0]

    def alphaBeta(self, player, depth, state, alpha, beta):
        # first decide whether the current node is the terminal node
        if depth == self.depth or state.isWin() or state.isLose():
            return None, self.evaluationFunction(state)
        # only increment depth when the last ghost has made a decision
        if player == state.getNumAgents() - 1:
            depth += 1
        best_move = None
        value = -float("inf") if player == 0 else float("inf")
        for move in state.getLegalActions(player):
            next_state = state.generateSuccessor(player, move)
            next_player = (player+1)%state.getNumAgents()
            next_move, next_val = self.alphaBeta(next_player, depth, next_state, alpha, beta)
            if player == 0:
                if value < next_val:
                    best_move, value = move, next_val
                if value >= beta:
                    return best_move, value
                alpha = max(alpha, value)
            else:
                if value > next_val:
                    best_move, value = move, next_val
                if value <= alpha:
                    return best_move, value
                beta = min(beta, value)
        return best_move, value


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectimax(0, 0, gameState)[0]

    def expectimax(self, player, depth, state):
        # first decide whether the current node is the terminal node
        if depth == self.depth or state.isWin() or state.isLose():
            return None, self.evaluationFunction(state)
        # only increment depth when the last ghost has made a decision
        if player == state.getNumAgents() - 1:
            depth += 1
        best_move = None
        value = -float("inf") if player == 0 else float(0)
        for move in state.getLegalActions(player):
            next_state = state.generateSuccessor(player, move)
            next_player = (player+1)%state.getNumAgents()
            next_move, next_val = self.expectimax(next_player, depth, next_state)
            if player == 0 and value < next_val:
                best_move, value = move, next_val
            elif player != 0:
                value += float(1)/float(len(state.getLegalActions(player))) * float(next_val)
        return best_move, value

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    cur_pos = currentGameState.getPacmanPosition()
    cur_food = currentGameState.getFood()
    cur_food_list = cur_food.asList()
    curr_food_count = currentGameState.getNumFood()
    cur_ghost_state = currentGameState.getGhostStates()
    cur_scared_times = [ghostState.scaredTimer for ghostState in cur_ghost_state]

    # print "-" * 20
    # print "current position: ", cur_pos
    # print "current food: ", curr_food_count

    # ghost_distance = [manhattanDistance(cur_pos, ghost.getPosition())for ghost in cur_ghost_state]
    # food_distance = breadthFirstSearch(currentGameState)
    #
    # # return 10 * max(ghost_distance) - food_distance
    # # print food_distance
    # return food_distance

    ghost_distance = sum([manhattanDistance(cur_pos, ghost.getPosition()) for ghost in cur_ghost_state])
    if sum(cur_scared_times) > 0:
        return 100 * curr_food_count
    else:
        return ghost_distance - curr_food_count


""" HELPER FUNCTIONS """

def breadthFirstSearch(state):
    """Search the shallowest nodes in the search tree first."""
    queue = util.Queue()
    queue.push([state])
    seen = {state.getPacmanPosition()}
    while not queue.isEmpty():
        curr_state_list = queue.pop()
        curr_state = curr_state_list[-1]
        curr_pos = curr_state.getPacmanPosition()
        cost = len(curr_state_list)

        if curr_pos in state.getFood().asList():
            return cost

        for action in curr_state.getLegalActions():
            succ_state = curr_state.generatePacmanSuccessor(action)
            # cycle checking
            succ_pos = succ_state.getPacmanPosition()
            if succ_pos not in seen:
                seen.add(succ_pos)
                queue.push(curr_state_list + [succ_state])
        # print seen

    print "No solution found using BFS"
    return None

# Abbreviation
better = betterEvaluationFunction

