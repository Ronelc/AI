import numpy as np
import abc
import util
from game import Agent, Action


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """

        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def get_sink(self, borde):
        borde = borde.board

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (GameState.py) and returns a number, where higher numbers are better.

        """

        # Useful information you can extract from a GameState (game_state.py)

        successor_game_state = current_game_state.generate_successor(action=action)
        board = successor_game_state.board
        max_tile = successor_game_state.max_tile
        score = successor_game_state.score

        "*** YOUR CODE HERE ***"
        # check the monoton, mean that the agent do move to the left or right
        diff = np.diff(board ** 2, axis=1)
        monoton_right = np.sum(diff[diff > 0])
        monoton_left = np.sum(-1 * diff[diff <= 0])
        monotone_modifier = min(monoton_right, monoton_left)
        nom_snik = 0
        diff3 = board.shape
        # print(diff3)
        # indx = [([3,0],[3,1]),([0,1],[0,2]),([3,2],[3,3])]
        #
        # for ind in indx:
        #     nom_snik += board[ind[0][0], ind[0][1]] - board[ind[1][0], ind[1][1]]
        return score - monotone_modifier + nom_snik + len(successor_game_state.get_empty_tiles())


def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.score


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        self.evaluation_function = util.lookup(evaluation_function, globals())
        self.depth = depth

    @abc.abstractmethod
    def get_action(self, game_state):
        return


class MinmaxAgent(MultiAgentSearchAgent):
    def get_action(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1

        Action.STOP:
            The stop direction, which is always legal

        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """
        """*** YOUR CODE HERE ***"""

        player = 0
        actions = game_state.get_legal_actions(player)

        # get oll succurors. That is, all the moves the computer can make
        successors = np.array(
            [self.Minimax(game_state.generate_successor(player, action), 0, False) for action in actions])
        #return the max
        return actions[np.argmax(successors)]

    def Minimax(self, node, depth, isMaxNode):
        """
        :param node: game state
        :param depth: How deep to look in tree
        :param isMaxNode: true if the agent is max else false
        :return: the best move for maxPlayer
        """

        # if we got to depth that we chose in constrctor
        if depth == self.depth:
            return self.evaluation_function(node)
        elif isMaxNode:
            successors = node.get_legal_actions(0)
            max = np.max(
                np.array([self.Minimax(node.generate_successor(0, s), depth, False) for s in successors])) if len(
                successors) else 0
            # return the best move that maximiz the point
            return max
        else:
            successors = node.get_legal_actions(1)
            # return the best move that mimaiz
            return np.min(np.array([self.Minimax(node.generate_successor(1, s), depth + 1, True) for s in successors]))


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        """*** YOUR CODE HERE ***"""
        actions = game_state.get_legal_actions(0)
        # Initializes with the maximum values for alpha, beta
        alpha, beta = -float('inf'), float('inf')
        best = actions[0]
        for action in actions:
            succ = game_state.generate_successor(0, action)
            n_alpha = self.AlphaBetaPruning(succ, depthe=0, alpha=alpha, beta=beta, maxPlayer=False)
            if n_alpha > alpha:
                best = action
                alpha = n_alpha
            if alpha >= beta:
                break
        return best

    def AlphaBetaPruning(self, node, depthe, alpha, beta, maxPlayer):
        """
        :param node: the game state
        :param depthe: How deep to look in tree
        :param alpha: the value of alpha plyer
        :param beta: the value of beta plyer
        :param maxPlayer: true if plyer is maxPlayer
        :return: alpha if case of maxPlayer is true and beta other
        """
        if depthe == self.depth:
            return self.evaluation_function(node)
        if maxPlayer:
            # get oll legal action for the agent
            secrets = node.get_legal_actions(0)
            for secus in secrets:
                alpha = max(alpha, self.AlphaBetaPruning(node.generate_successor(0, secus), depthe, alpha, beta, False))
                if alpha >= beta:
                    break
            return alpha
        else:
            # get oll legal action for the coputer
            secrets = node.get_legal_actions(1)
            for secus in secrets:
                # get the beat move that mzimiz the score of maxPlyer
                beta = min(beta,
                           self.AlphaBetaPruning(node.generate_successor(1, secus), depthe + 1, alpha, beta, True))
                if alpha >= beta:
                    break
            return beta


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        """*** YOUR CODE HERE ***"""
        player = 0
        actions = game_state.get_legal_actions(player)

        # get oll succurors. That is, all the moves the computer can make
        successors = np.array(
            [self.expectimaxAgent(game_state.generate_successor(player, action), 0, False) for action in actions])
        # return the max expect
        return actions[np.argmax(successors)]

    def expectimaxAgent(self, node, depth, isMaxNode):
        if depth == self.depth:
            return self.evaluation_function(node)
        elif isMaxNode:
            successors = node.get_legal_actions(0)
            max = np.max(
                np.array([self.expectimaxAgent(node.generate_successor(0, s), depth, False) for s in successors])) if len(
                successors) else 0
            # return the best move that maximiz the point
            return max
        else:
            successors = node.get_legal_actions(1)
            # return the best move that mean
            return np.mean(np.array([self.expectimaxAgent(node.generate_successor(1, s), depth + 1, True) for s in successors]))


def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    # successor_game_state = current_game_state.generate_successor(action=action)
    # board = current_game_state._board
    # print(board)
    # print(np.where(board!=0))
    # print(current_game_state.get_empty_tiles())
    # # max_tile = successor_game_state.max_tile
    # # score = successor_game_state.score
    # "*** YOUR CODE HERE ***"
    # n_1 = 128
    #
    # n_2 = len(current_game_state.get_empty_tiles()[0])*128
    # pices_in_borde = np.where(board!=0)
    # n= 0
    # max_high= board.shape
    # print(max_high)
    # for ind in pices_in_borde[0]:
    #     x,y = pices_in_borde[0],pices_in_borde[1]
    #     if x==0:
    #         if y>0 and y<max_high[1]-1:
    #             if not helper1(board,x,y):
    #
    #
    # n_4 = len(current_game_state.get_agent_legal_actions())*256
    # util.raiseNotDefined()

    board = current_game_state.board
    # max_tile = successor_game_state.max_tile
    score = current_game_state.score

    "*** YOUR CODE HERE ***"
    pusible_move = len(current_game_state.get_agent_legal_actions())*256
    #check the monoton, mean that the agent do move to the left or right
    diff_l_r = np.diff(board ** 2, axis=1)
    monoton_right = np.sum(diff_l_r[diff_l_r > 0])
    monoton_left = np.sum(-1 * diff_l_r[diff_l_r <= 0])
    # #check the smuthnes, mean that the agent do move to the up
    # diif_u_d = np.diff(board ** 2, axis=0)
    # monoton_up = np.sum(diif_u_d[diif_u_d > 0])
    # monoton_down = np.sum(-1*diif_u_d[diif_u_d < 0])
    #check if the max title in the right down corner
    max = current_game_state.max_tile
    corn = [board[-1][-1],board[-1][0]]
    score_for_max_in_corner = max*1000 if max in corn else -max*1000
    return score - min(monoton_right, monoton_left)+score_for_max_in_corner#-min(monoton_down,monoton_up)


def manhattanDistance(xy1, xy2):
    "Returns the Manhattan distance between points xy1 and xy2"
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])
# def helper1(borad,x,y):
#     return borad[x+1,y]==0 and borad[x,y+1]==0 and borad[x,y-1]==0
# Abbreviation
better = better_evaluation_function
