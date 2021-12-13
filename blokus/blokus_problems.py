from board import Board
from search import SearchProblem, ucs
import util
import numpy as np
import search

class BlokusFillProblem(SearchProblem):
    """
    A one-player Blokus game as a search problem.
    This problem is implemented for you. You should NOT change it!
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.expanded = 0

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        """
        state: Search state
        Returns True if and only if the state is a valid goal state
        """
        return not any(state.pieces[0])

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, 1) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        return len(actions)



#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################
class BlokusCornersProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.expanded = 0
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)


    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        "*** YOUR CODE HERE ***"
        w = self.board.board_w - 1
        h = self.board.board_h - 1
        first =state.get_position(0, 0) == 0
        second = state.get_position(w,h) == 0
        third = state.get_position(0,h) == 0
        four = state.get_position(w, 0) == 0

        return first and second and third and four


    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        "*** YOUR CODE HERE ***"
        c = 0
        for action in actions:
            c+=action.piece.get_num_tiles()
        return c


def find_corner(state,corners):
    #מחזיר true אם אי אפשר להגיע למטרה
    for corner in corners:

        #פותר בזמן מעולה אבל לא אדמסבילי
        # if state.get_position(corner[0],corner[1])!=0 and  not state.check_tile_attached(0,corner[0],corner[1]):
        #     # print(state)
        #     return True
        if state.get_position(corner[0],corner[1])!=0 and not state.check_tile_legal(0,corner[0],corner[1]):
            return True
    return False

def blokus_corners_heuristic(state, problem):
    """
    Your heuristic for the BlokusCornersProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come up
    with an admissible heuristic; almost all admissible heuristics will be consistent
    as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the other hand,
    inadmissible or inconsistent heuristics may find optimal solutions, so be careful.
    """
    "*** YOUR CODE HERE ***"

    if problem.is_goal_state(state):
        return 0
    left_idex_move = [(i,j)for i in range(state.board_w) for j in range(state.board_h) if state.check_tile_legal(0,i,j) and state.check_tile_attached(0, i, j)]
    a = np.array([piece.get_num_tiles() for piece in state.piece_list])
    left_pieces = np.array(a)[state.pieces[0]]
    corners = [(state.board_w-1,state.board_h-1),(state.board_w-1,0),(0,state.board_h-1),(0,0)]
    if len(left_idex_move)==0 or find_corner(state,corners) or len(left_pieces)==0:
        #בשני המקרים נגיע למצב שזה לא פתרון חוקי אזי אפשר להניח שאין לאיפה להשיך ונגרום לחיוש להמשיך למקום אחר
        return float('inf')
    # return 0
    return max(wghit(state),min(left_pieces))

def get_c(xy,xy1):
    #מרחק צבישב הוא אדמסבילי מאחר ובודקים מרחק באלכסון
    return max(abs(xy[0]-xy1[0]),abs(xy[1]-xy1[1]))
def get_lest_corner(state):
    corners = [(state.board_w - 1, state.board_h - 1), (state.board_w - 1, 0), (0, state.board_h - 1), (0, 0)]
    l = []
    for corner in corners:
        if state.get_position(corner[0], corner[1]) != 0:
            l.append(corner)
    return l
def wghit(state):
    lis = []
    l = get_lest_corner(state)
    a = state.get_legal_moves(0)
    if len(a)==0 and len(l)!=0:
        #לא נשארו מהלכים חוקיים והמטרה לא הושגה אז אין פתרון
        return float('inf')
    for i in a:
        for corner in l:
            lis.append(get_c([i.x, i.y], [corner[0],corner[1]]))
    c =min(lis)
    c= c/4-len(l)

    return c

class BlokusCoverProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=[(0, 0)]):
        self.targets = targets.copy()
        self.expanded = 0
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        "*** YOUR CODE HERE ***"

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        "*** YOUR CODE HERE ***"
        for t in self.targets:
            if state.state[t]!=0:
                return False
        return True


    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        "*** YOUR CODE HERE ***"
        cost = 0
        for action in actions:
            cost += action.piece.get_num_tiles()
        return cost



def get_lest_target(state,problem):
    l = []
    for t in problem.targets:
        if state.state[t] != 0:
            l.append(t)
    return l

def wghitForProblemB(state,problem):
    lis = []
    l = get_lest_target(state,problem)
    a = state.get_legal_moves(0)
    # print(len(l))
    if len(l) ==0:
        return 0
    if len(a) ==0:
        #לא נשארו מהלכים
        return float('inf')
    return 0


def blokus_cover_heuristic(state, problem):
    "*** YOUR CODE HERE ***"
    if problem.is_goal_state(state):
        return 0
    left_idex_move = [(i,j)for i in range(state.board_w) for j in range(state.board_h) if state.check_tile_legal(0,i,j) and state.check_tile_attached(0, i, j)]
    a = np.array([piece.get_num_tiles() for piece in state.piece_list])
    left_pieces = np.array(a)[state.pieces[0]]
    if len(left_idex_move)==0 or find_corner(state,problem.targets) or len(left_pieces)==0:
        #בשני המקרים נגיע למצב שזה לא פתרון חוקי אזי אפשר להניח שאין לאיפה להשיך ונגרום לחיוש להמשיך למקום אחר
        return float('inf')

    return max(wghitForProblemB(state,problem),min(left_pieces))



def solve(self):
    """
    This method should return a sequence of actions that covers all target locations on the board.
    This time we trade optimality for speed.
    Therefore, your agent should try and cover one target location at a time. Each time, aiming for the closest uncovered location.
    You may define helpful functions as you wish.

    Probably a good way to start, would be something like this --

    current_state = self.board.__copy__()
    backtrace = []

    while ....

        actions = set of actions that covers the closets uncovered target location
        add actions to backtrace

    return backtrace
    """
    "*** YOUR CODE HERE ***"
    current_state = self.board.__copy__()
    backtrace = []
    target = self.get_lest_target(current_state)
    queue = util.PriorityQueue()
    visited = set()
    cur_target = self.get_close_t(target)
    while not self.is_goal_state(current_state):
        # legal_point = [(i, j) for i in range(current_state.board_w) for j in range(current_state.board_h) if current_state.check_tile_legal(0, i, j) and current_state.check_tile_attached(0, i, j)]
        problem = BlokusCoverProblem(current_state.board_w, current_state.board_h, current_state.piece_list,
                                     self.starting_point, [cur_target])
        problem.board = current_state
        moves = search.serch_for_sub_prablom(visited.copy(), problem)
        if current_state not in visited:
            queue.push((current_state.__copy__(), backtrace.copy()), len(target))
        if moves == []:
            visited.add(current_state.__copy__())
            cur_state, backtrace = queue.pop()
            self.board = cur_state
            continue
        for move in moves:
            problem.board.add_move(0, move)
        target.remove(cur_target)
        cur_target = self.get_close_t(target, cur_target)
        self.expanded += problem.expanded
        self.board = problem.board
        current_state = self.board
        backtrace += moves

    return backtrace

class MiniContestSearch:
    """
    Implement your contest entry here
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=(0, 0)):
        self.targets = targets.copy()
        "*** YOUR CODE HERE ***"

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def solve(self):
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


class ClosestLocationSearch:
    """
    In this problem you have to cover all given positions on the board,
    but the objective is speed, not optimality.
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=(0, 0)):
        self.expanded = 0
        self.targets = targets.copy()
        "*** YOUR CODE HERE ***"
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.starting_point = starting_point
        self.original_targets = targets.copy()
        self.original_board = Board(board_w, board_h, 1, piece_list, starting_point)

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board
    def get_lest_target(self,state):
        l = []
        for t in self.targets:
            if state.state[t] != 0:
                l.append(t)
        return l
    def is_goal_state(self, state):
        "*** YOUR CODE HERE ***"
        for t in self.targets:
            if state.state[t]!=0:
                return False
        return True
    def get_close_t(self,target,t=0):
        if t == 0:
            return target[0]
        else:
            i = float('inf')
            return_t = None
            for targe in target:
                c = get_c(t,targe)
                if c<i:
                    i = c
                    return_t = targe
            return return_t



