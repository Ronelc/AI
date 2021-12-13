# # valueIterationAgents.py
# # -----------------------
# # Licensing Information: Please do not distribute or publish solutions to this
# # project. You are free to use and extend these projects for educational
# # purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# # John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# # For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html
#
# import mdp, util
# # import numpy as np
#
#
# from learningAgents import ValueEstimationAgent
#
#
# class ValueIterationAgent(ValueEstimationAgent):
#     """
#         * Please read learningAgents.py before reading this.*
#
#         A ValueIterationAgent takes a Markov decision process
#         (see mdp.py) on initialization and runs value iteration
#         for a given number of iterations using the supplied
#         discount factor.
#     """
#
#     def __init__(self, mdp, discount=0.9, iterations=100):
#         """
#           Your value iteration agent should take an mdp on
#           construction, run the indicated number of iterations
#           and then act according to the resulting policy.
#
#           Some useful mdp methods you will use:
#               mdp.getStates()
#               mdp.getPossibleActions(state)
#               mdp.getTransitionStatesAndProbs(state, action)
#               mdp.getReward(state, action, nextState)
#         """
#         self.mdp = mdp
#         self.discount = discount
#         self.iterations = iterations
#         self.values = util.Counter()  # A Counter is a dict with default 0
#         self.policys = util.Counter()
#
#         "*** YOUR CODE HERE ***"
#
#         for iter in range(iterations):
#             new_vals = util.Counter()
#             new_policies = util.Counter()
#             states = self.mdp.getStates()
#
#             for state in states:
#                 if self.mdp.isTerminal(state):
#                     new_vals[state] = 0
#                     new_policies[state] = None
#                     continue
#
#                 actions = self.mdp.getPossibleActions(state)
#                 if not actions:
#                     new_vals[state] = 0
#                     new_policies[state] = None
#                     continue
#
#                 best_action = None
#                 best_action_score = -1000
#                 for cur_action in actions:
#                     cur_action_score = self.getQValue(state, cur_action)
#                     if cur_action_score > best_action_score:
#                         best_action = cur_action
#                         best_action_score = cur_action_score
#
#                 new_vals[state] = best_action_score
#                 new_policies[state] = best_action
#
#             self.values = new_vals
#             self.policies = new_policies
#
#     def getValue(self, state):
#         """
#           Return the value of the state (computed in __init__).
#         """
#         return self.values[state]
#
#     def getQValue(self, state, action):
#         """
#           The q-value of the state action pair
#           (after the indicated number of value iteration
#           passes).  Note that value iteration does not
#           necessarily create this quantity and you may have
#           to derive it on the fly.
#         """
#         "*** YOUR CODE HERE ***"
#
#         q_val = 0
#         transitions = self.mdp.getTransitionStatesAndProbs(state, action)
#         if transitions:
#             R = self.mdp.getReward(state, action, None)
#             q_val += R
#             # TODO make sure that we're consistent with everyone else.
#             for s_tag, prob in transitions:
#                 q_val += prob * self.discount * self.getValue(s_tag)
#
#         return q_val
#         # self.mdp = mdp
#         # self.discount = discount
#         # self.iterations = iterations
#         # self.values = util.Counter()  # A Counter is a dict with default 0
#         # self.policys = util.Counter()
#         # "*** YOUR CODE HERE ***"
#         # stats = self.mdp.getStates()
#         # self.h()
#     #     return
#     #     self._helper_value()
#     #     for i in range(self.iterations):
#     #         for s in stats:
#     #             if self.mdp.isTerminal(s):
#     #                 self.values[s] = 0
#     #                 self.policys[s] = None
#     #                 continue
#     #             action = mdp.getPossibleActions(s)
#     #             num = self.mdp.getReward(s, s, s)
#     #             for act in action:
#     #                 a = self.mdp.getTransitionStatesAndProbs(s, act)
#     #                 for a1 in a:
#     #                     if a1[0] != s:
#     #                         num += a1[1] * self.values[a1[0]]
#     #             self.values[s] = self.discount * num
#     #             self.helper_pol(action,s)
#     # def _helper_value(self):
#     #     stats = self.mdp.getStates()
#     #     # sum= 0
#     #     h_v =  util.Counter()
#     #     for s in stats:
#     #         l = []
#     #         action = self.mdp.getPossibleActions(s)
#     #         for act in action:
#     #
#     #             next_moves = self.mdp.getTransitionStatesAndProbs(s, act)
#     #             print(next_moves)
#     #             sum = 0
#     #
#     #             for move in next_moves:
#     #                 sum+= move[1]*(self.mdp.getReward(s,act ,move[0] )+self.discount*self.values[move[0]])
#     #             l.append((act,sum))
#     #         print(l,s)
#     #         h_v[s] =0.0 if len(l)== 0 else max(l,key=lambda x:x[1])[1]
#     #     print(h_v)
#     #     self.values = h_v
#     #
#     #
#     # def h(self):
#     #     states = self.mdp.getStates()
#     #     val = util.Counter()
#     #     for i in range(self.iterations):
#     #         h_v = util.Counter()
#     #         for s in states:
#     #             l = []
#     #             action = self.mdp.getPossibleActions(s)
#     #             l =[self.getQValue(s,act) for act in action]
#     #
#     #             max_l =0 if len(l) == 0 else self.values[s]+(max(l)*self.discount)+self.mdp.getReward
#     #             h_v[s] =
#     #         val = h_v
#     #     self.values = val
#     #     print(val)
#     #
#     #
#     # def helper_pol(self, action, state):
#     #     l = []
#     #     n = 0
#     #     for act in action:
#     #         a = self.mdp.getTransitionStatesAndProbs(state, act)
#     #         for a1 in a:
#     #             n += a1[1] * self.values[a1[0]]
#     #         l.append((act, n))
#     #     max_l = max(l, key=lambda x: x[1])
#     #     self.policys[state] = max_l[0]
#     #
#     # def getValue(self, state):
#     #     """
#     #       Return the value of the state (computed in __init__).
#     #     """
#     #     return self.values[state]
#     #
#     # def getQValue(self, state, action):
#     #     """
#     #       The q-value of the state action pair
#     #       (after the indicated number of value iteration
#     #       passes).  Note that value iteration does not
#     #       necessarily create this quantity and you may have
#     #       to derive it on the fly.
#     #     """
#     #     "*** YOUR CODE HERE ***"
#     #     T = self.mdp.getTransitionStatesAndProbs(state, action)
#     #     q_sum = 0
#     #     if len(T)!=0:
#     #         R = self.mdp.getReward(state, action, None)
#     #         q_sum += R
#     #         for s_tag, prob in T:
#     #             q_sum += prob * self.discount * self.getValue(s_tag)
#     #     return q_sum
#     #     util.raiseNotDefined()
#
#     def getPolicy(self, state):
#         """
#           The policy is the best action in the given state
#           according to the values computed by value iteration.
#           You may break ties any way you see fit.  Note that if
#           there are no legal actions, which is the case at the
#           terminal state, you should return None.
#         """
#         "*** YOUR CODE HERE ***"
#         return self.policys[state]
#         util.raiseNotDefined()
#
#     def getAction(self, state):
#         "Returns the policy at the state (no exploration)."
#         return self.getPolicy(state)
# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util
# import numpy as np

from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.policies = util.Counter()

        "*** YOUR CODE HERE ***"

        for iter in range(iterations):
            new_vals = util.Counter()
            new_policies = util.Counter()
            states = self.mdp.getStates()

            for state in states:
                if self.mdp.isTerminal(state):
                    new_vals[state] = 0
                    new_policies[state] = None
                    continue

                actions = self.mdp.getPossibleActions(state)
                if not actions:
                    new_vals[state] = 0
                    new_policies[state] = None
                    continue
                best_action = None
                best_action_score = float('-inf')
                R = 0
                R += self.mdp.getReward(state, None, None)
                for act in actions:
                    cur_action_score = self.get(state,act)
                    if cur_action_score > best_action_score:
                        best_action = act
                        best_action_score = cur_action_score
                new_vals[state] = R+(best_action_score * self.discount)
                new_policies[state] = best_action
            self.values = new_vals
            self.policies = new_policies

    def get(self, state, action):
        """
          The q-value of the state action pair
          (after the indicated number of value iteration
          passes).  Note that value iteration does not
          necessarily create this quantity and you may have
          to derive it on the fly.
        """
        "*** YOUR CODE HERE ***"

        q_val = 0
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        if transitions:
            for s_t, prob in transitions:
                q_val += prob * self.getValue(s_t)
        return q_val
    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def getQValue(self, state, action):
        """
          The q-value of the state action pair
          (after the indicated number of value iteration
          passes).  Note that value iteration does not
          necessarily create this quantity and you may have
          to derive it on the fly.
        """
        "*** YOUR CODE HERE ***"

        q_val = 0
        T = self.mdp.getTransitionStatesAndProbs(state,action)
        if T:
            R = self.mdp.getReward(state,action,None)
            q_val += R
            for s_t, prob in T:
                q_val +=  prob  *  self.getValue(s_t)* self.discount

        return q_val



    def getPolicy(self, state):
        """
          The policy is the best action in the given state
          according to the values computed by value iteration.
          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        return self.policies[state]

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.getPolicy(state)
