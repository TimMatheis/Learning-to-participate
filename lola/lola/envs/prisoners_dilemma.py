"""
Iterated Prisoner's dilemma environment.
"""
import gym
import numpy as np

from gym.spaces import Discrete, Tuple

from .common import OneHot


class IteratedPrisonersDilemma(gym.Env):
    """
    A two-agent vectorized environment for the Prisoner's Dilemma game.
    Possible actions for each agent are (C)ooperate and (D)efect.
    """
    NAME = 'IPD'
    NUM_AGENTS = 2
    NUM_ACTIONS = 2
    NUM_STATES = 5 # 9 # 133 (12*10)+13 # 37 # 5 # 17    

    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.payout_mat = np.array([[-1., 0.], [-3., -2]])
        self.action_space = \
            Tuple([Discrete(self.NUM_ACTIONS), Discrete(self.NUM_ACTIONS)])
        self.observation_space = \
            Tuple([OneHot(self.NUM_STATES), OneHot(self.NUM_STATES)])

        self.step_count = None

        self.share_0 = 1.0     # added! ---------------------------------


    def reset(self):
        self.step_count = 0
        init_state = np.zeros(self.NUM_STATES)
        init_state[-1] = 1
        observations = [init_state, init_state]
        share_0 = 1.0      #--- added ----------------------
        return observations, share_0    #--- added:, share_0 ----------------------

    def step(self, action, share_0):   #--- added:, share_0
           
        self.step_count += 1

        state = np.zeros(self.NUM_STATES)

        # different implementations --> to change
        basic = False
        split = True
        share_decision = False
        share_decision_simple = False
        ten_perc_steps = False
        fifty_perc_steps = False


        if basic:
            ac0, ac1 = action
            rewards = [self.payout_mat[ac1][ac0], self.payout_mat[ac0][ac1]]
            state[ac0 * 2 + ac1] = 1


        if split:
            ac0, ac1 = action
            rewards = [ 0.5 * (self.payout_mat[ac1][ac0] + self.payout_mat[ac0][ac1]), 0.5 * (self.payout_mat[ac1][ac0] + self.payout_mat[ac0][ac1]) ]
            state[ac0 * 2 + ac1] = 1


        if share_decision:
        # NUM_STATES = 17
        # 4 actions per agent

            action_1, action_2 = action

            if action_1 == 0:
                ac0 = 0
                trade0 = 0
            elif action_1 == 1:
                ac0 = 1
                trade0 = 0
            elif action_1 == 2:
                ac0 = 0
                trade0 = 1
            elif action_1 == 3:
                ac0 = 1
                trade0 = 1
            else:
                print("ERROR")
                
            if action_2 == 0:
                ac1 = 0
                trade1 = 0
            elif action_2 == 1:
                ac1 = 1
                trade1 = 0
            elif action_2 == 2:
                ac1 = 0
                trade1 = 1
            elif action_2 == 3:
                ac1 = 1
                trade1 = 1
            else:
                print("ERROR")
            
            trading = False
                
            if (trade0 == 1) and (trade1 == 1):
                trading = True

            if trading == True:
                rewards = [0.5 * (self.payout_mat[ac1][ac0] + self.payout_mat[ac0][ac1]),
                    0.5 * (self.payout_mat[ac1][ac0] + self.payout_mat[ac0][ac1])]
            else:
                rewards = [self.payout_mat[ac1][ac0] , self.payout_mat[ac0][ac1] ]

            if (trade0 == 1) and (trade1 == 1):
                state[(ac0 * 2 + ac1)+12] = 1
            elif (trade0 == 1) and (trade1 == 0):
                state[(ac0 * 2 + ac1)+4] = 1
            elif (trade0 == 0) and (trade1 == 1):
                state[(ac0 * 2 + ac1)+8] = 1
            else:
                state[ac0 * 2 + ac1] = 1


        if share_decision_simple:
        # NUM_STATES = 9
        # actions per agent = 4

            action_1, action_2 = action

            if action_1 == 0:
                ac0 = 0
                trade0 = 0
            elif action_1 == 1:
                ac0 = 1
                trade0 = 0
            elif action_1 == 2:
                ac0 = 0
                trade0 = 1
            elif action_1 == 3:
                ac0 = 1
                trade0 = 1
            else:
                print("ERROR")
                
            if action_2 == 0:
                ac1 = 0
                trade1 = 0
            elif action_2 == 1:
                ac1 = 1
                trade1 = 0
            elif action_2 == 2:
                ac1 = 0
                trade1 = 1
            elif action_2 == 3:
                ac1 = 1
                trade1 = 1
            else:
                print("ERROR")
            
            trading = False
                
            if (trade0 == 1) and (trade1 == 1):
                trading = True

            if trading == True:
                rewards = [0.5 * (self.payout_mat[ac1][ac0] + self.payout_mat[ac0][ac1]),
                    0.5 * (self.payout_mat[ac1][ac0] + self.payout_mat[ac0][ac1])]
            else:
                rewards = [self.payout_mat[ac1][ac0] , self.payout_mat[ac0][ac1] ]

            if (trade0 == 1) and (trade1 == 1):
                state[(ac0 * 2 + ac1)+4] = 1
            else:
                state[ac0 * 2 + ac1] = 1


        if ten_perc_steps:
        # 6 actions per agent
        # 133 states
        # 0.1 steps

            action_1, action_2 = action
            print('share', share_0)
            print('action_1', action_1)
            
            # print(action)
            
            # choices: environment action, buy own share (and sell other share), buy other share (and sell own share)
            if action_1 == 0:
                ac0 = 0
                buy_own0 = 0
                buy_other0 = 0
            elif action_1 == 1:
                ac0 = 0
                buy_own0 = 1
                buy_other0 = 0
            elif action_1 == 2:
                ac0 = 0
                buy_own0 = 0
                buy_other0 = 1
            elif action_1 == 3:
                ac0 = 1
                buy_own0 = 0
                buy_other0 = 0
            elif action_1 == 4:
                ac0 = 1
                buy_own0 = 1
                buy_other0 = 0
            elif action_1 == 5:
                ac0 = 1
                buy_own0 = 0
                buy_other0 = 1
            else:
                print("ERROR")
                

            if action_2 == 0:
                ac1 = 0
                buy_own1 = 0
                buy_other1 = 0
            elif action_2 == 1:
                ac1 = 0
                buy_own1 = 1
                buy_other1 = 0
            elif action_2 == 2:
                ac1 = 0
                buy_own1 = 0
                buy_other1 = 1
            elif action_2 == 3:
                ac1 = 1
                buy_own1 = 0
                buy_other1 = 0
            elif action_2 == 4:
                ac1 = 1
                buy_own1 = 1
                buy_other1 = 0
            elif action_2 == 5:
                ac1 = 1
                buy_own1 = 0
                buy_other1 = 1
            else:
                print("ERROR")

            share_dict = {0.0: 0,
                        0.1: 12,
                        0.2: 24,
                        0.3: 36,
                        0.4: 48,
                        0.5: 60,
                        0.6: 72,
                        0.7: 84,
                        0.8: 96,
                        0.9: 108,
                        1.0: 120
                }

            if (buy_own0 == 1) and (buy_own1 == 1) and (share_0 < 1.0):
                share_0 += 0.1
                share_0 = round(share_0, 1)
                state[(ac0 * 2 + ac1) + 8 + share_dict[share_0]] = 1
            elif (buy_other0 == 1) and (buy_other1 == 1) and (share_0 >= 0.1):
                share_0 -= 0.1
                share_0 = round(share_0, 1)
                state[(ac0 * 2 + ac1) + 4 + share_dict[share_0]] = 1
            else:
                state[ac0 * 2 + ac1 + share_dict[share_0]] = 1

            rewards = [share_0 * self.payout_mat[ac1][ac0] + (1-share_0) * self.payout_mat[ac0][ac1],
        		(1-share_0) * self.payout_mat[ac1][ac0] + share_0 * self.payout_mat[ac0][ac1] ]
        

        if fifty_perc_steps:
        # 0.5 steps

            # choices: environment action, buy own share (and sell other share), buy other share (and sell own share)
            if action_1 == 0:
                ac0 = 0
                buy_own0 = 0
                buy_other0 = 0
            elif action_1 == 1:
                ac0 = 0
                buy_own0 = 1
                buy_other0 = 0
            elif action_1 == 2:
                ac0 = 0
                buy_own0 = 0
                buy_other0 = 1
            elif action_1 == 3:
                ac0 = 1
                buy_own0 = 0
                buy_other0 = 0
            elif action_1 == 4:
                ac0 = 1
                buy_own0 = 1
                buy_other0 = 0
            elif action_1 == 5:
                ac0 = 1
                buy_own0 = 0
                buy_other0 = 1
            else:
                print("ERROR")
                
            if action_2 == 0:
                ac1 = 0
                buy_own1 = 0
                buy_other1 = 0
            elif action_2 == 1:
                ac1 = 0
                buy_own1 = 1
                buy_other1 = 0
            elif action_2 == 2:
                ac1 = 0
                buy_own1 = 0
                buy_other1 = 1
            elif action_2 == 3:
                ac1 = 1
                buy_own1 = 0
                buy_other1 = 0
            elif action_2 == 4:
                ac1 = 1
                buy_own1 = 1
                buy_other1 = 0
            elif action_2 == 5:
                ac1 = 1
                buy_own1 = 0
                buy_other1 = 1
            else:
                print("ERROR")

            share_dict = {0.0: 0,
                        0.5: 12,
                        1.0: 24
                }

            if (buy_own0 == 1) and (buy_own1 == 1) and (share_0 <= 0.5):
                share_0 += 0.5
                share_0 = round(share_0, 1)
                state[(ac0 * 2 + ac1) + 8 + share_dict[share_0]] = 1
            elif (buy_other0 == 1) and (buy_other1 == 1) and (share_0 >= 0.5):
                share_0 -= 0.5
                share_0 = round(share_0, 1)
                state[(ac0 * 2 + ac1) + 4 + share_dict[share_0]] = 1
            else:
                state[ac0 * 2 + ac1 + share_dict[share_0]] = 1

            rewards = [share_0 * self.payout_mat[ac1][ac0] + (1-share_0) * self.payout_mat[ac0][ac1],
        		(1-share_0) * self.payout_mat[ac1][ac0] + share_0 * self.payout_mat[ac0][ac1] ]
        


        observations = [state, state]


        # rewards = [self.payout_mat[ac1][ac0] , self.payout_mat[ac0][ac1] ]
        # rewards = [share_0 * self.payout_mat[ac1][ac0] + (1-share_0) * self.payout_mat[ac0][ac1],
        #		(1-share_0) * self.payout_mat[ac1][ac0] + share_0 * self.payout_mat[ac0][ac1] ]


        done = (self.step_count == self.max_steps)

        return observations, rewards, done, share_0    #--- added:, share_0
