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
    NUM_ACTIONS = 2 # 9 # 6 # 5
    NUM_STATES = 5 # 133 # 133 # (4*3+1)+3 # 5

    # NUM_STATES_TRADE = 11

    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.payout_mat = np.array([[-1., 0.], [-3., -2.]])
        self.action_space = \
            Tuple([Discrete(self.NUM_ACTIONS), Discrete(self.NUM_ACTIONS)])
        self.observation_space = \
            Tuple([OneHot(self.NUM_STATES), OneHot(self.NUM_STATES)])

        self.step_count = None

    def reset(self):
        self.step_count = 0
        init_state = np.zeros(self.NUM_STATES)
        init_state[-1] = 1
        observations = [init_state, init_state]
        share_0 = 1.0
        return observations, share_0

    
    def trade_step(self, action):
            # print("action:", action)

        # self.step_count += 1
        state = np.zeros(self.NUM_STATES)

        # TO CHANGE
        basic = True
        ten_percent = False

        if basic: 
            ac0, ac1 = action
            rewards = [self.payout_mat[ac1][ac0], self.payout_mat[ac0][ac1]]
            # 50/50
            # rewards = [0.5 * (self.payout_mat[ac1][ac0] + self.payout_mat[ac0][ac1]), 0.5 * (self.payout_mat[ac0][ac1] + self.payout_mat[ac1][ac0]) ]
            state[ac0 * 2 + ac1] = 1

            share_0 = 1


        if ten_percent:

            action_1, action_2 = action

            # print(action)
            assert 5 < action_1 < 16 
            assert 5 < action_2 < 16 

            
            # print('share', share_0)
            # print('action_1', action_1)
            
            # print(action)
            
            # choices: environment action, buy own share (and sell other share), buy other share (and sell own share)
            if action_1 == 6:
                ac0 = 0
                tr0 = 0.0
            elif action_1 == 7:
                ac0 = 0
                tr0 = 0.2
            elif action_1 == 8:
                ac0 = 0
                tr0 = 0.5
            elif action_1 == 9:
                ac0 = 0
                tr0 = 0.7
            elif action_1 == 10:
                ac0 = 0
                tr0 = 1.0
            elif action_1 == 11:
                ac0 = 1
                tr0 = 0
            elif action_1 == 12:
                ac0 = 1
                tr0 = 0.2
            elif action_1 == 13:
                ac0 = 1
                tr0 = 0.5
            elif action_1 == 14:
                ac0 = 1
                tr0 = 0.7
            elif action_1 == 15:
                ac0 = 1
                tr0 = 1.0
            else:
                print("ERROR")
                

            if action_2 == 6:
                ac1 = 0
                tr1 = 0.0
            elif action_2 == 7:
                ac1 = 0
                tr1 = 0.2
            elif action_2 == 8:
                ac1 = 0
                tr1 = 0.5
            elif action_2 == 9:
                ac1 = 0
                tr1 = 0.7
            elif action_2 == 10:
                ac1 = 0
                tr1 = 1.0
            elif action_2 == 11:
                ac1 = 1
                tr1 = 0
            elif action_2 == 12:
                ac1 = 1
                tr1 = 0.2
            elif action_2 == 13:
                ac1 = 1
                tr1 = 0.5
            elif action_2 == 14:
                ac1 = 1
                tr1 = 0.7
            elif action_2 == 15:
                ac1 = 1
                tr1 = 1.0
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

            share_0 = max(tr0, tr1)

            state[ac0 * 2 + ac1 + share_dict[share_0]] = 1

            #
            # -----------------------------------------------
            # share_0 = 0.5
            # -----------------------------------------------
            #

            rewards = [share_0 * self.payout_mat[ac1][ac0] + (1-share_0) * self.payout_mat[ac0][ac1],
                (1-share_0) * self.payout_mat[ac1][ac0] + share_0 * self.payout_mat[ac0][ac1] ]

        observations = [state, state]

        return observations, share_0, rewards


    """
    def trade_step(self, action, share_0):
            # print("action:", action)

        # self.step_count += 1
        state = np.zeros(self.NUM_STATES_TRADE + 133)

        ac0, ac1 = action

        print(action)
        print(share_0)
        assert 5 < ac0 < 9 
        assert 5 < ac1 < 9 

        if (ac0==6 and ac1==6 and share_0 <= 0.9):
            share_0 += 0.1
            share_0 = round(share_0, 1)
        elif (ac0==7 and ac1==7 and share_0 >= 0.1):
            share_0 -= 0.1
            share_0 = round(share_0, 1)

        st = int(share_0*10 + 133)
        state[st] = 1

        observations = [state, state]

        rewards = [0, 0]

        return observations, share_0, rewards
    """


    def step(self, action, share_0):
        # print("action:", action)

        self.step_count += 1
        state = np.zeros(self.NUM_STATES)
        
        # TO CHANGE
        basic = True
        ten_percent = False

        if basic: 
            ac0, ac1 = action
            rewards = [self.payout_mat[ac1][ac0], self.payout_mat[ac0][ac1]]
            # 50/50
            # rewards = [0.5 * (self.payout_mat[ac1][ac0] + self.payout_mat[ac0][ac1]), 0.5 * (self.payout_mat[ac0][ac1] + self.payout_mat[ac1][ac0]) ]
            state[ac0 * 2 + ac1] = 1


        if ten_percent:
            action_1, action_2 = action
            # print('share', share_0)
            # print('action_1', action_1)
            
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

        observations = [state, state]

        done = (self.step_count == self.max_steps)

        return observations, rewards, done, share_0
