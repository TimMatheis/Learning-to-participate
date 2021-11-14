import numpy as np


class Actor(object):

    def __init__(self, agent_id, n_agents, l_obs):

        self.agent_id = agent_id
        self.l_obs = l_obs
        self.n_agents = n_agents
        self.position = 9   # was 1
        self.total_given = np.zeros(self.n_agents - 1)

    def act(self, action, reward_given, share_0, observe_given=False): # added: , share_0 ; observe_given=True
        # position: 3 (env actions) * 3 (shares)
        share_dict = {0.0: 0,
                0.25: 1,
                0.50: 2,
                0.75: 3,
                1.0: 4,
            }

        env_position = {
                0: 0,
                1: 0,
                2: 0,
                3: 1,
                4: 1,
                5: 1,
                6: 2,
                7: 2,
                8: 2
                }

        self.position = 5 * env_position[action] + share_dict[share_0] # was "action" before
        if observe_given:
            self.total_given += reward_given

    def get_obs(self, state, observe_given=False): # observe_given=True
        obs = np.zeros(self.l_obs)
        # position of self
        obs[state[self.agent_id]] = 1
        list_others = list(range(0, self.n_agents))
        del list_others[self.agent_id]
        # positions of other agents
        for idx, other_id in enumerate(list_others):
            obs[9*(idx + 1) + state[other_id]] = 1 #------------------------ CHANGED 3 to 9! -----------------

        # total amount given to other agents
        if observe_given:
            obs[-(self.n_agents - 1):] = self.total_given

        return obs

    def reset(self, randomize=False): # changed: was randomize=False -------------------
        if randomize:
            self.position = np.random.randint(9) # chnaged: was randint(3) -----------------
        else:
            self.position = 9 # was: 1
        self.total_given = np.zeros(self.n_agents - 1)
