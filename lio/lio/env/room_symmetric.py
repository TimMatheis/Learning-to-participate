import numpy as np
from lio.env import room_agent


class Env(object):

    def __init__(self, config_env):

        self.config = config_env

        self.n_agents = self.config.n_agents
        self.name = 'er'
        self.l_action = 9 # 3 ----------------------------------------------------------------
        # Observe self position (1-hot),
        # other agents' positions (1-hot for each other agent)
        # total amount given to each other agent
        # self.l_obs = 3 + 3*(self.n_agents - 1) + (self.n_agents - 1)

        self.l_obs = 15 + 15*(self.n_agents - 1) # new, changed line above ---------------------

        self.max_steps = self.config.max_steps
        self.min_at_lever = self.config.min_at_lever
        self.randomize = self.config.randomize

        self.actors = [room_agent.Actor(idx, self.n_agents, self.l_obs)
                       for idx in range(self.n_agents)]

        self.share_0 = 1.0     # added! ---------------------------------

    def get_door_status(self, actions):
        n_going_to_lever = actions.count(0) +actions.count(1) + actions.count(2) # was: actions.count(0)
        return n_going_to_lever >= self.min_at_lever

    def calc_reward(self, actions, given_rewards, door_open, share_0):  # added: , share_0 ----------
        assert len(actions) == self.n_agents
        rewards = np.zeros(self.n_agents)

        if self.config.reward_sanity_check:
            rewards[0] = 10 if actions[0] == 1 else -1
            rewards[1] = 2 if actions[1] == 0 else -1

        # ------------------------ added ------------------------------------------------------------
        else:
            # choices: environment action, buy own share (and sell other share), buy other share (and sell own share)
            if actions[0] == 0:
                ac0 = 0
                buy_own0 = 0
                buy_other0 = 0
            elif actions[0] == 1:
                ac0 = 0
                buy_own0 = 1
                buy_other0 = 0
            elif actions[0] == 2:
                ac0 = 0
                buy_own0 = 0
                buy_other0 = 1
            elif actions[0] == 3:
                ac0 = 1
                buy_own0 = 0
                buy_other0 = 0
            elif actions[0] == 4:
                ac0 = 1
                buy_own0 = 1
                buy_other0 = 0
            elif actions[0] == 5:
                ac0 = 1
                buy_own0 = 0
                buy_other0 = 1
            elif actions[0] == 6:
                ac0 = 2
                buy_own0 = 0
                buy_other0 = 0
            elif actions[0] == 7:
                ac0 = 2
                buy_own0 = 1
                buy_other0 = 0
            elif actions[0] == 8:
                ac0 = 2
                buy_own0 = 0
                buy_other0 = 1
            else:
                print("ERROR")
                

            if actions[1] == 0:
                ac1 = 0
                buy_own1 = 0
                buy_other1 = 0
            elif actions[1] == 1:
                ac1 = 0
                buy_own1 = 1
                buy_other1 = 0
            elif actions[1] == 2:
                ac1 = 0
                buy_own1 = 0
                buy_other1 = 1
            elif actions[1] == 3:
                ac1 = 1
                buy_own1 = 0
                buy_other1 = 0
            elif actions[1] == 4:
                ac1 = 1
                buy_own1 = 1
                buy_other1 = 0
            elif actions[1] == 5:
                ac1 = 1
                buy_own1 = 0
                buy_other1 = 1
            elif actions[1] == 6:
                ac0 = 2
                buy_own0 = 0
                buy_other0 = 0
            elif actions[1] == 7:
                ac0 = 2
                buy_own1 = 1
                buy_other1 = 0
            elif actions[1] == 8:
                ac0 = 2
                buy_own1 = 0
                buy_other1 = 1
            else:
                print("ERROR")


            if (buy_own0 == 1) and (buy_own1 == 1) and (share_0 <= 0.75):
                share_0 += 0.25
                share_0 = round(share_0, 2)
                # state[(ac0 * 2 + ac1) + 8 + share_dict[share_0]] = 1
            elif (buy_other0 == 1) and (buy_other1 == 1) and (share_0 >= 0.25):
                share_0 -= 0.25
                share_0 = round(share_0, 2)
                # state[(ac0 * 2 + ac1) + 4 + share_dict[share_0]] = 1
            else:
                pass
                # state[ac0 * 2 + ac1 + share_dict[share_0]] = 1

            env_position = {
                0: 0,
                1: 0,
                2: 0,
                3: 1,
                4: 1,
                5: 1,
                6: 2,
                7: 2,
                8: 2,
                }

            prev_position = {
                0: 0,
                1: 0,
                2: 0,
                3: 0,
                4: 0,
                5: 1,
                6: 1,
                7: 1,
                8: 1,
                9: 1,
                10: 2,
                11: 2,
                12: 2,
                13: 2,
                14: 2
                }


            for agent_id in range(0, self.n_agents):

                #print('agent', agent_id, 'env pos action', env_position[actions[agent_id]])
                #print('agent', agent_id, 'env pos previous', env_position[self.actors[agent_id].position])
                if door_open and (actions[agent_id] == 6 or actions[agent_id] == 7 or actions[agent_id] == 8):
                    rewards[agent_id] = 10
                elif env_position[actions[agent_id]] == prev_position[self.actors[agent_id].position]:
                    # no penalty for staying at current position
                    rewards[agent_id] = 0
                else:
                    rewards[agent_id] = -1

            reward_0 = rewards[0]
            reward_1 = rewards[1]
            rewards[0] = share_0*reward_0 + (1-share_0)*reward_1
            rewards[1] = share_0*reward_1 + (1-share_0)*reward_0
        # --------------------------------------------------------------------------------------------
        """ OLD version:
        else:
            for agent_id in range(0, self.n_agents):
                if door_open and actions[agent_id] == 2:
                    rewards[agent_id] = 10
                elif actions[agent_id] == self.actors[agent_id].position:
                    # no penalty for staying at current position
                    rewards[agent_id] = 0
                else:
                    rewards[agent_id] = -1
        """

        return rewards, share_0 # added , share_0 -----------------------------------------------------

    def get_obs(self):
        list_obs = []
        for actor in self.actors:
            list_obs.append(actor.get_obs(self.state))

        return list_obs

    def step(self, actions, given_rewards, share_0):    # added: , share_0

        door_open = self.get_door_status(actions)
        rewards, share_0 = self.calc_reward(actions, given_rewards, door_open, share_0) # added: , share_0 (2 times)
        for idx, actor in enumerate(self.actors):
            actor.act(actions[idx], given_rewards[idx], share_0) # added , share_0
        self.steps += 1
        self.state = [actor.position for actor in self.actors]
        list_obs_next = self.get_obs()

        # Terminate if (door is open and some agent ended up at door)
        # or reach max_steps
        # done = (door_open and (6 in self.state or 7 in self.state or 8 in self.state)) or self.steps == self.max_steps
        done = (door_open and (6 in actions or 7 in actions or 8 in actions)) or self.steps == self.max_steps

        return list_obs_next, rewards, done, share_0 # added: ',share_0' ---------------------------

    def reset(self):
        for actor in self.actors:
            actor.reset(self.randomize)
        self.state = [actor.position for actor in self.actors]
        self.steps = 0
        list_obs = self.get_obs()
        share_0 = 1 # added ----------------------------------------------------

        return list_obs, share_0 # added: ',share_0' ---------------------------
