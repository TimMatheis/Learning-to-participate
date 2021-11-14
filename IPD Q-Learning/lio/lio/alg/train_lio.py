"""Trains LIO agents on Escape Room game.

Three versions of LIO:
1. LIO built on top of policy gradient
2. LIO built on top of actor-critic
3. Fully decentralized version of LIO on top of policy gradient
"""

from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import random

import numpy as np

import sys
sys.path.append(r"C:\Users\timma\OneDrive - Handelshögskolan i Stockholm\Informatik\Bachelorarbeit\q_table\lio")

from lio.alg import config_ipd_lio
from lio.alg import evaluate
from lio.env import ipd_wrapper


def train(config):

    seed = config.main.seed
    np.random.seed(seed)
    random.seed(seed)

    dir_name = config.main.dir_name
    # exp_name = config.main.exp_name
    log_path = os.path.join('..', 'results', dir_name)
    model_name = config.main.model_name
    save_period = config.main.save_period

    os.makedirs(log_path, exist_ok=True)

    # Keep a record of parameters used for this run
    with open(os.path.join(log_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4, sort_keys=True)

    n_episodes = int(config.alg.n_episodes)
    n_eval = config.alg.n_eval
    period = config.alg.period

    epsilon = config.lio.epsilon_start
    epsilon_step = (
        epsilon - config.lio.epsilon_end) / config.lio.epsilon_div

    if config.env.name == 'ipd':
        env = ipd_wrapper.IPD(config.env)

    
    from lio_agent import LIO


    list_agents = []
    # for agent_id in range(env.n_agents):
    for agent_id in range(config.env.n_agents):
        list_agents.append(LIO(config.lio, env.l_obs, env.l_action,
                                   config.nn, 'agent_%d' % agent_id,
                                   config.env.r_multiplier, config.env.n_agents,
                                   agent_id))        

    for agent_id in range(config.env.n_agents):
        list_agents[agent_id].receive_list_of_agents(list_agents)

    list_agent_meas = []
    if config.env.name == 'er':
        list_suffix = ['reward_total', 'n_lever', 'n_door',
                       'received', 'given', 'r-lever', 'r-start', 'r-door']
    elif config.env.name == 'ipd':
        list_suffix = ['reward_env', 'share']
    for agent_id in range(1, config.env.n_agents + 1):
        for suffix in list_suffix:
            list_agent_meas.append('A%d_%s' % (agent_id, suffix))

    header = 'episode,step_train,step,'
    header += ','.join(list_agent_meas)
    if config.env.name == 'er':
        header += ',steps_per_eps\n'
    else:
        header += '\n'
    with open(os.path.join(log_path, 'log.csv'), 'w') as f:
        f.write(header)    

    step = 0
    step_train = 0
    for idx_episode in range(1, n_episodes + 1):

        list_buffers = run_episode(env, list_agents, epsilon,
                                   prime=False)
        step += len(list_buffers[0].obs)

        """
        for idx, agent in enumerate(list_agents):
            agent.update(list_buffers[idx], epsilon)
       

        list_buffers_new = run_episode(env, list_agents,
                                       epsilon, prime=True)
        step += len(list_buffers_new[0].obs)

        step_train += 1
    
        """

        if idx_episode % period == 0:

            if config.env.name == 'er':
                (reward_total, n_move_lever, n_move_door, rewards_received,
                 rewards_given, steps_per_episode, r_lever,
                 r_start, r_door) = evaluate.test_room_symmetric(
                     n_eval, env, sess, list_agents)
                matrix_combined = np.stack([reward_total, n_move_lever, n_move_door,
                                            rewards_received, rewards_given,
                                            r_lever, r_start, r_door])
            if config.env.name == 'ipd':
                # reward_env = evaluate.test_ipd(n_eval, env, list_agents)
                reward_env, av_share_0 = evaluate.test_ipd(n_eval, env, list_agents, config)
                matrix_combined = np.stack([reward_env, av_share_0])

            s = '%d,%d,%d' % (idx_episode, step_train, step)
            for idx in range(config.env.n_agents):
                s += ','
                if config.env.name == 'er':
                    s += ('{:.3e},{:.3e},{:.3e},{:.3e},{:.3e},'
                          '{:.3e},{:.3e},{:.3e}').format(
                              *matrix_combined[:, idx])
                elif config.env.name == 'ipd':
                    s += '{:.3e},{:.3e}'.format(
                        *matrix_combined[:, idx])
            if config.env.name == 'er':
                s += ',%.2f\n' % steps_per_episode
            else:
                s += '\n'
            with open(os.path.join(log_path, 'log.csv'), 'a') as f:
                f.write(s)


        if epsilon > config.lio.epsilon_end:
            epsilon -= epsilon_step
    

def run_episode(env, list_agents, epsilon, prime=False):
    list_buffers = [Buffer(config.env.n_agents) for _ in range(config.env.n_agents)]
    list_obs, share_0 = env.reset()
    done = False

    """
    trading_ep = 5
    while trading_ep > 0:
        list_actions_trade = []
        for agent in list_agents:
            action_trade = agent.run_trade(list_obs[agent.agent_id], epsilon)
            list_actions_trade.append(action_trade)

        if env.name == 'ipd':
            list_obs_next, share_0, env_rewards = env.trade_step(list_actions_trade, share_0)

        state = np.where(list_obs[agent.agent_id] == 1)
        next_state = np.where(list_obs_next[agent.agent_id] == 1)
        for idx, agent in enumerate(list_agents):
            agent.update(state, list_actions_trade[agent.agent_id], next_state, env_rewards[agent.agent_id])

        trading_ep -= 1
    """

    trade_actions = []
    for agent in list_agents:
        # action = agent.run_trade(list_obs[agent.agent_id], epsilon)
        action = agent.run_actor_basic(list_obs[agent.agent_id], epsilon)
        trade_actions.append(action)

    list_obs_next, share_0, env_rewards = env.trade_step(trade_actions)

    state = np.where(list_obs[agent.agent_id] == 1)
    next_state = np.where(list_obs_next[agent.agent_id] == 1)
    for idx, agent in enumerate(list_agents):
        agent.update(state, trade_actions[agent.agent_id], next_state, env_rewards[agent.agent_id])

    list_obs = list_obs_next

    while not done:
        list_actions = []
        for agent in list_agents:
            # action = agent.run_actor(list_obs[agent.agent_id], epsilon)
            action = agent.run_actor(list_obs[agent.agent_id], epsilon)
            list_actions.append(action)

        if env.name == 'ipd':
            list_obs_next, env_rewards, done, share_0 = env.step(list_actions, share_0)

        state = np.where(list_obs[agent.agent_id] == 1)
        next_state = np.where(list_obs_next[agent.agent_id] == 1)
        for idx, agent in enumerate(list_agents):
            agent.update(state, list_actions[agent.agent_id], next_state, env_rewards[agent.agent_id])

        """
        for idx, buf in enumerate(list_buffers):
            buf.add([list_obs[idx], list_actions[idx], env_rewards[idx],
                     list_obs_next[idx], done])
            buf.add_action_all(list_actions)
        """

        list_obs = list_obs_next

    return list_buffers


class Buffer(object):

    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.reset()

    def reset(self):
        self.obs = []
        self.action = []
        self.reward = []
        self.obs_next = []
        self.done = []
        self.action_all = []

    def add(self, transition):
        self.obs.append(transition[0])
        self.action.append(transition[1])
        self.reward.append(transition[2])
        self.obs_next.append(transition[3])
        self.done.append(transition[4])

    def add_action_all(self, list_actions):
        self.action_all.append(list_actions)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('exp', type=str, choices=['er', 'ipd'])
    args = parser.parse_args()

    if args.exp == 'er':
        config = config_room_lio.get_config()
    elif args.exp == 'ipd':
        config = config_ipd_lio.get_config()

    train(config)
