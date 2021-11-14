from lio.utils.configdict import ConfigDict


def get_config():
    """
    changed:
    agent --> run_actor()
    lola --> step()
    ipd wrapper
    """

    config = ConfigDict()

    config.alg = ConfigDict()
    config.alg.n_episodes = 20000 # 60000
    config.alg.n_eval = 100 # 100
    config.alg.n_test = 2 # 2
    config.alg.period = 100 # 1000

    config.env = ConfigDict()
    config.env.name = 'ipd'
    config.env.max_steps = 5 # 5 for q-learning, 40 for actor-critic and shares
    config.env.n_agents = 2
    config.env.r_multiplier = 3.0  # scale up sigmoid output

    config.lio = ConfigDict()
    config.lio.asymmetric = False
    config.lio.decentralized = False # False
    config.lio.entropy_coeff = 0.1  # 0.1
    config.lio.epsilon_div = 5000
    config.lio.epsilon_end = 0.01 # 0.01
    config.lio.epsilon_start = 1.0
    config.lio.gamma = 0.99
    config.lio.include_cost_in_chain_rule = False
    config.lio.idx_recipient = 1  # only used if asymmetric=True
    config.lio.lr_actor = 1e-3 # 1e-3
    config.lio.lr_cost = 1e-4 # added
    config.lio.lr_reward = 1e-3 # 1e-3
    config.lio.lr_v = 1e-2 # added
    config.lio.lr_opp = 1e-3 # added
    config.lio.tau = 0.01 # added
    config.lio.optimizer = 'adam'
    config.lio.reg = 'l1'
    config.lio.reg_coeff = 0.0
    config.lio.separate_cost_optimizer = True
    config.lio.use_actor_critic = True # False

    config.main = ConfigDict()
    config.main.dir_name = 'ipd_lio'
    config.main.exp_name = 'ipd'
    config.main.max_to_keep = 100
    config.main.model_name = 'model.ckpt'
    config.main.save_period = 100000
    config.main.seed = 12341 
    config.main.summarize = False
    config.main.use_gpu = False

    config.nn = ConfigDict()
    config.nn.n_h1 = 16
    config.nn.n_h2 = 8
    config.nn.n_hr1 = 16
    config.nn.n_hr2 = 8

    return config
