from lola.envs.prisoners_dilemma import IteratedPrisonersDilemma

class IPD(IteratedPrisonersDilemma):

    def __init__(self, config):

        super().__init__(max_steps=config.max_steps)
        self.n_agents = 2
        self.l_action = 2 # 2 # 4 # 6 for shares   
        self.l_obs = 5   # 17 # 133    # depending on shares, initial value: 5
        self.max_steps = config.max_steps
        self.share_0 = 1.0 # added!
        self.name = 'ipd'
