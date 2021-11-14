# Learning to Participate

This is the code for my bachelor thesis. It strongly builds on [Learning to Incentivize Other Learning Agents](https://arxiv.org/abs/2006.06051) for which the code can be found [here](https://github.com/011235813/lio).

## Setup

Follow the setup instructions for [LIO](https://github.com/011235813/lio). Importantly, at least in my case, installing the package ray instead of cloning the project [Ray](https://github.com/natashamjaques/ray.git), worked out better.

## Navigation

* `simulation_market.py` - Testing the theoretic ideas behind a participation market.
* `IPD Q-Learning/` - Implementing the Iterated Prisoner's Dilemma with Q-Learning by updating a q-table.
* `lio/`, `lola/`, `ray/`, `sequential_social_dilemma_games/` - Implementing the Iterated Prisoner's Dilemma and Cleanup with a neural network. I experiment with different participation models. `lio/` is the main folder, the others contain supporting scripts.
* `Modelling/` - Visualizing the log files from running the experiment.

## Implementations Iterated Prisoner's Dilemma

- `lio/lio/alg/lio_agent.py`: Set `self.can_give` on false for all scenarios with participation (all except LIO).
- `lio/lio/alg/lio_ac.py`: Set `self.can_give` on false for all scenarios with participation (all except LIO).
- `lio/lio/alg/config_ipd_lio.py`: Set parameters such as the number of agents, and the size of the environment.
- `lio/lio/env/ipd_wrapper.py`: Change `self.l_action` and `self.l_obs` depending on the scenario tested.
- `lola/lola/envs/prisoners_dilemma.py`: Change `NUM_STATES` depending on the scenario tested. Also set one boolean variable for a scenario on true at the beginning of `def step`.

## Running an implementation of the Iterated Prisoner's Dilemma

- For a single run, `cd` into the `alg` folder of `lio` and execute `$ python train_lio.py ipd`.
- For five run, `cd` into the `alg` folder of `lio` and execute `$ python train_multiprocess.py lio ipd`.

## Implementations Cleanup

- `lio/lio/alg/config_ipd_lio.py`: Set parameters such as the number of agents, and the size of the environment.
- `lio/lio/alg/train_ssd.py`: Set one boolean variable for a scenario at the start of `def run_episode` on true.
- `lio/lio/alg/evaluate.py`: Set one boolean variable for a scenario at the start of `def test_ssd` on true.

## Running an implementation of Cleanup

- For a single run, `cd` into the `alg` folder of `lio` and execute `$ python train_ssd.py`.
- For five run, `cd` into the `alg` folder of `lio` and execute `$ python train_multiprocess.py lio ssd`.
