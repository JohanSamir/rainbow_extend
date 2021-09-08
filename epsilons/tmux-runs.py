import os
import itertools

agents = ["rainbow"]
seeds = list(range(1, 11))
environments = ["acrobot", "cartpole", "lunarlander", "mountaincar"]
trials = list(dict([('agent', ag), ('env', env), ('seed', seed)]) for (ag, env, seed) in itertools.product(agents, environments, seeds))

for hyperparameters in trials:
    hyperparameters = dict(hyperparameters)
    env = hyperparameters["env"]
    seed = hyperparameters["seed"]
    agent = hyperparameters["agent"]
    os.system(f'tmux new-session -d -s {env}_{agent}_{seed} sh -c "conda activate rain; python main_online_experiments.py --env={env} --agent={agent} --initial_seed={seed}"')
