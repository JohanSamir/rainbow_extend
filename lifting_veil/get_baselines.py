import os
import itertools

agents = ["dqn", "rainbow"]
environments = ["acrobot", "cartpole", "lunarlander", "mountaincar"]
trials = list(dict([('agent', ag), ('env', env)]) for (ag, env) in itertools.product(agents, environments))

for hyperparameters in trials:
    hyperparameters = dict(hyperparameters)
    env = hyperparameters["env"]
    agent = hyperparameters["agent"]

    os.system(f'tmux new-session -d -s {env}_{agent} sh -c "python3 baseline.py --env={env} --agent={agent}"')
    