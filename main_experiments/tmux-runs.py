import os
import itertools

agents = ["dqn", "rainbow"]
types = ["online", "offline"]
seeds = list(range(1, 11))
environments = ["acrobot", "cartpole", "lunarlander", "mountaincar"]
experiments = ["activation", "depth", "epsilon", "init", "learning_rate", "normalization", "width", "update_period"]

trials = list(dict([('agent', ag), ('env', env), ('seed', seed), ('experiment', exp), ("type", typ)]) for (ag, env, seed, exp, typ) in itertools.product(agents, environments, seeds, experiments, types))

for hyperparameters in trials:
    hyperparameters = dict(hyperparameters)
    env = hyperparameters["env"]
    typ = hyperparameters["type"]
    seed = hyperparameters["seed"]
    agent = hyperparameters["agent"]
    exp = hyperparameters["experiment"]

    #os.system(f'tmux new-session -d -s {env}_{agent}_{seed} sh -c "conda activate rain; python3 main_online_experiments.py --env={env} --agent={agent} --initial_seed={seed}"')
    os.system(f'tmux new-session -d -s {typ}_{env}_{agent}_{seed}_{exp} sh -c "python3 main.py --env={env} --agent={agent} --initial_seed={seed} --experiment={exp} --type={typ}"')
