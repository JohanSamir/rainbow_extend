import os
import itertools

agents = ["dqn"]
seeds = list(range(1, 11))
environments = ["mountaincar"]
experiments = ["learning_rate"]
weights_biases = [False]

trials = list(dict([('agent', ag), ('env', env), ('seed', seed), ('experiment', exp)]) for (ag, env, seed, exp) in itertools.product(agents, environments, seeds, experiments))

for hyperparameters in trials:
    hyperparameters = dict(hyperparameters)
    env = hyperparameters["env"]
    seed = hyperparameters["seed"]
    agent = hyperparameters["agent"]
    exp = hyperparameters["experiment"]

        os.system(f'tmux new-session -d -s {env}_{agent}_{seed}_{exp} sh -c "python3 main_offline_experiments.py --env={env} --agent={agent} --initial_seed={seed} --exp={exp}"')
        # os.system(f'tmux new-session -d -s {env}_{agent}_{seed}_{exp} sh -c "conda activate rain; python3 main_offline_experiments.py --env={env} --agent={agent} --initial_seed={seed} --exp={exp} --wb=True"')
