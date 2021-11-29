import os
import itertools

agents = ["rainbow"]
seeds = [0]
environments = ["breakout"]
experiments = ["activation=relu", "depth=1", "epsilon=0.00003", "init=orthogonal", 
                "learning_rate=0.001", "normalization=non_normalization", "width=64", 
                "update_period=2", "target_update_period=50","gamma=0.99",
                "min_replay_history=30", "num_atoms=31", "update_horizon=4", "clip_rewards=False"]

trials = list(dict([('agent', ag), 
                    ('env', env), 
                    ('seed', seed), 
                    ('experiment', exp)]) 
                    for (ag, env, seed, exp) in itertools.product(agents, environments, seeds, experiments))

for hyperparameters in trials:
    hyperparameters = dict(hyperparameters)
    env = hyperparameters["env"]
    seed = hyperparameters["seed"]
    agent = hyperparameters["agent"]
    exp = hyperparameters["experiment"]
    exp_name = exp.split("=")[0]

    if "num_atoms" in exp and agent == "dqn":
        continue
    os.system(f'tmux new-session -d -s {env}_{agent}_{seed}_{exp_name} sh -c "conda activate rain; python3 main.py --env={env} --agent={agent} --seed={seed} --experiment={exp} --test=True"')
    # os.system(f'tmux new-session -d -s {env}_{agent}_{seed}_{exp_name} sh -c "python3 main.py --env={env} --agent={agent} --seed={seed} --exp={exp} --test=True"')
