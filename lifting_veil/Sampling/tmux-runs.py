import os
import itertools

agents = ["dqn", "rainbow"]
seeds = list(range(1, 11))
environments = ["acrobot", "cartpole", "lunarlander", "mountaincar"]
experiments = ["activation", "depth", "epsilon", "init", 
                "learning_rate", "normalization", "width", 
                "update_period", "target_update_period","gamma",
                "min_replay_history", "num_atoms", "update_horizon", "clip_rewards"]

experiments = { "effective_horizon" : ["update_period", "gamma"],
                "constancy_of_parameters" : ["init", "update_horizon", "noisy_net"],
                "network_starting point" : ["init", "activation", "depth", "normalization"],
                "network_architecture" : ["depth", "width", "normalization"],
                #"algorithmic_parameters" : ["update_period", "gamma"],
                #"distribution_parameterization" : ["update_period", "gamma"],
                "optimizer_parameters" : ["learning_rate", "epsilon", "batch_sizes"]
                #"bellman_updates" : ["update_period", "gamma"]
                }

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

    if exp == "num_atoms" and agent == "dqn":
        continue
    #os.system(f'tmux new-session -d -s {env}_{agent}_{seed} sh -c "conda activate rain; python3 main_online_experiments.py --env={env} --agent={agent} --initial_seed={seed}"')
    os.system(f'tmux new-session -d -s {env}_{agent}_{seed}_{exp} sh -c "python3 main.py --env={env} --agent={agent} --initial_seed={seed} --exp={exp}"')
