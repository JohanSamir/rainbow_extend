import itertools
import utils

agents = ["dqn", "rainbow"]
environments = ["acrobot", "cartpole"]#, "lunarlander", "mountaincar"]
seeds = list(range(1))
num_runs = 2
groups = ["effective_horizon"]#, "constancy_of_parameters", 
                #"network_starting point", "network_architecture",
                #"optimizer_parameters"]
experiments = []
for grp in groups:
    values = utils.sample_group(grp, num_runs)
    experiments.extend([f"{grp}=" + f"{tuple(val)}"[1:-1] for val in values])

trials = list({'agent': ag, 'env': env, 
                'experiment':exp, 
                'seed':sd} for (ag, env, exp, sd) in itertools.product(agents, environments, experiments, seeds))

for hyperparameters in trials:
    hyperparameters = dict(hyperparameters)
    print(hyperparameters)