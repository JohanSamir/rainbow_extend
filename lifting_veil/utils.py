from agents.dqn_agent_new import *
from agents.rainbow_agent_new import *
from absl import logging
import numpy as np
import itertools
import bisect
import classic_params
import minatar_params
import atari_100k_params


suites = {
    "classic": classic_params,
    "minatar": minatar_params,
    "atari_100k": atari_100k_params
}

agents = {
    'dqn': JaxDQNAgentNew,
    'rainbow': JaxRainbowAgentNew,
    'c51': JaxRainbowAgentNew,
    'rainbow_without': JaxDQNAgentNew,
    'drq_eps': JaxDQNAgentNew    
}

def get_init_bidings(agent_name, init, seed=None):
    initializer = inits[init]['function'].__name__
    if init == 'zeros' or init == 'ones':
        gin_bindings = [f"{agent_name}.seed={seed}",
                        f"{agent_name}.initzer = @{initializer}"]

    elif init == "orthogonal":
        gin_bindings = [f"{agent_name}.seed={seed}",
                        f"{agent_name}.initzer = @{initializer}()",
                        f"{initializer}.scale = 1"]
    else:
        mode = '"'+inits[init]['mode']+'"'
        scale = inits[init]['scale']
        distribution = '"'+inits[init]['distribution']+'"'
        gin_bindings = [f"{agent_name}.seed={seed}",
                        f"{agent_name}.initzer = @{initializer}()",
                        f"{initializer}.scale = {scale}",
                        f"{initializer}.mode = {mode}",
                        f"{initializer}.distribution = {distribution}"
                        ]
    return gin_bindings

def get_gin_bindings(exp, agent_name, initial_seed, value, test):
    gin_bindings = [f"{agent_name}.seed={initial_seed}"]
    if exp == "epsilon":
        gin_bindings += [f"create_opt.eps = {value}"]

    elif exp == "learning_rate":
        gin_bindings += [f"create_opt.learning_rate = {value}"]
    
    elif exp == "weight_decay":
        gin_bindings += [f"create_opt.weight_decay = {value}"]

    elif exp == "width":
        gin_bindings += [f"{agent_name}.neurons = {value}"]

    elif exp == "depth":
        gin_bindings += [f"{agent_name}.hidden_layer = {value}"]

    elif exp == "conv":
        gin_bindings += [f"{agent_name}.hidden_conv = {value}"]

    elif exp == "normalization":
        gin_bindings += [f"{agent_name}.normalization = '{value}'"]

    elif "init" in exp:
        gin_bindings = get_init_bidings(agent_name, value, initial_seed)

    elif exp == "activation":
        gin_bindings += [f"{agent_name}.layer_funct = '{value}'"]

    elif exp == "update_period":
        gin_bindings += [f"{agent_name}.update_period = {value}"]

    elif exp == "target_update_period":
        gin_bindings += [f"{agent_name}.target_update_period = {value}"]
    
    elif exp == "gamma":
        gin_bindings += [f"{agent_name}.gamma = {value}"]
    
    elif exp == "min_replay_history":
        gin_bindings += [f"{agent_name}.min_replay_history = {value}"]

    elif exp == "num_atoms":
        gin_bindings += [f"{agent_name}.num_atoms = {value}"]

    elif exp == "update_horizon":
        gin_bindings += [f"{agent_name}.update_horizon = {value}"]

    elif exp == "clip_rewards":
        gin_bindings += [f"Runner.clip_rewards = {value}"]
    
    elif exp == "batch_size":   
        gin_bindings += [f"OutOfGraphPrioritizedReplayBuffer.batch_size = {value}"]
        
    elif exp == "noisy_net":
        gin_bindings += [f"{agent_name}.noisy = {value}"]
    
    else:
        logging.error("Error! Check the kind of experiment")
        raise ValueError("Experiment not recognized")

    if test:
        gin_bindings.extend(["Runner.num_iterations=4", "Runner.training_steps=200"])
        
    return gin_bindings


def repr_values(values):
    cat = "_".join(str(val) for val in values)
    return cat.replace(".", "p")

def cast_to_int(lst):
    for idx, el in enumerate(lst):
        if type(el) == np.float64 and el.is_integer():
            lst[idx] = int(el)
    return lst

def sample_group(category, grp, seed, num=1): 

    if grp == "default":
        sample = []

    else:
        rng = np.random.default_rng(seed)
        total = list(itertools.product(*[suites[category].experiments[exp] for exp in suites[category].groups[grp]]))
        total = np.array(total)
        indices = rng.choice(len(total), num, replace=False)
        sample = cast_to_int(list(total[indices][0]))        
        cs = np.cumsum([0] + [len(suites[category].experiments[exp]) for exp in suites[category].groups[grp]])
        seed %= cs[-1]
        idx = bisect.bisect(cs, seed) - 1
        sample[idx] = suites[category].experiments[suites[category].groups[grp][idx]][seed - cs[idx]]
        logging.info(f"Sample Seed Index = {idx}")
        logging.info(f"Changed {suites[category].groups[grp][idx]} of group {grp} to {sample[idx]}")

    return sample

def print_groups(category="classic"):
    print('groups = {}')
    for grp in suites[category].groups:
      print(f"groups['{grp}'] = " + "{")
      cs = np.cumsum([0] + [len(suites[category].experiments[exp]) for exp in suites[category].groups[grp]])
      for seed in range(cs[-1]):
        idx = bisect.bisect(cs, seed) - 1
        sample = suites[category].experiments[suites[category].groups[grp][idx]][seed - cs[idx]]
        print(f"  '{seed}': '{suites[category].groups[grp][idx]}={sample}',")
      print('}')

if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
