from utils import *
full_exps = []
for k, v in experiments.items():
    full_exps.extend(f"{k}={vv}" for vv in v)
print(full_exps)