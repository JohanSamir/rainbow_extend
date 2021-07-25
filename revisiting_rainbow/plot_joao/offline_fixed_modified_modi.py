import dopamine
from dopamine.colab import utils as colab_utils
import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib
import matplotlib.pyplot as plt

agent = ["rainbow"]
environments = ['acrobot', 'lunarlander', 'mountaincar']
component= ['target_opt','replay_scheme','dueling','noisy']
initializers = ["zeros", "ones", "variance_baseline",
                "variance_0.1", "variance_0.3", "variance_0.8", 
                "variance_3", "variance_5", "variance_10",
                "he_nor", "he_uni", "xavier_uni", "xavier_nor",
                "lecun_uni", "lecun_nor", "orthogonal"]

index_agt=0
index_en=0
index_com=0

name = 'joaoguilhermearujo'
name_sub = 'tests_joao'
name_sub_sub='offline_last_pct'

for env in environments:
  h = []
  for init in initializers:
    for i in range (1, 10):
      print(i)
      base_dir = f'/home/{name}/{name_sub}/{name_sub_sub}/rainbow_{env}_{init}_fixed_20/test'
      LOG_PATH = os.path.join(base_dir + str(i))
      data = colab_utils.read_experiment(LOG_PATH, verbose=True, summary_keys=['eval_episode_returns'])
      data = data.values
      if i == 1:
        f = data
      # data['iteration'] = data['iteration']
      else:
        f = np.vstack((f, data))

    a = np.array([init]*f.shape[0])
    a = a.reshape(f.shape[0],1)
    h.append(np.hstack((f, a)))

  total = np.vstack(h)
  total = pd.DataFrame(total, columns = ['iteration', 'eval_episode_returns','agent_id'])
  total = total.astype({"iteration": float, "eval_episode_returns": float, "agent_id":str })
  fig, ax = plt.subplots(figsize=(16,8))
  sns.lineplot(x='iteration', y='eval_episode_returns', hue='agent_id', style='agent_id', markers=True, dashes=False, data=total)
  #sns.lineplot(x='iteration', y='train_episode_returns', hue='agent_id', style='agent_id', markers=True, dashes=False, palette="flare", data=total)
  plt.savefig(f'fixed_rainbow_{env}_20%.pdf',bbox_inches='tight')
