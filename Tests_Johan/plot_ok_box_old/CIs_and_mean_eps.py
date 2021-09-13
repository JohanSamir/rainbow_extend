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
#environments = ['cartpole', 'acrobot', 'lunarlander', 'mountaincar']
environments = ['cartpole', 'acrobot', 'lunarlander','mountaincar']

component= ['target_opt','replay_scheme','dueling','noisy']
initializers = ['10','5', '2', '1',  '0.5', '0.3125', '0.03125', '0.003125', '0.0003125', '0.00003125']

name = 'johan'
#name_sub = 'ExperimentsInitializer'
#name_sub_sub='2_activation_experiments_Rainbow'

name_sub = 'ExperimentsInitializer/2_epsilons_experiments_Rainbow/'
name_sub_sub='2_experiments_Rainbow_eps'
num_runs = 7


for env in environments:
  h = []
  h_idx = False
  for init in initializers:
    for i in range (1, num_runs):
      print(i)
      base_dir = f'/home/{name}/{name_sub}/True{i}_rainbow_{env}_{init}/dqn_test'
      LOG_PATH = os.path.join(base_dir + str(i))
      data = colab_utils.read_experiment(LOG_PATH, verbose=True, summary_keys=['train_episode_returns'])
      data = data.values[data.shape[0]-5:data.shape[0],:]
      if i == 1:
        f = data
      else:
        f = np.vstack((f, data))

    print('f:', f.shape)
    a = np.array([init]*f.shape[0])
    a = a.reshape(f.shape[0],1)
    dat = np.hstack((f, a))
    print('dat:', dat.shape)
    dat = dat[:,1:2]
    print('dat:', dat.shape)
    dat = dat.reshape((dat.shape[0],1))
    print('dat:', dat.shape)

    if h_idx== False:
      h = dat
      print('h:False', h.shape)
    else:
      h = np.append(h, dat, axis =1)
      #h = np.hstack((h, dat))
      print('h:True', h.shape)
    h_idx = True
    #h = h[:,1:2]

  #total = np.hstack(h)
  total = h
  print('total:', total.shape)
  df = pd.DataFrame(total, columns = ['10','5', '2', '1',  '0.5', '0.3125', '0.03125', '0.003125', '0.0003125', '0.00003125'])
  df = df.astype({'10': float, '5': float,'2': float,
    '1': float,'0.5': float,'0.3125': float,'0.03125': float,
    '0.003125': float, '0.0003125': float,'0.00003125': float})

  plt.figure(figsize=(16,8)) 
  vals, names, xs = [],[],[]
  for i, col in enumerate(df.columns):
    vals.append(df[col].values)
    names.append(col)
    xs.append(np.random.normal(i + 1, 0.04, df[col].values.shape[0]))

  plt.boxplot(vals, labels=names, vert=False)
  palette = ['r', 'g', 'b', 'y','c', 'm', 'r', 'g', 'b', 'y']
  for x, val, c in zip(xs, vals, palette):
    plt.scatter(val, x, alpha=0.4, color=c)

  plt.savefig(f'{name_sub_sub}_{env}_box.pdf',bbox_inches='tight')
