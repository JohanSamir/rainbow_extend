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
environments = ['cartpole', 'acrobot', 'mountaincar']

component= ['target_opt','replay_scheme','dueling','noisy']
initializers = ["conf_0_relu", "conf_1_relu", "conf_2_relu6",
                "conf_3_sigmoid", "conf_4_softplus", "conf_5_soft_sign", 
                "conf_6_silu", "conf_7_swish", "conf_8_log_sigmoid",
                "conf_9_hard_sigmoid", "conf_10_hard_silu", "conf_11_hard_swish",
                "conf_12_hard_tanh", "conf_13_elu", "conf_14_celu", 
                "conf_15_selu", "conf_16_gelu", "conf_17_glu"]
name = 'johan'
name_sub = 'ExperimentsInitializer'
name_sub_sub='2_activation_experiments_Rainbow'
num_runs = 7


for env in environments:
  h = []
  h_idx = False
  for init in initializers:
    for i in range (1, num_runs):
      print(i)
      base_dir = f'/home/{name}/{name_sub}/{name_sub_sub}/True{i}_rainbow_{env}_{init}/dqn_test'
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
  df = pd.DataFrame(total, columns = ["conf_0_relu", "conf_1_relu", "conf_2_relu6",
                "conf_3_sigmoid", "conf_4_softplus", "conf_5_soft_sign", 
                "conf_6_silu", "conf_7_swish", "conf_8_log_sigmoid",
                "conf_9_hard_sigmoid", "conf_10_hard_silu", "conf_11_hard_swish",
                "conf_12_hard_tanh", "conf_13_elu", "conf_14_celu", 
                "conf_15_selu", "conf_16_gelu", "conf_17_glu"])

  df = df.astype({'conf_0_relu': float, 'conf_1_relu': float,'conf_2_relu6': float,
    'conf_3_sigmoid': float,'conf_4_softplus': float,'conf_5_soft_sign': float,'conf_6_silu': float,
    'conf_7_swish': float, 'conf_8_log_sigmoid': float,'conf_9_hard_sigmoid': float,'conf_10_hard_silu': float,
    'conf_11_hard_swish': float,'conf_12_hard_tanh': float,'conf_13_elu': float,'conf_14_celu': float, 'conf_15_selu':float,
    'conf_16_gelu': float,'conf_17_glu': float})

  plt.figure(figsize=(16,8)) 
  vals, names, xs = [],[],[]
  for i, col in enumerate(df.columns):
    vals.append(df[col].values)
    names.append(col)
    xs.append(np.random.normal(i + 1, 0.04, df[col].values.shape[0]))

  plt.boxplot(vals, labels=names, vert=False)
  palette = ['r', 'g', 'b', 'y','c', 'm', 'r', 'g', 'b', 'y','c', 'm', 'r', 'g', 'b', 'y','c','m']
  for x, val, c in zip(xs, vals, palette):
    plt.scatter(val, x, alpha=0.4, color=c)

  plt.savefig(f'{name_sub_sub}_{env}_box.pdf',bbox_inches='tight')