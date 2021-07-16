#!/usr/bin/env python
# coding: utf-8

#In[1]:
import dopamine
from dopamine.colab import utils as colab_utils
import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib
import matplotlib.pyplot as plt

#In[2]:

agent =['dqn','rainbow','quantile', 'implicit']
#agent =['dqn','rainbow','quantile']
env= ['cartpole','acrobot']
component= ['target_opt','replay_scheme','dueling','noisy']

index_agt=3
index_en=0
index_com=0

name = 'joaoguilhermearujo'
name_sub = 'tests_joao'
name_sub_sub='offline_init_mse'

#agent = 'dqn'
#agent = 'rainbow'
#agent = 'quantile'
#agent = 'implicit'

dic = {1:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_zeros_fixed_first_20/test1',
       2:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_zeros_fixed_first_20/test2',
       3:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_zeros_fixed_first_20/test3',
       4:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_zeros_fixed_first_20/test4',
       5:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_zeros_fixed_first_20/test5',
       6:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_zeros_fixed_first_20/test6',
       7:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_zeros_fixed_first_20/test7',
       8:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_zeros_fixed_first_20/test8',
       9:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_zeros_fixed_first_20/test9',
       10:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_zeros_fixed_first_20/test10'}

n = 10
for i in range (1,n):
  print(i)
  if i == 1:
    LOG_PATH = os.path.join(dic[i])
    data = colab_utils.read_experiment(LOG_PATH, verbose=True, summary_keys=['eval_episode_returns'])
    data = data.values
    f = data

  LOG_PATH = os.path.join(dic[i])
  data = colab_utils.read_experiment(LOG_PATH, verbose=True, summary_keys=['eval_episode_returns'])
  data['iteration'] = data['iteration']
  data = data.values
  f = np.vstack((f,data))

a = np.array(['zeros']*f.shape[0])
a = a.reshape(f.shape[0],1)
#a = np.zeros((f.shape[0],1))+0
f1 = np.hstack((f,a))

dic = {1:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_ones_fixed_first_20/test1',
       2:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_ones_fixed_first_20/test2',
       3:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_ones_fixed_first_20/test3',
       4:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_ones_fixed_first_20/test4',
       5:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_ones_fixed_first_20/test5',
       6:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_ones_fixed_first_20/test6',
       7:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_ones_fixed_first_20/test7',
       8:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_ones_fixed_first_20/test8',
       9:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_ones_fixed_first_20/test9',
       10:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_ones_fixed_first_20/test10'}

for i in range (1,n):
  print(i)
  if i == 1:
    LOG_PATH = os.path.join(dic[i])
    data = colab_utils.read_experiment(LOG_PATH, verbose=True, summary_keys=['eval_episode_returns'])
    data = data.values
    f = data

  LOG_PATH = os.path.join(dic[i])
  data = colab_utils.read_experiment(LOG_PATH, verbose=True, summary_keys=['eval_episode_returns'])
  data['iteration'] = data['iteration']
  data = data.values
  f = np.vstack((f,data))

a = np.array(['ones']*f.shape[0])
a = a.reshape(f.shape[0],1)
f2 = np.hstack((f,a))

dic = {1:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_baseline_fixed_first_20/test1',
       2:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_baseline_fixed_first_20/test2',
       3:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_baseline_fixed_first_20/test3',
       4:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_baseline_fixed_first_20/test4',
       5:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_baseline_fixed_first_20/test5',
       6:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_baseline_fixed_first_20/test6',
       7:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_baseline_fixed_first_20/test7',
       8:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_baseline_fixed_first_20/test8',
       9:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_baseline_fixed_first_20/test9',
       10:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_baseline_fixed_first_20/test10'}

for i in range (1,n):
  print(i)
  if i == 1:
    LOG_PATH = os.path.join(dic[i])
    data = colab_utils.read_experiment(LOG_PATH, verbose=True, summary_keys=['eval_episode_returns'])
    data = data.values
    f = data

  LOG_PATH = os.path.join(dic[i])
  data = colab_utils.read_experiment(LOG_PATH, verbose=True, summary_keys=['eval_episode_returns'])
  data['iteration'] = data['iteration']
  data = data.values
  f = np.vstack((f,data))

a = np.array(['variance_baseline']*f.shape[0])
a = a.reshape(f.shape[0],1)
f3 = np.hstack((f,a))

dic = {1:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_0.1_fixed_first_20/test1',
       2:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_0.1_fixed_first_20/test2',
       3:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_0.1_fixed_first_20/test3',
       4:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_0.1_fixed_first_20/test4',
       5:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_0.1_fixed_first_20/test5',
       6:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_0.1_fixed_first_20/test6',
       7:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_0.1_fixed_first_20/test7',
       8:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_0.1_fixed_first_20/test8',
       9:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_0.1_fixed_first_20/test9',
       10:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_0.1_fixed_first_20/test10'}


for i in range (1,n):
  print(i)
  if i == 1:
    LOG_PATH = os.path.join(dic[i])
    data = colab_utils.read_experiment(LOG_PATH, verbose=True, summary_keys=['eval_episode_returns'])
    data = data.values
    f = data

  LOG_PATH = os.path.join(dic[i])
  data = colab_utils.read_experiment(LOG_PATH, verbose=True, summary_keys=['eval_episode_returns'])
  data['iteration'] = data['iteration']
  data = data.values
  f = np.vstack((f,data))

a = np.array(['variance_0.1']*f.shape[0])
a = a.reshape(f.shape[0],1)
f4 = np.hstack((f,a))

dic = {1:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_0.3_fixed_first_20/test1',
       2:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_0.3_fixed_first_20/test2',
       3:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_0.3_fixed_first_20/test3',
       4:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_0.3_fixed_first_20/test4',
       5:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_0.3_fixed_first_20/test5',
       6:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_0.3_fixed_first_20/test6',
       7:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_0.3_fixed_first_20/test7',
       8:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_0.3_fixed_first_20/test8',
       9:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_0.3_fixed_first_20/test9',
       10:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_0.3_fixed_first_20/test10'}

for i in range (1,n):
  print(i)
  if i == 1:
    LOG_PATH = os.path.join(dic[i])
    data = colab_utils.read_experiment(LOG_PATH, verbose=True, summary_keys=['eval_episode_returns'])
    data = data.values
    f = data

  LOG_PATH = os.path.join(dic[i])
  data = colab_utils.read_experiment(LOG_PATH, verbose=True, summary_keys=['eval_episode_returns'])
  data['iteration'] = data['iteration']
  data = data.values
  f = np.vstack((f,data))

a = np.array(['variance_0.3']*f.shape[0])
a = a.reshape(f.shape[0],1)
f5 = np.hstack((f,a))

dic = {1:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_0.8_fixed_first_20/test1',
       2:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_0.8_fixed_first_20/test2',
       3:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_0.8_fixed_first_20/test3',
       4:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_0.8_fixed_first_20/test4',
       5:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_0.8_fixed_first_20/test5',
       6:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_0.8_fixed_first_20/test6',
       7:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_0.8_fixed_first_20/test7',
       8:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_0.8_fixed_first_20/test8',
       9:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_0.8_fixed_first_20/test9',
       10:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_0.8_fixed_first_20/test10'}

for i in range (1,n):
  print(i)
  if i == 1:
    LOG_PATH = os.path.join(dic[i])
    data = colab_utils.read_experiment(LOG_PATH, verbose=True, summary_keys=['eval_episode_returns'])
    data = data.values
    f = data

  LOG_PATH = os.path.join(dic[i])
  data = colab_utils.read_experiment(LOG_PATH, verbose=True, summary_keys=['eval_episode_returns'])
  data['iteration'] = data['iteration']
  data = data.values
  f = np.vstack((f,data))

a = np.array(['variance_0.8']*f.shape[0])
a = a.reshape(f.shape[0],1)
f6 = np.hstack((f,a))

dic = {1:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_he_uni_fixed_first_20/test1',
       2:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_he_uni_fixed_first_20/test2',
       3:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_he_uni_fixed_first_20/test3',
       4:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_he_uni_fixed_first_20/test4',
       5:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_he_uni_fixed_first_20/test5',
       6:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_he_uni_fixed_first_20/test6',
       7:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_he_uni_fixed_first_20/test7',
       8:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_he_uni_fixed_first_20/test8',
       9:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_he_uni_fixed_first_20/test9',
       10:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_he_uni_fixed_first_20/test10'}


for i in range (1,n):
  print(i)
  if i == 1:
    LOG_PATH = os.path.join(dic[i])
    data = colab_utils.read_experiment(LOG_PATH, verbose=True, summary_keys=['eval_episode_returns'])
    data = data.values
    f = data

  LOG_PATH = os.path.join(dic[i])
  data = colab_utils.read_experiment(LOG_PATH, verbose=True, summary_keys=['eval_episode_returns'])
  data['iteration'] = data['iteration']
  data = data.values
  f = np.vstack((f,data))

a = np.array(['he_uni (variance_2)']*f.shape[0])
a = a.reshape(f.shape[0],1)
f8 = np.hstack((f,a))

dic = {1:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_5_fixed_first_20/test1',
       2:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_5_fixed_first_20/test2',
       3:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_5_fixed_first_20/test3',
       4:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_5_fixed_first_20/test4',
       5:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_5_fixed_first_20/test5',
       6:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_5_fixed_first_20/test6',
       7:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_5_fixed_first_20/test7',
       8:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_5_fixed_first_20/test8',
       9:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_5_fixed_first_20/test9',
       10:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_5_fixed_first_20/test10'}

for i in range (1,n):
  print(i)
  if i == 1:
    LOG_PATH = os.path.join(dic[i])
    data = colab_utils.read_experiment(LOG_PATH, verbose=True, summary_keys=['eval_episode_returns'])
    data = data.values
    f = data

  LOG_PATH = os.path.join(dic[i])
  data = colab_utils.read_experiment(LOG_PATH, verbose=True, summary_keys=['eval_episode_returns'])
  data['iteration'] = data['iteration']
  data = data.values
  f = np.vstack((f,data))

a = np.array(['variance_5']*f.shape[0])
a = a.reshape(f.shape[0],1)
f9 = np.hstack((f,a))

dic = {1:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_10_fixed_first_20/test1',
       2:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_10_fixed_first_20/test2',
       3:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_10_fixed_first_20/test3',
       4:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_10_fixed_first_20/test4',
       5:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_10_fixed_first_20/test5',
       6:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_10_fixed_first_20/test6',
       7:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_10_fixed_first_20/test7',
       8:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_10_fixed_first_20/test8',
       9:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_10_fixed_first_20/test9',
       10:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_10_fixed_first_20/test10'}

for i in range (1,n):
  print(i)
  if i == 1:
    LOG_PATH = os.path.join(dic[i])
    data = colab_utils.read_experiment(LOG_PATH, verbose=True, summary_keys=['eval_episode_returns'])
    data = data.values
    f = data

  LOG_PATH = os.path.join(dic[i])
  data = colab_utils.read_experiment(LOG_PATH, verbose=True, summary_keys=['eval_episode_returns'])
  data['iteration'] = data['iteration']
  data = data.values
  f = np.vstack((f,data))

a = np.array(['variance_10']*f.shape[0])
a = a.reshape(f.shape[0],1)
f10 = np.hstack((f,a))

dic = {1:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_xavier_uni_fixed_first_20/test1',
       2:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_xavier_uni_fixed_first_20/test2',
       3:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_xavier_uni_fixed_first_20/test3',
       4:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_xavier_uni_fixed_first_20/test4',
       5:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_xavier_uni_fixed_first_20/test5',
       6:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_xavier_uni_fixed_first_20/test6',
       7:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_xavier_uni_fixed_first_20/test7',
       8:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_xavier_uni_fixed_first_20/test8',
       9:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_xavier_uni_fixed_first_20/test9',
       10:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_xavier_uni_fixed_first_20/test10'}

for i in range (1,n):
  print(i)
  if i == 1:
    LOG_PATH = os.path.join(dic[i])
    data = colab_utils.read_experiment(LOG_PATH, verbose=True, summary_keys=['eval_episode_returns'])
    data = data.values
    f = data

  LOG_PATH = os.path.join(dic[i])
  data = colab_utils.read_experiment(LOG_PATH, verbose=True, summary_keys=['eval_episode_returns'])
  data['iteration'] = data['iteration']
  data = data.values
  f = np.vstack((f,data))

a = np.array(['xavier_uni']*f.shape[0])
a = a.reshape(f.shape[0],1)
f11 = np.hstack((f,a))

dic = {1:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_xavier_nor_fixed_first_20/test1',
       2:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_xavier_nor_fixed_first_20/test2',
       3:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_xavier_nor_fixed_first_20/test3',
       4:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_xavier_nor_fixed_first_20/test4',
       5:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_xavier_nor_fixed_first_20/test5',
       6:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_xavier_nor_fixed_first_20/test6',
       7:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_xavier_nor_fixed_first_20/test7',
       8:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_xavier_nor_fixed_first_20/test8',
       9:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_xavier_nor_fixed_first_20/test9',
       10:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_xavier_nor_fixed_first_20/test10'}


for i in range (1,n):
  print(i)
  if i == 1:
    LOG_PATH = os.path.join(dic[i])
    data = colab_utils.read_experiment(LOG_PATH, verbose=True, summary_keys=['eval_episode_returns'])
    data = data.values
    f = data

  LOG_PATH = os.path.join(dic[i])
  data = colab_utils.read_experiment(LOG_PATH, verbose=True, summary_keys=['eval_episode_returns'])
  data['iteration'] = data['iteration']
  data = data.values
  f = np.vstack((f,data))

a = np.array(['xavier_nor']*f.shape[0])
a = a.reshape(f.shape[0],1)
f12 = np.hstack((f,a))

dic = {1:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_lecun_uni_fixed_first_20/test1',
       2:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_lecun_uni_fixed_first_20/test2',
       3:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_lecun_uni_fixed_first_20/test3',
       4:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_lecun_uni_fixed_first_20/test4',
       5:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_lecun_uni_fixed_first_20/test5',
       6:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_lecun_uni_fixed_first_20/test6',
       7:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_lecun_uni_fixed_first_20/test7',
       8:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_lecun_uni_fixed_first_20/test8',
       9:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_lecun_uni_fixed_first_20/test9',
       10:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_lecun_uni_fixed_first_20/test10'}

for i in range (1,n):
  print(i)
  if i == 1:
    LOG_PATH = os.path.join(dic[i])
    data = colab_utils.read_experiment(LOG_PATH, verbose=True, summary_keys=['eval_episode_returns'])
    data = data.values
    f = data

  LOG_PATH = os.path.join(dic[i])
  data = colab_utils.read_experiment(LOG_PATH, verbose=True, summary_keys=['eval_episode_returns'])
  data['iteration'] = data['iteration']
  data = data.values
  f = np.vstack((f,data))

a = np.array(['lecun_uni (variance_1)']*f.shape[0])
a = a.reshape(f.shape[0],1)
f13 = np.hstack((f,a))

dic = {1:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_lecun_nor_fixed_first_20/test1',
       2:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_lecun_nor_fixed_first_20/test2',
       3:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_lecun_nor_fixed_first_20/test3',
       4:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_lecun_nor_fixed_first_20/test4',
       5:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_lecun_nor_fixed_first_20/test5',
       6:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_lecun_nor_fixed_first_20/test6',
       7:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_lecun_nor_fixed_first_20/test7',
       8:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_lecun_nor_fixed_first_20/test8',
       9:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_lecun_nor_fixed_first_20/test9',
       10:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_lecun_nor_fixed_first_20/test10'}


for i in range (1,n):
  print(i)
  if i == 1:
    LOG_PATH = os.path.join(dic[i])
    data = colab_utils.read_experiment(LOG_PATH, verbose=True, summary_keys=['eval_episode_returns'])
    data = data.values
    f = data

  LOG_PATH = os.path.join(dic[i])
  data = colab_utils.read_experiment(LOG_PATH, verbose=True, summary_keys=['eval_episode_returns'])
  data['iteration'] = data['iteration']
  data = data.values
  f = np.vstack((f,data))

a = np.array(['lecun_nor']*f.shape[0])
a = a.reshape(f.shape[0],1)
f14 = np.hstack((f,a))

dic = {1:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_he_nor_fixed_first_20/test1',
       2:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_he_nor_fixed_first_20/test2',
       3:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_he_nor_fixed_first_20/test3',
       4:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_he_nor_fixed_first_20/test4',
       5:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_he_nor_fixed_first_20/test5',
       6:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_he_nor_fixed_first_20/test6',
       7:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_he_nor_fixed_first_20/test7',
       8:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_he_nor_fixed_first_20/test8',
       9:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_he_nor_fixed_first_20/test9',
       10:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_he_nor_fixed_first_20/test10'}

for i in range (1,n):
  print(i)
  if i == 1:
    LOG_PATH = os.path.join(dic[i])
    data = colab_utils.read_experiment(LOG_PATH, verbose=True, summary_keys=['eval_episode_returns'])
    data = data.values
    f = data

  LOG_PATH = os.path.join(dic[i])
  data = colab_utils.read_experiment(LOG_PATH, verbose=True, summary_keys=['eval_episode_returns'])
  data['iteration'] = data['iteration']
  data = data.values
  f = np.vstack((f,data))

a = np.array(['he_nor']*f.shape[0])
a = a.reshape(f.shape[0],1)
f15 = np.hstack((f,a))

dic = {1:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_3_fixed_first_20/test1',
       2:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_3_fixed_first_20/test2',
       3:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_3_fixed_first_20/test3',
       4:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_3_fixed_first_20/test4',
       5:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_3_fixed_first_20/test5',
       6:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_3_fixed_first_20/test6',
       7:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_3_fixed_first_20/test7',
       8:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_3_fixed_first_20/test8',
       9:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_3_fixed_first_20/test9',
       10:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_variance_3_fixed_first_20/test10'}

for i in range (1,n):
  print(i)
  if i == 1:
    LOG_PATH = os.path.join(dic[i])
    data = colab_utils.read_experiment(LOG_PATH, verbose=True, summary_keys=['eval_episode_returns'])
    data = data.values
    f = data

  LOG_PATH = os.path.join(dic[i])
  data = colab_utils.read_experiment(LOG_PATH, verbose=True, summary_keys=['eval_episode_returns'])
  data['iteration'] = data['iteration']
  data = data.values
  f = np.vstack((f,data))

a = np.array(['variance_3']*f.shape[0])
a = a.reshape(f.shape[0],1)
f16 = np.hstack((f,a))


dic = {1:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_orthogonal_fixed_first_20/test1',
       2:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_orthogonal_fixed_first_20/test2',
       3:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_orthogonal_fixed_first_20/test3',
       4:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_orthogonal_fixed_first_20/test4',
       5:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_orthogonal_fixed_first_20/test5',
       6:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_orthogonal_fixed_first_20/test6',
       7:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_orthogonal_fixed_first_20/test7',
       8:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_orthogonal_fixed_first_20/test8',
       9:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_orthogonal_fixed_first_20/test9',
       10:f'/home/{name}/{name_sub}/{name_sub_sub}/{agent[index_agt]}_orthogonal_fixed_first_20/test10'}
for i in range (1,n):
  print(i)
  if i == 1:
    LOG_PATH = os.path.join(dic[i])
    data = colab_utils.read_experiment(LOG_PATH, verbose=True, summary_keys=['eval_episode_returns'])
    data = data.values
    f = data

  LOG_PATH = os.path.join(dic[i])
  data = colab_utils.read_experiment(LOG_PATH, verbose=True, summary_keys=['eval_episode_returns'])
  data['iteration'] = data['iteration']
  data = data.values
  f = np.vstack((f,data))

a = np.array(['orthogonal']*f.shape[0])
a = a.reshape(f.shape[0],1)
f17 = np.hstack((f,a))

# In[10]:
total = np.vstack((f1,f2,f3,f4,f5,f6,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17))
total = pd.DataFrame(total, columns = ['iteration', 'eval_episode_returns','agent_id'])
total = total.astype({"iteration": float, "eval_episode_returns": float, "agent_id":str })
fig, ax = plt.subplots(figsize=(16,8))
sns.lineplot(x='iteration', y='eval_episode_returns', hue='agent_id', style='agent_id', markers=True, dashes=False, data=total)
#sns.lineplot(x='iteration', y='train_episode_returns', hue='agent_id', style='agent_id', markers=True, dashes=False, palette="flare", data=total)
plt.savefig(f'fixed_{agent[index_agt]}_{env[index_en]}_20.pdf',bbox_inches='tight')
