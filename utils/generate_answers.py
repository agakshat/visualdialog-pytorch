import numpy as np
import torch
import os
import shutil
import pdb

fpath = 'save/v2_temp/14-8-17/'
cocopath = '/serverdata/akshat/'
files = ['predictions_SL1Q1A.npy','predictions1Q1A.npy','predictions1Q3A.npy','predictions3Q1A.npy']
loads = [np.load(fpath+f) for f in files]
alphadir = {0:'A',1:'B',2:'C',3:'D'}
for idx in range(loads[0].shape[0]):
  os.mkdir(fpath+'evalsamples/'+str(idx))
  shutil.copy(cocopath+loads[0][idx][0]['img_dir'],fpath+'evalsamples/'+str(idx)+'/img.jpg')
  arr = ''
  arr += 'Caption: %s \n\n'%(loads[0][idx][0]['caption'])
  for k,l in enumerate(loads):
    arr += 'System: %s \n'%(alphadir[k])
    for j in range(1,11):
      #pdb.set_trace()
      arr += 'Round: %s | Question: %s | Answer: %s \n'%(l[idx][j]['rnd'],l[idx][j]['sample_ques'],l[idx][j]['sample_ans'])
    arr += '\n\n'
    #arr.append({'System': k,'Round':loads[l][idx][j]['rnd'],'Question':loads[l][idx][j]['sample_ques'],'Answer':loads[l][idx][j]['sample_ans']} for j in range(1:11)])
  np.savetxt(fpath+'evalsamples/'+str(idx)+'/dialogs.txt',[arr],fmt="%s")
