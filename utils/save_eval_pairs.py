import numpy as np
import os
import pdb

a = np.load('save_tmp.npy')
save_dir = "human_eval/q3a1/"
for i in range(4):
  im_loc = '/serverdata/akshat/'+a[-4+i,0]['img_dir']
  b = a[-4+i,:]
  np.save(save_dir+'dialogue'+str(i), b)
  os.system("cp "+im_loc + ' ' + save_dir)