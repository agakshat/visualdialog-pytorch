import torch.utils.data as data
from PIL import Image
import torch
import numpy as np
import h5py
import json
import pdb
import random
from utils.tools import repackage_hidden, clip_gradient, adjust_learning_rate, decode_txt

class train(data.Dataset): # torch wrapper
    def __init__(self, input_img_h5, input_ques_h5, input_json, v09_img, v09_param, negative_sample, num_val, data_split):

        print('DataLoader loading: %s' %data_split)
        print('Loading image feature from %s' %input_img_h5)

        if data_split == 'test':
            split = 'val'
        else:
            split = 'train' # train and val split both corresponding to 'train'

        f = json.load(open(input_json, 'r')) # has img_train, img_val and itow
        self.itow = f['itow'] # dict with 8963 unique words, and their mapping?
        self.img_info = f['img_'+split] # has image id number, path in COCO dataset

        # get the data split.

        # load the data.
        f = h5py.File(input_img_h5, 'r')
        f2 = h5py.File(v09_img)['images_'+split]
        f2ids = json.load(open(v09_param))['unique_img_'+split]
        f2idsa = np.argsort(np.array(f2ids))
        f1ids = []
        f1dirs = []
        for x in self.img_info:
            f1ids.append(x['imgId'])
            f1dirs.append(x['path'])
        f1idsa = np.argsort(np.array(f1ids))
        self.f1ids = np.array(f1ids)
        self.f1dirs = np.array(f1dirs)
        self.imgs = np.zeros(f2.shape)
        self.imgs[f1idsa] = np.array(f2)[f2idsa]

        total_num = len(self.img_info) # 82783 for train, 40504 for val
        if data_split == 'train':
            s = 0
            e = total_num - num_val
        elif data_split == 'val':
            s = total_num - num_val
            e = total_num
        else:
            s = 0
            e = total_num
        print('%s number of data: %d' %(data_split, e-s))
       
        self.img_info = self.img_info[s:e]
        self.imgs = self.imgs[s:e]
        self.imgs_j = f['images_'+split][s:e]
        f.close()

        print('Loading txt from %s' %input_ques_h5)
        f = h5py.File(input_ques_h5, 'r')
        self.ques = f['ques_'+split][s:e] # h5py._hl.dataset.DataSet array of size 82783x10x16
        self.ans = f['ans_'+split][s:e] # size 82783x10x8
        self.cap = f['cap_'+split][s:e] # size 82783x24

        self.ques_len = f['ques_len_'+split][s:e]
        self.ans_len = f['ans_len_'+split][s:e]
        self.cap_len = f['cap_len_'+split][s:e]

        self.ans_ids = f['ans_index_'+split][s:e]
        self.opt_ids = f['opt_'+split][s:e]
        self.opt_list = f['opt_list_'+split][:]
        self.opt_len = f['opt_len_'+split][:]
        f.close()

        self.ques_length = self.ques.shape[2]
        self.ans_length = self.ans.shape[2]
        self.his_length = self.ques_length + self.ans_length
        self.vocab_size = len(self.itow)+1 # 8964
        self.cap_length = self.cap.shape[1]

        print('Vocab Size: %d' % self.vocab_size)
        self.split = split
        self.rnd = 10
        self.negative_sample = negative_sample

        
    def __getitem__(self, index):
        # get the image
        img_dir = self.f1dirs[index]
        img_ids = self.f1ids[index]
        img = torch.from_numpy(self.imgs[index])
        img_j = torch.from_numpy(self.imgs_j[index])
        cap = torch.from_numpy(self.cap[index].astype(float))
        # get the history
        his = np.zeros((self.rnd, self.his_length))
        his[0,self.his_length-self.cap_len[index]:] = self.cap[index,:self.cap_len[index]]

        ques = np.zeros((self.rnd, self.ques_length))
        ques_inp = np.zeros((self.rnd, self.ques_length+1))
        ques_target = np.zeros((self.rnd, self.ques_length+1))
        ans = np.zeros((self.rnd, self.ans_length+1))
        ans_target = np.zeros((self.rnd, self.ans_length+1))
        ques_ori = np.zeros((self.rnd, self.ques_length))
        fact = np.zeros((self.rnd, self.ques_length+self.ans_length))

        opt_ans = np.zeros((self.rnd, self.negative_sample, self.ans_length+1))
        ans_len = np.zeros((self.rnd))
        opt_ans_len = np.zeros((self.rnd, self.negative_sample))

        ans_idx = np.zeros((self.rnd))
        opt_ans_idx = np.zeros((self.rnd, self.negative_sample))

        for i in range(self.rnd):
            # get the index
            q_len = self.ques_len[index, i]
            a_len = self.ans_len[index, i]
            qa_len = q_len + a_len

            if i+1 < self.rnd:
                his[i+1, self.his_length-qa_len:self.his_length-a_len] = self.ques[index, i, :q_len]
                his[i+1, self.his_length-a_len:] = self.ans[index, i, :a_len]
                fact[i+1, self.his_length-qa_len:self.his_length-a_len] = self.ques[index, i, :q_len]
                fact[i+1, self.his_length-a_len:] = self.ans[index, i, :a_len]

            ques[i, self.ques_length-q_len:] = self.ques[index, i, :q_len]

            ques_inp[i, 1:q_len+1] = self.ques[index, i, :q_len]
            ques_inp[i, 0] = self.vocab_size

            ques_target[i, :q_len] = self.ques[index, i, :q_len]
            ques_target[i, q_len] = self.vocab_size

            ques_ori[i, :q_len] = self.ques[index, i, :q_len]  

            ans[i, 1:a_len+1] = self.ans[index, i, :a_len]
            ans[i, 0] = self.vocab_size

            ans_target[i, :a_len] = self.ans[index, i, :a_len]
            ans_target[i, a_len] = self.vocab_size
            ans_len[i] = self.ans_len[index, i]

            ##########This concatenation needs to be looked into more carefully############
            ########## Add the first fact as zeros for generating first Question###########
            #fact[i, :q_len] = self.ques[index, i, :q_len]
            #fact[i, q_len] =  self.vocab_size
            #fact[i, q_len:q_len+a_len] = self.ans[index, i, :a_len]
            
            opt_ids = self.opt_ids[index, i] # since python start from 0
            # random select the negative samples.
            ans_idx[i] = opt_ids[self.ans_ids[index, i]]
            # exclude the gt index.
            opt_ids = np.delete(opt_ids, ans_idx[i], 0)
            random.shuffle(opt_ids)
            for j in range(self.negative_sample):
                ids = opt_ids[j]
                opt_ans_idx[i,j] = ids

                opt_len = self.opt_len[ids]

                opt_ans_len[i, j] = opt_len
                opt_ans[i, j, :opt_len] = self.opt_list[ids,:opt_len]
                opt_ans[i, j, opt_len] = self.vocab_size

        his = torch.from_numpy(his)
        ques = torch.from_numpy(ques)
        ans = torch.from_numpy(ans)
        ans_target = torch.from_numpy(ans_target)
        ques_ori = torch.from_numpy(ques_ori)
        ans_len = torch.from_numpy(ans_len)
        opt_ans_len = torch.from_numpy(opt_ans_len)
        opt_ans = torch.from_numpy(opt_ans)
        ans_idx = torch.from_numpy(ans_idx)
        opt_ans_idx = torch.from_numpy(opt_ans_idx)
        ques_inp = torch.from_numpy(ques_inp)
        ques_target = torch.from_numpy(ques_target)
        fact = torch.from_numpy(fact)
        return img_ids, img_dir, img_j, img, his, ques, ans, ans_target, ans_len, ans_idx, ques_ori, \
                opt_ans, opt_ans_len, opt_ans_idx, ques_inp, ques_target, fact, cap

    def __len__(self):
        return self.ques.shape[0]