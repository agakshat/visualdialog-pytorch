from __future__ import print_function
import argparse
import os
import random
import sys
sys.path.append(os.getcwd())

import pdb
import time
import numpy as np
import datetime
from tensorboardX import SummaryWriter
import copy

import torch
import torch.optim as optim

from utils.tools import repackage_hidden, decode_txt, sample_batch_neg, create_joint_seq, \
                    create_joint_seq_gt, reverse_padding, load_my_state_dict, compute_perplexity_batch
from utils.create_dataloader import create_dataloader
import networks.model as model
from networks.encoder_QIH import _netE
from networks.netG import _netG
from networks.encoder_QBot import _netE as _netQE
from arguments import get_args

def evaluate():
    n_neg = opt.negative_sample
    ques_hidden1 = abots[0][0].init_hidden(opt.batch_size)
    hist_hidden1 = abots[0][0].init_hidden(opt.batch_size)
    hist_hiddenQ = qbots[0][0].init_hidden(opt.batch_size)
    fact_hiddenQ = qbots[0][0].init_hidden(opt.batch_size) 
    data_iter = iter(dataloader_val)
    imgs = dataset.imgs

    iteration = 0
    global img_input
    prev_time = time.time()
    mean_rank = np.zeros((len(dataloader_val)*opt.batch_size,11))

    imgs_tensor = torch.FloatTensor(imgs)
    imgs_tensor = imgs_tensor.cuda()

    #Used to track metrics
    mean_ranks = torch.zeros(10)
    r1 = torch.zeros(10)
    r5 = torch.zeros(10)
    r10 = torch.zeros(10)
    mean_rec_ranks = torch.zeros(10)
    q_perplex = torch.zeros(10)
    a_perplex = torch.zeros(10)
    n_elts = 0
    num_samples = 0
    while iteration < len(dataloader_val):
        embed_array = []
        data = data_iter.next()
        img_ids, img_dir, image_j, image, history, question, answer, answerT, answerLen, answerIdx, questionL, \
                                  opt_answerT, opt_answerLen, opt_answerIdx, question_input, question_target, facts, cap = data # image is 100x7x7x512, cap is 100x24
        batch_size = question.size(0)
        img_input1.data.resize_(image_j.size()).copy_(image_j)
        img_input.data.resize_(image.size()).copy_(image)
        cap = cap.t()
        cap_input.data.resize_(cap.size()).copy_(cap) # 24x100

        arr_reward = []
        save_tmp = [[] for j in range(batch_size)]
        cap_sample_txt = decode_txt(itow, cap)
        
        for j in range(batch_size):
            save_tmp[j].append({'caption':cap_sample_txt[j], 'img_ids': img_ids[j], 'img_dir':img_dir[j]})
        
        his = history[:,:1,:].clone().view(-1, his_length).t()
        fact = facts[:,0,:].t()
        his_input.data.resize_(his.size()).copy_(his)
        fact_input.data.resize_(fact.size()).copy_(fact)

        his_embQ = qbots[0][1](his_input, format = 'index')
        fact_embQ = qbots[0][1](fact_input, format = 'index')
        his_emb_g = abots[0][1](his_input, format = 'index')

        ques_hidden1 = repackage_hidden(ques_hidden1, batch_size)
        hist_hidden1 = repackage_hidden(hist_hidden1, his_emb_g.size(1))
        fact_hiddenQ = repackage_hidden(fact_hiddenQ, fact_input.size(1))
        hist_hiddenQ = repackage_hidden(hist_hiddenQ, his_input.size(1))
        encoder_featQ,img_embedQ,fact_hiddenQ = qbots[0][0](fact_embQ,his_embQ,\
                                                     fact_hiddenQ, hist_hiddenQ, 1)
        embed_array.append(img_embedQ)
        num_samples += batch_size
        n_elts += 1

        for rnd in range(0,10):
            sample_opt = {'beam_size':1, 'seq_length':17, 'sample_max':0}
            _,fact_hiddenQ = qbots[0][2](encoder_featQ.view(1,-1,opt.ninp), fact_hiddenQ)
            UNK_ques_input.data.resize_((1, batch_size)).fill_(vocab_size) # 1x100
            Q,logprobQ = qbots[0][2].sample_differentiable(qbots[0][1],UNK_ques_input,fact_hiddenQ, sample_opt) # Q is 100x16, same logprobQ

            sample_opt_per = {'beam_size':1, 'seq_length':17, 'sample_max':1}
            Q_per,logprobQ_per = qbots[0][2].sample_differentiable(qbots[0][1],UNK_ques_input,fact_hiddenQ, sample_opt_per) # Q is 100x16, same logprobQ
            q_perplex[rnd] += compute_perplexity_batch(logprobQ_per.cpu(), is_log=True).item()

            Q = Q.data.t()
            Q = Q[:-1]
            Q_idx_search = (Q==vocab_size)
            _, Q_idx_search = torch.max(Q_idx_search,dim=0)
            Q_rev = reverse_padding(Q.t(), Q_idx_search.squeeze()).t()
            ques_emb_g = abots[0][1](Q_rev, format = 'index') # 16x100x300

            featG,ques_hidden1 = abots[0][0](ques_emb_g,his_emb_g,img_input,\
                                      ques_hidden1, hist_hidden1, rnd+1)
            # featG is float 100x300, ques_hidden1 is tuple of 2 1x100x512 floats
            _,ques_hidden1 = abots[0][2](featG.view(1,-1,opt.ninp),ques_hidden1)
            UNK_ans_input.data.resize_((1, batch_size)).fill_(vocab_size)
            sample_opt = {'beam_size':1, 'seq_length':9, 'sample_max':0}
            A, logprobsA = abots[0][2].sample_differentiable(abots[0][1], UNK_ans_input, ques_hidden1, sample_opt) # 100x9,100x9
            
            sample_opt_per = {'beam_size':1, 'seq_length':9, 'sample_max':1}
            A_per, logprobsA_per = abots[0][2].sample_differentiable(abots[0][1], UNK_ans_input, ques_hidden1, sample_opt_per) # 100x9,100x9
            a_perplex[rnd] += compute_perplexity_batch(logprobsA_per.cpu(), is_log=True).item()

            mrank, m_rec, batch_r1, batch_r5, batch_r10 = abots[0][2].sample_opt_eval(abots[0][1], UNK_ans_input, ques_hidden1, answerT, opt_answerT, rnd, sample_opt) # 100x9,100x9
            mean_ranks[rnd] += mrank
            mean_rec_ranks[rnd] += m_rec
            r1[rnd] += batch_r1.float()
            r5[rnd] += batch_r5.float()
            r10[rnd] += batch_r10.float()

            A = A.data.t()
            A = A[:-1]

            A_idx_search = (A==vocab_size)
            _, A_idx_search = torch.max(A_idx_search,dim=0)
            QA = create_joint_seq(Q.t(), A.t(), Q_idx_search.squeeze() , A_idx_search.squeeze()).t()
            fact_embA = abots[0][1](QA,format='index') # float 25x100x300
            fact_embQ = qbots[0][1](QA,format='index') # float 25x100x300
            # do this concatenation properly by having all non-UNK tokens first followed by UNKs
            his_embQ = torch.cat((his_embQ.view(his_length, batch_size, -1, opt.ninp),fact_embQ.view(his_length, batch_size, 1, opt.ninp)),dim=2).view(his_length,-1,opt.ninp) # float <N>x100x300
            his_emb_g = torch.cat((his_emb_g.view(his_length, batch_size, -1, opt.ninp),fact_embA.view(his_length, batch_size, 1, opt.ninp)),dim=2).view(his_length,-1,opt.ninp) # float <N2>x100x300
            
            ques_hidden1 = repackage_hidden(ques_hidden1, batch_size)
            hist_hidden1 = repackage_hidden(hist_hidden1, his_emb_g.size(1))
            fact_hiddenQ = repackage_hidden(fact_hiddenQ, fact_embQ.size(1))
            hist_hiddenQ = repackage_hidden(hist_hiddenQ, his_emb_g.size(1))

            encoder_featQ,img_embedQ,fact_hiddenQ = qbots[0][0](fact_embQ,his_embQ,\
                                                     fact_hiddenQ, hist_hiddenQ, rnd+2)

            embed_array.append(img_embedQ)

            ans_sample_txt = decode_txt(itow, A)
            ques_sample_txt = decode_txt(itow, Q)

        iteration+=1
        if iteration%2==0:
          print("Done with Batch # {} | Av. Time Per Batch: {:.3f}s".format(iteration,(time.time()-prev_time)/20))
          prev_time = time.time()
          mean_rec_rank_final = mean_rec_ranks/ float(n_elts)
          mean_rank_final = mean_ranks/ float(n_elts)
          print("MRR: ",mean_rec_rank_final,torch.mean(mean_rec_rank_final))
          print("mean_rank",mean_rank_final,torch.mean(mean_rank_final))
    
    mean_rank_final = mean_ranks/ float(n_elts)
    mean_rec_rank_final = mean_rec_ranks/ float(n_elts)
    r1_final = 100*r1/float(num_samples)
    r5_final = 100*r5/float(num_samples)
    r10_final = 100*r10/float(num_samples)
    np.save('mean_rank'+str(opt.num_qbots)+str(opt.num_abots),mean_rank_final.cpu().numpy())
    np.save('MRR'+str(opt.num_qbots)+str(opt.num_abots),mean_rec_rank_final.cpu().numpy())
    np.save('r1'+str(opt.num_qbots)+str(opt.num_abots),r1_final.cpu().numpy())
    np.save('r5'+str(opt.num_qbots)+str(opt.num_abots),r5_final.cpu().numpy())
    np.save('r10'+str(opt.num_qbots)+str(opt.num_abots),r10_final.cpu().numpy())
    return


##############################
# Main Code Execution Starts Here
##############################

opt = get_args()
opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

t = datetime.datetime.now()
cur_time = '%s-%s-%s' %(t.day, t.month, t.hour)
save_path = os.path.join(opt.outf, cur_time)
try:
    os.makedirs(save_path)
except OSError:
    pass

dataset,dataset_val,dataloader,dataloader_val = create_dataloader(opt)
writer = SummaryWriter(save_path)

vocab_size = dataset.vocab_size
ques_length = dataset.ques_length
ans_length = dataset.ans_length + 1
his_length = dataset.ans_length  + dataset.ques_length
cap_length = dataset.cap_length
itow = dataset.itow

qbots = []
abots = []
print('Initializing A-Bot and Q-Bot...')
for j in range(opt.num_abots):
    abots.append((_netE(opt.model, opt.ninp, opt.nhid, opt.nlayers, opt.dropout, opt.img_feat_size),
                 model._netW(vocab_size, opt.ninp, opt.dropout),
                _netG(opt.model, vocab_size, opt.ninp, opt.nhid, opt.nlayers, opt.dropout)))
for j in  range(opt.num_qbots):
    qbots.append((_netQE(opt.model, opt.ninp, opt.nhid, opt.nlayers, opt.dropout),
                model._netW(vocab_size, opt.ninp, opt.dropout),
                _netG(opt.model, vocab_size, opt.ninp, opt.nhid, opt.nlayers, opt.dropout)))
critLM = model.LMCriterion()
critImg = torch.nn.MSELoss()

if opt.model_path != '':
    print("=> loading checkpoint '{}'".format(opt.model_path))
    checkpoint = torch.load(opt.model_path, map_location=lambda storage, loc: storage)

    if opt.scratch:
        pass
    elif opt.curr:
        print('Loading A and Q-Bots from SL')
        for j in range(opt.num_abots):
            load_my_state_dict(abots[j][1],checkpoint['netWA'])
            load_my_state_dict(abots[j][0],checkpoint['netEA'])
            load_my_state_dict(abots[j][2],checkpoint['netGA'])
        for j in range(opt.num_qbots):
            load_my_state_dict(qbots[j][1],checkpoint['netWQ'])
            load_my_state_dict(qbots[j][0],checkpoint['netEQ'])
            load_my_state_dict(qbots[j][2],checkpoint['netGQ'])
    else:
        print('Loading A and Q-Bots from RL')
        for j in range(opt.num_abots):
            load_my_state_dict(abots[j][1],checkpoint['netWA'+str(j)])
            load_my_state_dict(abots[j][0],checkpoint['netEA'+str(j)])
            load_my_state_dict(abots[j][2],checkpoint['netGA'+str(j)])
        for j in range(opt.num_qbots):
            load_my_state_dict(qbots[j][1],checkpoint['netWQ'+str(j)])
            load_my_state_dict(qbots[j][0],checkpoint['netEQ'+str(j)])
            load_my_state_dict(qbots[j][2],checkpoint['netGQ'+str(j)])

else:
    assert not opt.eval and opt.scratch, "Must specify model files if not starting evaluating or training from scratch"

if opt.cuda: # ship to cuda, if has GPU
    for k in range(opt.num_abots):
        abots[k][0].cuda(),abots[k][1].cuda(),abots[k][2].cuda()
    for k in range(opt.num_qbots):
        qbots[k][0].cuda(),qbots[k][1].cuda(),qbots[k][2].cuda()
    critLM.cuda(), critImg.cuda()

################
img_input = torch.FloatTensor(opt.batch_size).requires_grad_()
img_input1 = torch.FloatTensor(opt.batch_size).requires_grad_()
cap_input = torch.LongTensor(opt.batch_size).requires_grad_()
ques = torch.LongTensor(ques_length, opt.batch_size).requires_grad_()
ques_input = torch.LongTensor(ques_length, opt.batch_size).requires_grad_()
his_input = torch.LongTensor(his_length, opt.batch_size).requires_grad_()
fact_input = torch.LongTensor(ques_length+ans_length,opt.batch_size).requires_grad_()
ans_input = torch.LongTensor(ans_length, opt.batch_size).requires_grad_()
ans_target = torch.LongTensor(ans_length, opt.batch_size).requires_grad_()
wrong_ans_input = torch.LongTensor(ans_length, opt.batch_size).requires_grad_()
UNK_ans_input = torch.LongTensor(1, opt.batch_size).requires_grad_()
UNK_ques_input = torch.LongTensor(1, opt.batch_size).requires_grad_()
ques_target = torch.LongTensor(ques_length,opt.batch_size).requires_grad_()

if opt.cuda:
    ques, ques_input, his_input, img_input, img_input1, cap_input = ques.cuda(), ques_input.cuda(), his_input.cuda(), img_input.cuda(), img_input.cuda(), cap_input.cuda()
    ans_input, ans_target = ans_input.cuda(), ans_target.cuda()
    ques_target = ques_target.cuda()
    UNK_ans_input = UNK_ans_input.cuda()
    UNK_ques_input = UNK_ques_input.cuda()
    fact_input = fact_input.cuda()

##################

for k in range(opt.num_abots):
    abots[k][0].eval(),abots[k][1].eval(),abots[k][2].eval()
for k in range(opt.num_qbots):
  qbots[k][0].eval(),qbots[k][1].eval(),qbots[k][2].eval()
evaluate()
