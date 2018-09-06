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

def train(epoch,k_curr):
    n_neg = opt.negative_sample
    ques_hidden1 = abots[0][0].init_hidden(opt.batch_size)
    hist_hidden1 = abots[0][0].init_hidden(opt.batch_size)
    hist_hiddenQ = qbots[0][0].init_hidden(opt.batch_size)
    fact_hiddenQ = qbots[0][0].init_hidden(opt.batch_size) # 1x100x512

    data_iter = iter(dataloader)

    iteration = 0
    global img_input
    prev_time = time.time()
    imloss_epoch = []
    while iteration < len(dataloader):

        abot_idx = random.randint(0,opt.num_abots-1)
        qbot_idx = random.randint(0,opt.num_qbots-1)

        data = data_iter.next()
        img_ids, img_dir, image_j, image, history, question, answer, answerT, answerLen, answerIdx, questionL, \
                                    opt_answerT, opt_answerLen, opt_answerIdx, question_input, question_target, facts, cap = data # image is 100x7x7x512, cap is 100x24
        batch_size = question.size(0)
        img_input.data.resize_(image.size()).copy_(image)
        qloss_array = []
        aloss_array = []
        imgloss_array = []
        cap = cap.t()
        cap_input.data.resize_(cap.size()).copy_(cap) # 24x100

        arr_reward = []
        arr_grad_netWA = []
        arr_grad_netWQ = []
        arr_grad_netEA = []
        arr_grad_netEQ = []
        arr_grad_netGA = []
        arr_grad_netGQ = []

        imgloss_arr_grad_netWA = []
        imgloss_arr_grad_netWQ = []
        imgloss_arr_grad_netEA = []
        imgloss_arr_grad_netEQ = []
        imgloss_arr_grad_netGA = []
        imgloss_arr_grad_netGQ = []

        arr_lpq = []
        arr_lpa = []
        old_diff_embed = 0#img_input.pow(2).mean()

        for rnd in range(k_curr):
            # get the corresponding round QA and history.
            ques_tensor = question[:,rnd,:].t()
            his = history[:,:rnd+1,:].clone().view(-1, his_length).t()

            ans = answer[:,rnd,:].t()
            tans = answerT[:,rnd,:].t()
            ques_inp = question_input[:,rnd,:].t()
            tques = question_target[:,rnd,:].t()
            fact = facts[:,rnd,:].t()

            real_len = answerLen[:,rnd].long()

            ques.data.resize_(ques_tensor.size()).copy_(ques_tensor)
            his_input.data.resize_(his.size()).copy_(his)

            ans_input.data.resize_(ans.size()).copy_(ans)
            ans_target.data.resize_(tans.size()).copy_(tans)
            ques_input.data.resize_(ques_inp.size()).copy_(ques_inp)
            fact_input.data.resize_(fact.size()).copy_(fact)
            ques_target.data.resize_(tques.size()).copy_(tques)

            # -----------------------------------------
            # update the Generator using MLE loss.
            # -----------------------------------------
            ques_emb_g = abots[abot_idx][1](ques, format = 'index')
            his_emb_g = abots[abot_idx][1](his_input, format = 'index')

            ques_embQ = qbots[qbot_idx][1](ques_input, format = 'index')
            his_embQ = qbots[qbot_idx][1](his_input, format = 'index')
            fact_embQ = qbots[qbot_idx][1](fact_input, format = 'index')

            ques_hidden1 = repackage_hidden(ques_hidden1, batch_size)
            hist_hidden1 = repackage_hidden(hist_hidden1, his_emb_g.size(1))
            fact_hiddenQ = repackage_hidden(fact_hiddenQ, fact_input.size(1))
            hist_hiddenQ = repackage_hidden(hist_hiddenQ, his_input.size(1))

            featG, ques_hidden1 = abots[abot_idx][0](ques_emb_g, his_emb_g, img_input, \
                                                ques_hidden1, hist_hidden1, rnd+1)

            _, ques_hidden1 = abots[abot_idx][2](featG.view(1, -1, opt.ninp), ques_hidden1)

            encoder_featQ, img_embedQ, fact_hiddenQ = qbots[qbot_idx][0](fact_embQ, his_embQ, \
                                            fact_hiddenQ, hist_hiddenQ, rnd+1)
            _, fact_hiddenQ = qbots[qbot_idx][2](encoder_featQ.view(1,-1,opt.ninp), fact_hiddenQ)

            # MLE loss for generator
            ans_emb = abots[abot_idx][1](ans_input)
            logprob, _ = abots[abot_idx][2](ans_emb, ques_hidden1)

            lm_loss = critLM(logprob, ans_target.view(-1, 1))
            lm_loss = lm_loss / torch.sum(ans_target.data.gt(0)).float() #  loss value for 1 round

            abots[abot_idx][1].zero_grad()
            abots[abot_idx][2].zero_grad()
            abots[abot_idx][0].zero_grad()

            lm_loss.backward()
            optimizerAbotSLarr[abot_idx].step()

            logprobQ, ques_hiddenQ = qbots[qbot_idx][2](ques_embQ, fact_hiddenQ)
            lossQ = critLM(logprobQ, ques_target.view(-1, 1))  
            lossQ /= torch.sum(ques_target.data.gt(0)).float()
            imgloss = 10.*critImg(img_embedQ,img_input.detach())
            imgloss_array.append(imgloss.item())
            qloss_array.append(lm_loss.item())
            aloss_array.append(lossQ.item())
            lossQ = lossQ + opt.image_loss_weight*imgloss # will store value of loss for 1 round
            qbots[qbot_idx][1].zero_grad()
            qbots[qbot_idx][0].zero_grad()
            qbots[qbot_idx][2].zero_grad()
            lossQ.backward()
            optimizerQbotSLarr[qbot_idx].step()

        if k_curr==0:
            rnd = -1

        if k_curr!=10:
            his = history[:,:rnd+2,:].clone().view(-1, his_length).t()
            fact = facts[:,rnd+1,:].t()

            his_input.data.resize_(his.size()).copy_(his)

            fact_input.data.resize_(fact.size()).copy_(fact)

            his_embQ = qbots[qbot_idx][1](his_input, format = 'index')
            fact_embQ = qbots[qbot_idx][1](fact_input, format = 'index')
            his_emb_g = abots[abot_idx][1](his_input, format = 'index')

            ques_hidden1 = repackage_hidden(ques_hidden1, batch_size)
            hist_hidden1 = repackage_hidden(hist_hidden1, his_emb_g.size(1))
            fact_hiddenQ = repackage_hidden(fact_hiddenQ, fact_input.size(1))
            hist_hiddenQ = repackage_hidden(hist_hiddenQ, his_input.size(1))
            encoder_featQ,img_embedQ,fact_hiddenQ = qbots[qbot_idx][0](fact_embQ,his_embQ,\
                                                            fact_hiddenQ, hist_hiddenQ, rnd+2)
            imgloss = 10.*critImg(img_embedQ,img_input.detach())
            imgloss_array.append(imgloss.item())
            qbots[qbot_idx][1].zero_grad()
            qbots[qbot_idx][0].zero_grad()
            qbots[qbot_idx][2].zero_grad()
            imgloss.backward(retain_graph=True)
            optimizerQbotSLarr[qbot_idx].step()
            old_diff_embed = (img_input-img_embedQ).pow(2).mean()

        # print("imgloss: ",imgloss_array)
        # print("qloss: ",qloss_array)
        # print("aloss: ",aloss_array)
        for rnd in range(k_curr,10):            
            sample_opt = {'beam_size':1, 'seq_length':17, 'sample_max':0}
            _,fact_hiddenQ = qbots[qbot_idx][2](encoder_featQ.view(1,-1,opt.ninp), fact_hiddenQ)
            UNK_ques_input.data.resize_((1, batch_size)).fill_(vocab_size) # 1x100
            Q,logprobQ = qbots[qbot_idx][2].sample_differentiable(qbots[qbot_idx][1],UNK_ques_input,fact_hiddenQ, sample_opt) # Q is 100x16, same logprobQ
            # fact_hiddenQ is tuple with 2 variables of size 1x100x512
            # generatedQ is float of 100x8965 and is a log-probability vector
            # with each row (of size 8965) being a (log) probability vector
            #for i in range(ques_length):
            #    Q[i] = generatedQ.multinomial()
            #Q = torch.exp(generatedQ).multinomial(ques_length,replacement=True).t()

            # Note: Do [[itow[str(Q.data[i][j])] for j in range(ques_length)] for i in range(batch_size)]
            # to see the generated questions
            Q = Q.data.t()
            Q = Q[:-1]
            Q_idx_search = (Q==vocab_size)
            _, Q_idx_search = torch.max(Q_idx_search,dim=0)
            Q_rev = reverse_padding(Q.t(), Q_idx_search.squeeze()).t()
            ques_emb_g = abots[abot_idx][1](Q_rev, format = 'index') # 16x100x300
            
            featG,ques_hidden1 = abots[abot_idx][0](ques_emb_g,his_emb_g,img_input,\
                                        ques_hidden1, hist_hidden1, rnd+1)
            # featG is float 100x300, ques_hidden1 is tuple of 2 1x100x512 floats
            
            _,ques_hidden1 = abots[abot_idx][2](featG.view(1,-1,opt.ninp),ques_hidden1)
            UNK_ans_input.data.resize_((1, batch_size)).fill_(vocab_size)
            sample_opt = {'beam_size':1, 'seq_length':9, 'sample_max':0}
            A, logprobsA = abots[abot_idx][2].sample_differentiable(abots[abot_idx][1], UNK_ans_input, ques_hidden1, sample_opt) # 100x9,100x9
            # generated A is 100x8965, and ques_hidden1 is tuple of 2 1x100x512 floats
            #A = torch.exp(generatedA).multinomial(ans_length,replacement=True).t()
            # A is long 9x100
            A = A.data.t()
            A = A[:-1]
            A_idx_search = (A==vocab_size)
            _, A_idx_search = torch.max(A_idx_search,dim=0)
            
            QA = create_joint_seq(Q.t(), A.t(), Q_idx_search.squeeze() , A_idx_search.squeeze()).t()
            fact_embA = abots[abot_idx][1](QA,format='index') # float 25x100x300
            fact_embQ = qbots[qbot_idx][1](QA,format='index') # float 25x100x300
            # do this concatenation properly by having all non-UNK tokens first followed by UNKs
            his_embQ = torch.cat((his_embQ.view(his_length, batch_size, -1, opt.ninp),fact_embQ.view(his_length, batch_size, 1, opt.ninp)),dim=2).view(his_length,-1,opt.ninp) # float <N>x100x300
            his_emb_g = torch.cat((his_emb_g.view(his_length, batch_size, -1, opt.ninp),fact_embA.view(his_length, batch_size, 1, opt.ninp)),dim=2).view(his_length,-1,opt.ninp) # float <N2>x100x300
            
            lpq = torch.sum(logprobQ,dim=1).mean()
            lpa = torch.sum(logprobsA,dim=1).mean()

            optimizerAbotRLarr[abot_idx].zero_grad()
            optimizerQbotRLarr[qbot_idx].zero_grad()

            lpq.backward(retain_graph=True)
            lpa.backward(retain_graph=True)

            arr_grad_netWA.append([p.grad.clone() for p in abots[abot_idx][1].parameters()])
            arr_grad_netWQ.append([p.grad.clone() for p in qbots[qbot_idx][1].parameters()])
            arr_grad_netEA.append([p.grad.clone() for p in abots[abot_idx][0].parameters()])
            arr_grad_netEQ.append([p.grad.clone() for p in qbots[qbot_idx][0].parameters()])
            arr_grad_netGA.append([p.grad.clone() for p in abots[abot_idx][2].parameters()])
            arr_grad_netGQ.append([p.grad.clone() for p in qbots[qbot_idx][2].parameters()])

            ques_hidden1 = repackage_hidden(ques_hidden1, batch_size)
            hist_hidden1 = repackage_hidden(hist_hidden1, his_emb_g.size(1))
            fact_hiddenQ = repackage_hidden(fact_hiddenQ, fact_embQ.size(1))
            hist_hiddenQ = repackage_hidden(hist_hiddenQ, his_emb_g.size(1))

            encoder_featQ,img_embedQ,fact_hiddenQ = qbots[qbot_idx][0](fact_embQ,his_embQ,\
                                                        fact_hiddenQ, hist_hiddenQ, rnd+2)
            diff_embed = (img_input-img_embedQ).pow(2).mean()
            arr_reward.append((old_diff_embed - diff_embed)*1000)

            imgloss_array.append(10*diff_embed.item())
            old_diff_embed.data.copy_(diff_embed.data)

            ### gradients form diff_embed shouldn't weighted by the discount factor
            optimizerQbotRLarr[qbot_idx].zero_grad()
            optimizerAbotRLarr[abot_idx].zero_grad()
            (10*diff_embed).backward(retain_graph=True)

            imgloss_arr_grad_netWA.append([p.grad.clone() for p in abots[abot_idx][1].parameters()])
            imgloss_arr_grad_netWQ.append([p.grad.clone() for p in qbots[qbot_idx][1].parameters()])
            imgloss_arr_grad_netEA.append([p.grad.clone() for p in abots[abot_idx][0].parameters()])
            imgloss_arr_grad_netEQ.append([p.grad.clone() for p in qbots[qbot_idx][0].parameters()])
            imgloss_arr_grad_netGA.append([p.grad.clone() for p in abots[abot_idx][2].parameters()])
            imgloss_arr_grad_netGQ.append([p.grad.clone() for p in qbots[qbot_idx][2].parameters()])
                    
        iteration+=1
        reward_dict = {}
        for i in range(k_curr,10):
            reward_dict[str(i)] = arr_reward[i-k_curr].data
        writer.add_scalars('RL/scalar_rewards', reward_dict, epoch*batch_size+iteration)
        img_dict = {}
        for i in range(10):
            img_dict[str(i)] = imgloss_array[i]
        writer.add_scalars('RL/scalar_imgloss', img_dict, epoch*batch_size+iteration)
        qloss_dict = {}
        aloss_dict = {}
        for i in range(k_curr):
            qloss_dict[str(i)] = qloss_array[i]
            aloss_dict[str(i)] = aloss_array[i]
        writer.add_scalars('SL/scalar_qloss', qloss_dict, epoch*batch_size+iteration)
        writer.add_scalars('SL/scalar_aloss', aloss_dict, epoch*batch_size+iteration)
        writer.export_scalars_to_json(save_path+"/all_scalars.json")
        
        if k_curr!=10:
            T =  len(arr_reward)
            G = [0]*T
            G[T-1] = arr_reward[T-1]
            for t in reversed(range(T-1)):
                G[t] = opt.gamma*G[t+1] + arr_reward[t]
            Gtensor = torch.stack(G).detach()

            fufu = [[-arr_grad_netGA[i][j]*Gtensor[i]+imgloss_arr_grad_netGA[i][j] for i in range(0,10-k_curr)] for j in range(len(arr_grad_netGA[0]))]        
            grad_ga = [torch.sum(torch.stack(fufu[j]),dim=0) for j in range(len(arr_grad_netGA[0]))]
            fufu = [[-arr_grad_netGQ[i][j]*Gtensor[i]+imgloss_arr_grad_netGQ[i][j] for i in range(0,10-k_curr)] for j in range(len(arr_grad_netGQ[0]))]        
            grad_gq = [torch.sum(torch.stack(fufu[j]),dim=0) for j in range(len(arr_grad_netGQ[0]))]
            fufu = [[-arr_grad_netWA[i][j]*Gtensor[i]+imgloss_arr_grad_netWA[i][j] for i in range(0,10-k_curr)] for j in range(len(arr_grad_netWA[0]))]        
            grad_wa = [torch.sum(torch.stack(fufu[j]),dim=0) for j in range(len(arr_grad_netWA[0]))]
            fufu = [[-arr_grad_netWQ[i][j]*Gtensor[i]+imgloss_arr_grad_netWQ[i][j] for i in range(0,10-k_curr)] for j in range(len(arr_grad_netWQ[0]))]        
            grad_wq = [torch.sum(torch.stack(fufu[j]),dim=0) for j in range(len(arr_grad_netWQ[0]))]
            fufu = [[-arr_grad_netEA[i][j]*Gtensor[i]+imgloss_arr_grad_netEA[i][j] for i in range(0,10-k_curr)] for j in range(len(arr_grad_netEA[0]))]        
            grad_ea = [torch.sum(torch.stack(fufu[j]),dim=0) for j in range(len(arr_grad_netEA[0]))]
            fufu = [[-arr_grad_netEQ[i][j]*Gtensor[i]+imgloss_arr_grad_netEQ[i][j] for i in range(0,10-k_curr)] for j in range(len(arr_grad_netEQ[0]))]        
            grad_eq = [torch.sum(torch.stack(fufu[j]),dim=0) for j in range(len(arr_grad_netEQ[0]))]

            for idx,p in enumerate(abots[abot_idx][0].parameters()):
                p.grad = grad_ea[idx]
            for idx,p in enumerate(qbots[qbot_idx][0].parameters()):
                p.grad = grad_eq[idx]
            for idx,p in enumerate(abots[abot_idx][2].parameters()):
                p.grad = grad_ga[idx]
            for idx,p in enumerate(qbots[qbot_idx][2].parameters()):
                p.grad = grad_gq[idx]
            for idx,p in enumerate(abots[abot_idx][1].parameters()):
                p.grad = grad_wa[idx]
            for idx,p in enumerate(qbots[qbot_idx][1].parameters()):
                p.grad = grad_wq[idx]

            torch.nn.utils.clip_grad_norm_(abots[abot_idx][0].parameters(),2)
            torch.nn.utils.clip_grad_norm_(abots[abot_idx][1].parameters(),2)
            torch.nn.utils.clip_grad_norm_(abots[abot_idx][2].parameters(),2)
            torch.nn.utils.clip_grad_norm_(qbots[qbot_idx][0].parameters(),2)
            torch.nn.utils.clip_grad_norm_(qbots[qbot_idx][1].parameters(),2)
            torch.nn.utils.clip_grad_norm_(qbots[qbot_idx][2].parameters(),2)

            optimizerQbotRLarr[qbot_idx].step()
            optimizerAbotRLarr[abot_idx].step()

        if iteration%20==0:
            print("Done with Batch # {} | Av. Time Per Batch: {:.3f}s".format(iteration,(time.time()-prev_time)/20))
            prev_time = time.time()
            imloss_epoch.append(imgloss_array[0])

    return np.mean(imloss_epoch)

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

    n_elts = 0
    save_tmp = []

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

        n_elts += 1

        for rnd in range(0,10):
            sample_opt = {'beam_size':1, 'seq_length':17, 'sample_max':0}
            _,fact_hiddenQ = qbots[0][2](encoder_featQ.view(1,-1,opt.ninp), fact_hiddenQ)
            UNK_ques_input.data.resize_((1, batch_size)).fill_(vocab_size) # 1x100
            Q,logprobQ = qbots[0][2].sample_differentiable(qbots[0][1],UNK_ques_input,fact_hiddenQ, sample_opt) # Q is 100x16, same logprobQ

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
            for j in range(batch_size):
              save_tmp[j].append({'sample_ques':ques_sample_txt[j], \
                       'sample_ans':ans_sample_txt[j], 'rnd':rnd})

        # If you want to save the predictions in a npy file, uncomment the next line
        # np.save(save_path+'/predictions'+str(opt.num_qbots)+'Q'+str(opt.num_abots)+'A.npy',save_tmp)
        
        iteration+=1
        
        # For obtaining image retrieval percentile
        reward_dict = {}
        for i, elt in enumerate(embed_array):
          for img_no in range(elt.size(0)):
              img = elt[img_no]
              l2_dist = torch.sum(((imgs_tensor - img) ** 2), 1)
              img_diff = torch.sum((img-img_input[img_no])**2)
              found_rank = torch.sum((l2_dist <= img_diff).float())
              mean_rank[(iteration-1)*opt.batch_size+img_no,i] += found_rank

        if iteration%2==0:
          print("Done with Batch # {} | Av. Time Per Batch: {:.3f}s".format(iteration,(time.time()-prev_time)/20))
          prev_time = time.time()
          print("Image Retrieval Rank {}".format(mean_rank[:(iteration-1)*opt.batch_size+img_no].mean(axis=0)))

    np.save('image_retrieval_rank'+str(opt.num_qbots)+str(opt.num_abots),mean_rank)
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

if not opt.eval:
    optimizerAbotSLarr = []
    optimizerQbotSLarr = []
    optimizerAbotRLarr = []
    optimizerQbotRLarr = []
    for j in range(opt.num_abots):
        optimizerAbotSLarr.append(optim.Adam([{'params': abots[j][1].parameters()},
                              {'params': abots[j][0].parameters()},
                              {'params': abots[j][2].parameters()}], lr=opt.A_lr, betas=(opt.beta1, 0.999)))

    for j in range(opt.num_qbots):
        optimizerQbotSLarr.append(optim.Adam([{'params': qbots[j][1].parameters()},
                              {'params': qbots[j][0].parameters()},
                              {'params': qbots[j][2].parameters()}], lr=opt.Q_lr, betas=(opt.beta1, 0.999)))

    for j in range(opt.num_abots):
        optimizerAbotRLarr.append(optim.Adam([{'params': abots[j][1].parameters()},
                              {'params': abots[j][0].parameters()},
                              {'params': abots[j][2].parameters()}], lr=opt.A_lr, betas=(opt.beta1, 0.999)))

    for j in range(opt.num_qbots):
        optimizerQbotRLarr.append(optim.Adam([{'params': qbots[j][1].parameters()},
                              {'params': qbots[j][0].parameters()},
                              {'params': qbots[j][2].parameters()}], lr=opt.Q_lr, betas=(opt.beta1, 0.999)))

    for k in range(opt.num_abots):
        abots[k][0].train(),abots[k][1].train(),abots[k][2].train()
    for k in range(opt.num_qbots):
        qbots[k][0].train(),qbots[k][1].train(),qbots[k][2].train()

    k_curr = 10 if opt.start_curr is None else opt.start_curr+1
    for epoch in range(opt.start_epoch+1, opt.niter):
        if opt.k_curr is not None:
            assert not opt.scratch and not opt.curr and opt.start_curr is None, "Don't provide any --curr flags if you want training with fixed K"
            k_curr = opt.k_curr
        elif opt.scratch:
            assert opt.start_curr is None, "Don't give --start_curr flag if you want to train from scratch"
            if epoch>15:
                k_curr -= 1
        elif opt.curr or opt.start_curr is not None:
            assert not opt.scratch and opt.k_curr is None, "Don't give --scratch or --k_curr flag if you want curriculum training"
            k_curr -= 1/3
        else:
            print("No Training Method provided, assuming default is curriculum starting from k=10")
            k_curr -= 1

        k_curr = int(max(0,k_curr))
        print("Starting Epoch: {} | K: {}".format(epoch,k_curr))
        t = time.time()
        im_loss_epoch_n = train(epoch,k_curr)
        print("Finished Epoch: {} | K: {} | Time: {:.3f}".format(epoch,k_curr,time.time()-t))

        if epoch%opt.save_iter == 0 and not opt.no_save:
            savedict = {'epoch': epoch,
                    'k': k,
                    'opt': opt}
            for k in range(len(abots)):
                savedict['netWA'+str(k)] = abots[k][1].state_dict()
                savedict['netEA'+str(k)] = abots[k][0].state_dict()
                savedict['netGA'+str(k)] = abots[k][2].state_dict()
            for k in range(len(qbots)):
                savedict['netWQ'+str(k)] = qbots[k][1].state_dict()
                savedict['netEQ'+str(k)] = qbots[k][0].state_dict()
                savedict['netGQ'+str(k)] = qbots[k][2].state_dict()
            
            torch.save(savedict,'%s/epoch_%d.pth' % (save_path, epoch))

            print("Epoch Done, Saved Model")

else:
    for k in range(opt.num_abots):
        abots[k][0].eval(),abots[k][1].eval(),abots[k][2].eval()
    for k in range(opt.num_qbots):
        qbots[k][0].eval(),qbots[k][1].eval(),qbots[k][2].eval()
    evaluate()
