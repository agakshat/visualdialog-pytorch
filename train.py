import torch
from torch.autograd import Variable
import numpy as np
import random
import pdb
from tensorboardX import SummaryWriter

def main(opt,qbots,abots,critLM,critImg,optimizerdict,dataloader,writer):
  k = 10
  for k in range(opt.num_abots):
    abots[k][0].train(),abots[k][1].train(),abots[k][2].train()
  for k in range(opt.num_qbots):
    qbots[k][0].train(),qbots[k][1].train(),qbots[k][2].train()
  

  for epoch in range(opt.start_epoch+1, opt.niter):
    if opt.k_curr is not None:
      k = opt.k_curr
    elif opt.scratch:
      if epoch>15:
        k -= 1
    elif opt.curr:
      k -= 1
    k = max(0,k)
    return k
    print("Starting Epoch: {} | K: {}".format(epoch,k))
    t = time.time()
    im_loss_epoch_n = train(epoch,k,opt,qbots,abots,critLM,critImg,optimizerdict,dataloader,writer)
    print("Finished Epoch: {} | K: {} | Time: {:.3f}".format(epoch,k,time.time()-t))


def train(epoch,k_curr,opt,qbots,abots,critLM,critImg,optimizerdict,dataloader,writer):
  n_neg = opt.negative_sample
  ques_hidden1 = abots[0][0].init_hidden(opt.batch_size)
  hist_hidden1 = abots[0][0].init_hidden(opt.batch_size)
  hist_hiddenQ = qbots[0][0].init_hidden(opt.batch_size)
  fact_hiddenQ = qbots[0][0].init_hidden(opt.batch_size) # 1x100x512 
  
  data_iter = iter(dataloader)
  error_Abot = 0
  average_lossQ = 0
  average_lossImg = 0
  count = 0
  iteration = 0
  loss_store = []
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