import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import pdb
"""
Some utility Functions.
"""

def repackage_hidden_volatile(h):
    if type(h) == torch.Tensor:
        return h.requires_grad_(False)
    else:
        return tuple(repackage_hidden_volatile(v) for v in h)

def repackage_hidden(h, batch_size):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == torch.Tensor:
        return h.detach().resize_(h.size(0), batch_size, h.size(2)).zero_()
    else:
        return tuple(repackage_hidden(v, batch_size) for v in h)

def clip_gradient(model):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        p.grad.data.clamp_(-5, 5)

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    lr = lr * (0.5 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def decode_txt(itow, x):
    """Function to show decode the text."""
    out = []
    for b in range(x.size(1)):
        txt = ''
        for t in range(x.size(0)):
            idx = x[t,b]
            if idx == 0 or idx == len(itow)+1:
                break
            txt += itow[str(int(idx))]
            txt += ' '
        out.append(txt)

    return out

def l2_norm(input):
    """
    input: feature that need to normalize.
    output: normalziaed feature.
    """
    input_size = input.size()
    buffer = torch.pow(input, 2)

    normp = torch.sum(buffer, 1).add_(1e-10)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output


def sample_batch_neg(answerIdx, negAnswerIdx, sample_idx, num_sample):
    """
    input:
    answerIdx: batch_size
    negAnswerIdx: batch_size x opt.negative_sample

    output:
    sample_idx = batch_size x num_sample
    """

    batch_size = answerIdx.size(0)
    num_neg = negAnswerIdx.size(0) * negAnswerIdx.size(1)
    negAnswerIdx = negAnswerIdx.clone().view(-1)
    for b in range(batch_size):
        gt_idx = answerIdx[b]
        for n in range(num_sample):
            while True:
                rand = int(random.random() * num_neg)
                neg_idx = negAnswerIdx[rand]
                if gt_idx != neg_idx:
                    sample_idx.data[b, n] = rand
                    break

def create_joint_seq(ques, ans, stop_q, stop_a):
    '''
    Input:
    ques: batch_size X max_qlen with token id containing vocab index of each word   
    ans: batch_size X max_alen with token id containing vocab index of each word
    stop_q:  vector of length batch_size with index (between 0 and 15) corresponding to first index where the stop token occurs in the question for each batch entry
    stop_q:  vector of length batch_size with index (between 0 and 15) corresponding to first index where the stop token occurs in the answer for each batch entry

    Output: either a  padded sequence with question and answer concatenated or a batch_size X 25 dim matrix with question and answer concatenated with padding only at the START with stop token but no padding in the middle or end

    '''
    qa_joint = torch.zeros(ques.size()[0], ques.size()[1] +  ans.size()[1]).long().cuda()
    for i in range(ques.size()[0]):
        zeros = ques.size()[1] +  ans.size()[1] - (stop_q[i] + stop_a[i])
        if stop_q[i]!=0:
            qa_joint[i][zeros : zeros + stop_q[i]] = ques[i][:stop_q[i]]
        if stop_a[i]!=0:
            qa_joint[i][zeros + stop_q[i] : ] = ans[i][:stop_a[i]]
        
    return qa_joint

def create_joint_seq_gt(ques, ans, stop_a):
    '''
    Input:
    ques: batch_size X max_qlen with token id containing vocab index of each word   
    ans: batch_size X max_alen with token id containing vocab index of each word
    stop_q:  vector of length batch_size with index (between 0 and 15) corresponding to first index where the stop token occurs in the question for each batch entry
    stop_q:  vector of length batch_size with index (between 0 and 15) corresponding to first index where the stop token occurs in the answer for each batch entry

    Output: either a  padded sequence with question and answer concatenated or a batch_size X 25 dim matrix with question and answer concatenated with padding only at the START with stop token but no padding in the middle or end

    '''
    qa_joint = torch.zeros(ques.size()[0], ques.size()[1] +  ans.size()[1]).long().cuda()
    for i in range(ques.size()[0]):
        if stop_a[i]!=0:
            qa_joint[i][ ans.size()[1] - stop_a[i] : ans.size()[1] - stop_a[i]+ques.size()[1]] = ques[i]
            qa_joint[i][ ans.size()[1] - stop_a[i]+ ques.size()[1] : ]= ans[i][:stop_a[i]]
        else:
            qa_joint[i][-ques.size()[1]:] = ques[i]
        
    return qa_joint

def reverse_padding(ques, stop):
    '''
    Input:
    ques: batch_size X max_qlen with token id containing vocab index of each word   
    ans: batch_size X max_alen with token id containing vocab index of each word
    stop_q:  vector of length batch_size with index (between 0 and 15) corresponding to first index where the stop token occurs in the question for each batch entry
    stop_q:  vector of length batch_size with index (between 0 and 15) corresponding to first index where the stop token occurs in the answer for each batch entry

    Output: either a  padded sequence with question and answer concatenated or a batch_size X 25 dim matrix with question and answer concatenated with padding only at the START with stop token but no padding in the middle or end

    '''
    q_rev = torch.zeros(ques.size()[0], ques.size()[1]).long().cuda()
    for i in range(ques.size()[0]):
        zeros = ques.size()[1] - (stop[i])
        if stop[i]!=0:
            q_rev[i][zeros : zeros + stop[i]] = ques[i][:stop[i]]
        
    return q_rev

def load_my_state_dict(model, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
             continue
        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)

def compute_perplexity_batch(sentences_batch, is_log=False):    #use is_log if log probabilities given
    if not is_log:
        perplex = torch.sum(torch.log(sentences_batch))
    else:
        perplex = torch.sum(sentences_batch)
    words = sentences_batch.view(-1, 1).size(0)

    return (-1 * perplex)/ (float(words) * sentences_batch.size(0))   #Words will be used to take average

    #return torch.exp(perplex / (1.0 * words))  #Use to directly get perplexity per batch. Note cant average this over batches, use pre exponential if u want to average and then exponentiate.
