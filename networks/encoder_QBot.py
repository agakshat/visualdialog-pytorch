import torch
import torch.nn as nn
import pdb
import math
import numpy as np
import torch.nn.functional as F

class _netE(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ninp, nhid, nlayers, dropout): #nhid should be 512
        super(_netE, self).__init__()

        self.d = dropout
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.nhid = nhid
        self.ninp = ninp # because fact = answer + question

        self.cap_encoder = getattr(nn, rnn_type)(self.ninp, self.nhid, nlayers, dropout=self.d)

        self.fact_rnn = getattr(nn, rnn_type)(self.ninp, self.nhid, nlayers, dropout=self.d)
        self.his_rnn = getattr(nn, rnn_type)(self.ninp, self.nhid, nlayers, dropout=self.d)
        self.Wq_1 = nn.Linear(self.nhid, self.nhid)
        self.Wh_1 = nn.Linear(self.nhid, self.nhid)
        self.Wa_1 = nn.Linear(self.nhid, 1)

        self.fc1 = nn.Linear(3*self.nhid, 512)
        self.fc2 = nn.Linear(3*self.nhid, self.ninp)
        self.fc3 = nn.Linear(512,4096)

    def forward(self, fact_emb, hist_emb, fact_hidden, his_hidden, rnd):

        fact_feat, fact_hidden = self.fact_rnn(fact_emb, fact_hidden)
        fact_feat = fact_feat[-1]
        batch_size = fact_emb.size()[1]

        cap = (hist_emb.clone()).view(hist_emb.size(0), -1, rnd, self.ninp)
        
        hist_emb = hist_emb.view(hist_emb.size(0), -1, rnd, self.ninp).clone()
        hist_emb[:, :, 0, :] = 0


        cap = cap[:, :, 0, :]
        cap_emb, _ = self.cap_encoder(cap)
        cap_emb = cap_emb[-1, :, :]
        his_feat, _ = self.his_rnn(hist_emb.view(hist_emb.size(0), -1, self.ninp), his_hidden)
        his_feat = his_feat[-1]
        fact_emb_1 = self.Wq_1(fact_feat).view(-1, 1, self.nhid)
        his_emb_1 = self.Wh_1(his_feat).view(-1, rnd, self.nhid)

        
        atten_emb_1 = F.tanh(his_emb_1 + fact_emb_1.expand_as(his_emb_1))

        his_atten_weight = F.softmax(self.Wa_1(F.dropout(atten_emb_1, self.d, training=self.training
                                                ).view(-1, self.nhid)).view(-1, rnd),dim=1)
        his_attn_feat = torch.bmm(his_atten_weight.view(-1, 1, rnd),his_feat.view(-1, rnd, self.nhid))

        his_attn_feat = his_attn_feat.view(-1, self.nhid)
        concat_feat = torch.cat((fact_feat, his_attn_feat.view(-1, self.nhid)),1)
        concat_feat = torch.cat((concat_feat, cap_emb), 1)

        encoder_feat = F.tanh(self.fc2(F.dropout(concat_feat, self.d, training=self.training)))
        img_embedding = self.fc3(F.relu(self.fc1(concat_feat)))
        fact_feat = fact_feat.unsqueeze(0)

        return encoder_feat, img_embedding, fact_hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (weight.new(self.nlayers, bsz, self.nhid).zero_(),
                    weight.new(self.nlayers, bsz, self.nhid).zero_())
        else:
            return weight.new(self.nlayers, bsz, self.nhid).zero_()
