

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np


class Discriminator(nn.Module):
    def __init__(self, in_dim=500,train=False):
        super(Discriminator, self).__init__()
        self.tot_infs = 0.0
        self.tot_infs_nor = 0.0
        self.train = train
        self.x_threshold = 0.001
        self.inf_scale = 100.0
        self.Zstar = 1000
        
        self.in_dim = in_dim

        np.random.seed(0)
        self.Omega = Variable(torch.from_numpy(risk_mat)).float()

        self.card_upper = 70.
        self.card_lower = 69.
        
        
    def forward(self, x):
        tot_infs = Variable(torch.zeros(x.size()[0]))
        x_bw = Variable(torch.from_numpy(bw).float())
        alp = Variable(torch.Tensor(alpha).float())   
        bet = Variable(torch.Tensor(beta).float())
 
        # (4)
        tot_infs += F.relu(1. - torch.sum(x,1))
        tot_infs += F.relu(torch.sum(x,1) - 1.)
        
        # (5)
        tot_infs += F.relu(torch.abs(x-x_bw)@ Variable(torch.ones(self.in_dim))-0.05)

        # (6)
        for j in range(nbsector):
            k= (x-x_bw) @ Variable(
                torch.Tensor([1 if i in group_by_sector[j] else 0 for i in range(self.in_dim)]).float())  #sum
            l = torch.abs(k)
            tot_infs += F.relu(l - 0.1)
            
        # (7)
        for j in range(nbmq):
            k = (x-x_bw) @ Variable(
                torch.Tensor([1 if i in group_by_mq[j] else 0 for i in range(self.in_dim)]).float())  #sum
            l = torch.abs(k)
            tot_infs += F.relu(l - 0.1)
            
        # (8)
        k = (x - x_bw)@bet
        l = torch.abs(k)
        tot_infs += F.relu(l - 0.1)
        
        # (9) lb <= card(x) <= b
        if self.train:
            num_non_zeros = torch.sum(torch.tanh(x / self.x_threshold), 1)
            tot_infs += F.relu(num_non_zeros - self.card_upper) + F.relu(self.card_lower - num_non_zeros)
        else:
            num_non_zeros = torch.sum(x.data>self.x_threshold, 1).float()
            tot_infs += Variable(np.maximum(num_non_zeros - self.card_upper, 0) + np.maximum(self.card_lower - num_non_zeros, 0))
    
        
        # (10)
        l = 0.5*torch.sum(torch.abs(x - x_bw))
        tot_infs += F.relu(0.6 - l)
        #tot_infs += F.relu(l - 1.)
    
        
        # (11) a<=dQd <= b
        dQ = (x-x_bw) @ self.Omega
        size_x = x.size()
        dQd = torch.bmm(dQ.view(size_x[0],1,size_x[1]), (x-x_bw).view(size_x[0],size_x[1],1)).squeeze(1).squeeze(1)
        tot_infs += F.relu(dQd - 0.01)
        tot_infs += F.relu(0.0025 - dQd)
        
        
        # min dQd + ad (<= Zstar)
        l = (x-x_bw)@alp

        MONTH =0
        turnover=x_bw
        
        if MONTH > 0 :
            tot_infs += (F.relu(100.*dQd - 100.*l -0.5*turnover - self.Zstar))
            self.objs = 100.*dQd - 100.*l -0.5*turnover
        else:
            tot_infs += (F.relu(100.*dQd - 100.*l  - self.Zstar ))
            self.objs = 100.*dQd - 100.*l
        
        
        fea_probs = F.relu(1. - F.tanh(tot_infs/self.inf_scale))  

        
        return fea_probs