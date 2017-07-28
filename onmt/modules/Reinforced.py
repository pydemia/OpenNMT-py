"""
Implementation of "A Deep Reinforced Model for Abstractive Summarization"
Romain Paulus, Caiming Xiong, Richard Rocheri
https://arxiv.org/abs/1705.04304
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import onmt
import onmt.modules


class IntraAttention(nn.Module):
    def __init__(self, dim, temporal):
        super(IntraAttention, set).__init__()
        self.temporal = temporal
        self.W_attn = nn.Parameter(torch.randn(dim, dim))
        self.past_e = None
        self.past_ee = None

    def reset(self):
        self.past_e, self.past_ee = None, None

    def forward(self, h_t, h, t1=False):
        """"
        h_t (FloatTensor): batch x dim
        h   (FloatTensor): batch x src_len x dim
        """"
        if self.past_e is None:
            self.past_e = torch.zeros(h.size()[0], 1, h.size()[1])
        if self.past_ee is None:
            self.past_ee = torch.zeros(h.size()[0], 1, h.size()[1])

        past_e, past_ee, W_attn = self.past_e, self.past_ee, self.W_attn
        
        # bmm: [b x n x m] * [b x m x p] = [b x n x p]
        # <=> for b in range(dim(0)):
        #        res[b] = x[b] @ y[b]
        
        # h_t as: [b x 1 x dim]
        h_t.unsqueeze_(1)

        # W_att as: [b x dim x dim]
        W_attn = W_attn.unsqueeze(0).expand_as(h_t) 

        # Equations (1) and (2) 
        # [b x 1 x src]
        e_t = torch.bmm(torch.bmm(h_t, W_attn), h.transpose(1, 2))

        # Equation (3)
        if self.temporal:
            if t1:
                ee_t = e_t.exp_()
            else::
                # e_prev is [b x (t-1) x src]
                # d [b x 1 x src]
                ee_t = e_t.exp().div(past_e)
            past_e.add_(e_t.exp())
        else:
            ee_t = e_t
        
        # Equation (4)
        alpha_t = ee_t.div(past_ee)

        past_ee.add_(ee_t)
        
        # Equation (5) is equivalent to:
        # for b<batch:
        #     for i<n:
        #           sum += alpha_t[i] x h[b][i]
        #
        # in term of matrix, we first want to get the
        # sum[b x src_len x dim] such that:
        # sum[b, i, :] = alpha_t[i] x h[b][i]
        #
        # first we make alpha_t a tensor[b x src_len x src_len] such that:
        # alpha_t[b, i, x] = alpha_t[b, i, y] for any (x, y) 
        # i.e. make each alpha_t[b] a [src_len, 1] vector, 
        # and then expand it to [src_len, src_len] (columns are then identicals)
        alpha.transpose_(1, 2)
        s = alpha.size()

        # [b, src, src]
        alpha.expand(s[0], s[1], s[1])

        # Equation (5) 
        # [b, src, src] bmm [b, src, dim] = [b, src, dim]
        c_t = torch.sum(alpha.bmm(h), dim=) 

        return c_t 

class Pointer(nn.Module):
    def __init__(self, opt):
        super(Pointer, self).__init__()
        Wout = nn.Parameter(____)
        Wu = nn.Parameter(_____)
        pass

    def forward(self, _):
        pass

class Encoder(nn.Module):
    def __init__(self, opt):
        pass

    def forward(self):
        
        pass

class Decoder(nn.Module):
    def __init__(self, opt):
        enc_attn = IntraAttention(dim, temporal=True)
        dec_attn = IntraAttention(dim)

        pass

    def forward(self):
        pass 
