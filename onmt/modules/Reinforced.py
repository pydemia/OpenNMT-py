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
def nparams(_):
  return sum([p.nelement() for p in _.parameters()])

class _Module(nn.Module):
  def __init__(self, opt):
      super(_Module, self).__init__()
      self.opt = opt

  def maybe_cuda(self, o):
      """o may be a Variable or a Tensor
      """
      if len(self.opt.gpus) >= 1:
          return o.cuda()
      return o
      

def assert_size(v, size_list):
    """Check that variable(s) have size() == size_list
       v may be a variable, a tensor or a list
    """
    if type(v) not in [tuple, list]:
        v = [v]
        
    for variable in v:
        _friendly_aeq(real=list(variable.size()), expected=size_list)

def _friendly_aeq(real, expected):
    assert real == expected, "got %s expected: %s" % (str(real), str(expected))

class IntraAttention(_Module):
    def __init__(self, opt, dim, temporal=False):
        super(IntraAttention, self).__init__(opt)
        self.dim = dim
        self.temporal = temporal
        self.W_attn = nn.Parameter(torch.randn(dim, dim))
        self.past_e = None
        self.past_ee = None

    def reset(self):
        self.past_e, self.past_ee = None, None

    def forward(self, h_t, h, t1=False):
        """
        inputs:
            h_t (FloatTensor): batch x dim
            h   (FloatTensor): n x batch x dim
                for encoder_attn n is as in eq. (4), (5) i.e. src_len
                for decoder_attn n is t-1

        returns:
            alpha: [b x src]
            c_t: [b x dim]
        """
        dim = self.dim
        bs = h_t.size(0)
        n = h.size(0)
        assert_size(h_t, [bs, dim])
        assert_size(h, [n, bs, dim])
      

        # h: [b x dim x n]
        h = h.view([bs, dim, n])

        assert_size(self.W_attn, [dim, dim])
        # [b x dim] @ [dim x dim] = [b x dim]
        # [b x dim].unsqueeze(2).transpose(1,2) = [b x 1 x dim]
        # [b x 1 x dim] bmm [b x dim x n] = [b x 1 x n]
        # E = [b x 1 x n].unsqueeze, [b x n]
        # e_ti(b) = E[b][i]

        E_t = (h_t @ self.W_attn).unsqueeze(2).transpose(1, 2).bmm(h).squeeze(1)

        EE_t = E_t
        #print(n)
        #print(EE_t.size())
        # sum(EE_t).usqueeze = [b x 1]
        # [b x n] / [b x 1] = [b x n] (broadcasting) 
        A_t = EE_t / (EE_t.sum(dim=1).unsqueeze(1))

        # [b x dim x n] bmm [b x n x 1] = [b x dim x 1]
        C_t = h.bmm(A_t.unsqueeze(2)).squeeze(2)

        
        assert_size(C_t, [bs, dim])
        assert_size(A_t, [bs, n])

        return A_t, C_t


        """"
        #print("\n\n\n\n ***************Intra ATTN FW************\n\n\n")
        #print("h_t: %s" % str(h_t.size()))
        if type(h) == list:
          #print(len(h))
          #print(h[0].size())
          exit()
        bs, dim = list(h_t.size())

        W_attn = self.W_attn
        
        # bmm: [b x n x m] * [b x m x p] = [b x n x p]
        # <=> for b in range(dim(0)):
        #        res[b] = x[b] @ y[b]
        
        # h_t as: [b x 1 x dim]
        # h as: [b x dim x src]
        h_t = h_t.unsqueeze(1)
        h = h.transpose(0, 1).transpose(1, 2)
        # W_att as: [b x dim x dim]
        #print(W_attn.size())
        W_attn = W_attn.unsqueeze(0).expand([bs, dim, dim]) 

        # Equations (1) and (2) 
        # [b x 1 x src]
        #print(h_t.size())
        #print(W_attn.size())
        #print(h.size())
        #print((torch.bmm(h_t, W_attn)).size())
        e_t = torch.bmm(torch.bmm(h_t, W_attn), h)
       
        #print("e_t:")
        #print(e_t)
        # Equation (3)
        if self.temporal:
            if t1:
                ee_t = e_t.exp_()
                self.past_e = [ee_t]

            else:
                # e_prev is [b x (t-1) x src]
                # d [b x 1 x src]
                #print(type(self.past_e))
                #print(type(e_t))
                #print(type(e_t.exp()))
                sum_past_e = torch.stack(self.past_e).sum(dim=0)
                ee_t = torch.div(e_t.exp(), sum_past_e)
                self.past_e += [ee_t]

        else:
            ee_t = e_t
        
        # Equation (4)
        # alpha_t = [b x 1 x src]
        alpha_t = ee_t / ee_t.sum()
        
        
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
        _alpha_t = alpha_t.transpose(1, 2)
        s = _alpha_t.size()

        # [b, src, src]
        _alpha_t = _alpha_t.expand(s[0], s[1], s[1])
        _h = h.transpose(1, 2)
        # Equation (5) 
        # [b, src, src] bmm [b, src, dim] = [b, src, dim] {a}
        # sum( (a) , dim=1) = [b x 1 x dim] {b}
        # squeeze( {b} ) = [b x dim]
        #print(_alpha_t.size())
        #print(_h.size())
        c_t = torch.sum(_alpha_t.bmm(_h), dim=1).squeeze()

        # [b x src]
        alpha_t = alpha_t.squeeze()
        return alpha_t, c_t
        """

class PointerGenerator(_Module):
    def __init__(self, opt, W_emb):
        super(PointerGenerator, self).__init__(opt)
        self.dim = dim = opt.rnn_size

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

        self.W_emb = W_emb
        n_emb, emb_dim = list(W_emb.size())
        
        self.W_proj = nn.Parameter(torch.Tensor(emb_dim, 3*dim))
        print(type(self.W_emb.data))
                #self.proj_u = nn.Linear(3*dim, 1, bias=True)
        self.W_u = nn.Parameter(torch.Tensor(1, 3*dim))
        self.b_u = nn.Parameter(torch.Tensor(1))
        self.proj_u = lambda V: self.W_u @ V + self.b_u

        self.b_out = nn.Parameter(torch.Tensor(n_emb, 1))
        

    def forward(self, x, ae_t, ce_t, hd_t, cd_t):
        """
        Implementing sect. 2.3: "Token generation and pointer"
        (and sect. 2.4 about weights sharing)
        Args:
            x: [src x bs x 1]
            ae_t: [b x src]
            ce_t: [b x dim]
            hd_t: [b x dim]
            cd_t: [b x dim]

        Parameters:
          W_proj, b_out
          W_u, b_u

        Somehow required:
          W_emb
          x
          alpha_e
          c_e
          h_d
          c_d
        
        Must return:
            p_gen: [bs x vocab_size]
            p_copy: [bs x src_len]
        """
        dim = self.dim
        src_len, bs, _ = list(x.size())
        assert_size(ae_t, [bs, src_len])
        assert_size(ce_t, [bs,dim])
        assert_size(hd_t, [bs,dim])
        assert_size(cd_t, [bs,dim])
        n_emb = self.b_out.size(0)
        
        # [3*dim x bs]
        c_cat = torch.cat((hd_t, ce_t, cd_t), dim=1).t()
        
        # (11), [bs] 
        p_switch = self.sigmoid(self.proj_u(c_cat)).squeeze(0) # [bs]
        assert_size(p_switch, [bs])
       
        self.W_out = self.tanh(self.W_emb @ self.W_proj)
        self.proj_out = lambda V: self.W_out @ V + self.b_out


        p_gen = ((1-p_switch) * self.softmax(self.proj_out(c_cat))).t()
        p_copy = (p_switch * ae_t.t()).t() # t() needed for broadcasting
        
        assert_size(p_gen, [bs, n_emb])
        assert_size(p_copy, [bs, src_len])
        return p_gen, p_copy
        """
        # W_emb: [n_emb x emb_dim]
        # W_proj:[emb_dim x 3*dim]
        # W_out: [n_emb x 3*dim]
        # then expand it to:
        # W_out: [bs, n_emb, 3*dim]
        #print(self.W_emb.size())
        #print(self.W_proj.size())
        W_out = self.tanh(self.W_emb.matmul(self.W_proj)).unsqueeze(0).expand([bs, n_emb, 3*dim])
        b_out = self.b_out

        # concatenate contexts & current state and make it
        # c_cat: [bs x 3*dim x 1]
        #print(type(hd_t))
        #print(type(ce_t))
        #print(type(cd_t))
        c_cat = torch.cat((hd_t, ce_t, cd_t), 1)
        #print(c_cat.size())
        c_cat = c_cat.unsqueeze(2)
        #print(c_cat.size())

        # in (9) [hd_t | ce_t | cd_t] is [3*dim x 1]
        # here: [bs x 3*dim x 1]
        # p_gen(y_t|u_t=0) = softmax(W_out x [hd_t | ce_t | cd_t] +b_out )
        #print("W_out: %s" % str(W_out.size()))
        #print(c_cat.size())
        p0 = torch.bmm(W_out, c_cat)
        #print(p0.size())
        #print("bout: %s" % str(b_out.size()))
        p0 = p0 + b_out
        p0 = self.softmax(p0.squeeze()) # [bs x n_emb]

        
        # (10)
        # p_copy_ti(y_t=x_i) = alpha_e_ti
        _x = x.squeeze()
        p1 = self.maybe_cuda(torch.autograd.Variable(torch.zeros([bs, n_emb, 1])))
        for _b in range(bs):
            for _s in range(src_len):
                _w = _x[_s, _b]
                assert list(_w.size()) == [1]
                _w = _w.data[0]
                #print(_w)
                #print(ae_t.size())
                p1[_b, _w] = ae_t[_b, _s]

        # (11)
        # p_copy = p(u_t=1) = sigmoid(W_u [hd_t | ce_t | cd_t] + b_u)
        #print(self.W_u.size())
        W_u = self.tanh(self.W_emb.matmul(self.W_proj_u))
        W_u = W_u.unsqueeze(0).expand([bs, n_emb, 3*dim])
        b_u = self.b_u
        p_u = self.sigmoid(torch.bmm(W_u, c_cat)+b_u)
        
        #print(type(p_u))
        #print(type(p0))
        #print(type(p1))
        p_y = (p_u * p1 + (1-p_u) * p0).squeeze()
        return p_y
        """
class ReinforcedDecoder(_Module):
    def __init__(self, opt, embeddings):
        super(ReinforcedDecoder, self).__init__(opt)
        self.input_size = opt.word_vec_size
        
        self.dim = opt.rnn_size
        self.rnn = onmt.modules.StackedLSTM(1, self.input_size, opt.rnn_size, opt.dropout)

        self.enc_attn = IntraAttention(opt, self.dim, temporal=True)
        self.dec_attn = IntraAttention(opt, self.dim)

        W_emb = embeddings.word_lut.weight
        self.pointer_generator = PointerGenerator(opt, W_emb)
        self.embeddings = embeddings
        
        print("ReinforcedDecoder params: %d" % nparams(self))
        print("enc_attn: %s" % nparams(self.enc_attn))
        print({n: p.nelement() for n,p in self.enc_attn.named_parameters()})
        print("dec_attn: %d" % nparams(self.dec_attn))
        print({n: p.nelement() for n,p in self.dec_attn.named_parameters()})
        print("poingen: %d" % nparams(self.pointer_generator))
        print({n: p.nelement() for n,p in self.pointer_generator.named_parameters()})

        print("rnn: %d" % nparams(self.rnn))
        print({n: p.nelement() for n,p in self.rnn.named_parameters()})

    def forward(self, batch, h_e, init_state):
        """
        Args:
            batch: onmt.Dataset.Batch
            #TODO remove it
            ***inputs: tgt_len x batch -- tgt tokens
            ***src: src_len x batch x 1
            h_e: src_len x batch x dim

        Returns:
            p_gen: tgt_len x batch x voc_size
            p_copy: tgt_len x batch x src_len
            dec_state
        
        """
        def bottle(v):
            return v.view(-1, v.size(2))

        src = batch.src                 # [src_len x bs x 1]
        tgt = batch.tgt                # [tgt_len x bs]
        lengths = batch.lengths         # [1 x bs]
        align = batch.alignment
        print(align.size())
        print(tgt.size()) 
        inputs = tgt[:-1]
    
        crit = onmt.modules.CopyCriterion

        dim = self.dim
        src_len, bs, _ = list(src.size())
        tgt_len = tgt.size(0)
        #print(inputs.size())
        assert_size(tgt, [tgt_len, bs])
        assert_size(h_e, [src_len, bs, dim])
        
        emb = self.embeddings(inputs.unsqueeze(2))
        print(emb.size())
        #print("DECODER EMBEDDED")
        hd_history = None #[]
        outputs, copies = [], []
        
        # chose one
        #loss = []
        loss = None
        #losses = []
        def _copy(var):
            return torch.autograd.Variable(var.data, requires_grad=True)

        hidden = init_state#[h_e[-1]]*2
        for t, emb_t in enumerate(emb.split(1)):
            #print("DECODER It: %d" % t)
            emb_t = emb_t.squeeze(0)
              
            #print("it %d" %len(hidden))
            #print(hidden[0].size())
            #print("emb_t: %s" % str(emb_t.size()))
            out, hidden = self.rnn(emb_t, hidden)
            hd_t = hidden[0].squeeze(0)
            alpha_e, c_e = self.enc_attn(_copy(hd_t), _copy(h_e), t1=(t==0))
           
            
            # list of hd_j for j in [0, t] (i.e. eq (8))
            if hd_history is None:
                hd_history = hd_t.unsqueeze(0)
            else:
                hd_history = torch.cat([hd_history, hd_t.unsqueeze(0)], dim=0)
            
            if t==0:
                cd_t = self.maybe_cuda(torch.autograd.Variable(torch.zeros([bs, dim])))
            else:
                # stacking history to a [t x bs x dim] tensor 
                alpha_d, cd_t = self.dec_attn(_copy(hd_t), _copy(hd_history))
                #alpha_d, cd_t = self.dec_attn(hd_t, hd_history)
            
            p_gen, p_copy = self.pointer_generator(_copy(src), alpha_e, c_e, _copy(hd_t), cd_t)
            l = crit(p_gen, p_copy, tgt[t, :].unsqueeze(1), align[t, :, :].squeeze(0))
            l = l.div(batch.batchSize)
            
            #l.backward(retain_graph=True)
            print("going to bw (%d)" % t)
            l.backward()
            print("backwarded (%d)" % t) 
            
            #loss += [l]
             
            #loss = l if loss is None else loss + l
            
            #print("pause %d" % t)
            #input()


            #outputs += [p_gen]
            #copies += [p_copy]

        # (tgt_len, bs, n_emb)
        #print("DECODER OUT")
        #torch.stack(outputs), torch.stack(copies)
        return None, None, hidden, loss#torch.stack(losses, dim=0)

class ReinforcedModel(nn.Module):
    def __init__(self, encoder, decoder, multigpu=False):
        super(ReinforcedModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        print("#PARAMS TOTAL: %d"
                % nparams(self))
        print({n: p.nelement() for n,p in self.named_parameters()})
    def forward(self, batch, dec_state=None):
        """
        Args:
            batch: a batch object (onmt.Dataset.Batch)
            dec_state: A decoder state object


        Regular Model (for ref):
            p_gen: (bs x vocab_size)
            p_copy: (bs x src_len)
            
            ##outputs (FloatTensor): (len x batch x rnn_size) -- Decoder outputs.
            ##attns (FloatTensor): Dictionary of (src_len x batch)
            ##dec_hidden (FloatTensor): tuple (1 x batch x rnn_size)
                                      Init hidden state
        """
        src = batch.src                 # [src_len x bs x 1]
        tgt = batch.tgt                # [tgt_len x bs]
        lengths = batch.lengths         # [1 x bs]
        align = batch.alignment
        

        #print("MODEL FW 0")
        #print(type(src))
        #print("src size: %s" % str(src.size()))
        #print("tgt size: %s" % str(tgt.size()))
        #print("length size: %s" % str(lengths.size()))
        #print("lengths: %s" % str(lengths))
        bs = src.size(1)

        src = src
        tgt = tgt[:-1]  # exclude last target from inputs
        #encoder
        #in: input (LongTensor): len x batch x nfeat, lengths (LongTensor): batch, hidden: Initial hidden state.
        #out: hidden_t pair of layers x batch x rnn_size, outputs:  len x batch x rnn_size
        enc_hidden, enc_out = self.encoder(src, lengths)
        
        # TODO 1 below is actually opt.layers
        # must check not to destroy anything
        # not a priority
        
        enc_hidden = [ state.view([1, bs, 2*state.size(2)]) for state in enc_hidden]
        # he_t = enc_hidden[0].view([bs, dim])
        
        #print(enc_out.size())
        #print(len(enc_hidden))
        #print(enc_hidden[0].size())
        #print(enc_hidden[1].size())

        #print("goto decode")
        #print("MODEL FW ENCODED")
        """enc_state = self.init_decoder_state(context, enc_hidden)"""

        #decoder:
        #in     input: tgt_len x batch, 
        #       src: src_len x batch, 
        #       h_e: src_len x batch x dim
        #       (h,c) encoder states
        #
        #out    outputs, dec_state
        p_gen, p_copy, dec_state, loss = self.decoder(batch, enc_out, enc_hidden)
        #print(p_gen.size())
        #print(p_copy.size())
        #exit()
        #print("MODEL FW DECODED; RETURN")
        #if self.multigpu:
        #    # Not yet supported on multi-gpu
        #    dec_state = None
        #    attns = None
        _ = None
        dec_state = onmt.Models.RNNDecoderState(dec_state)
        return p_gen, p_copy, dec_state, loss

