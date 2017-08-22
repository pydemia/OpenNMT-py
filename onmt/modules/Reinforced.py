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

    def mkvar(self, tensor):
        assert type(tensor) == torch.Tensor, "Parameter must be a tensor, got %s" % type(tensor)
        return self.maybe_cuda(torch.autograd.Variable(tensor))

      

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

        #TODO local attention
        EE_t = E_t.exp()
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
        

    def forward(self, ae_t, ce_t, hd_t, cd_t):
        """
        Implementing sect. 2.3: "Token generation and pointer"
        (and sect. 2.4 about weights sharing)
        Args:
            ae_t: [b x src]
            ce_t: [b x dim]
            hd_t: [b x dim]
            cd_t: [b x dim]

        Returns:
            p_gen: [bs x vocab_size]
            p_copy: [bs x src_len]
        """
        dim = self.dim
        bs,src_len = list(ae_t.size())
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

        p_gen = (1-p_switch.view(-1, 1)) * self.softmax(self.proj_out(c_cat).t())
        p_copy = (p_switch * ae_t.t()).t() # t() needed for broadcasting
        
        assert_size(p_gen, [bs, n_emb])
        assert_size(p_copy, [bs, src_len])
        return p_gen, p_copy


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

        self.crit = MLCriterion(opt)

    def forward(self, batch, h_e, init_state):
        """
        Args:
            batch: onmt.Dataset.Batch
            #TODO remove it
            ***inputs: tgt_len x batch -- tgt tokens
            ***src: src_len x batch x 1
            h_e: src_len x batch x dim

        Returns:
            dec_state: onmt.Models.RNNDecoderState
            stats: onmt.Loss.Statistics  
        """
        def bottle(v):
            return v.view(-1, v.size(2))

        def _copy(var):
            return torch.autograd.Variable(var.data, requires_grad=True)

        stats = onmt.Loss.Statistics()
        src = batch.src                 # [src_len x bs x 1]
        tgt = batch.tgt                # [tgt_len x bs]
        lengths = batch.lengths         # [1 x bs]
        align = batch.alignment
        inputs = tgt[:-1]
        
        dim = self.dim
        src_len, bs, _ = list(src.size())
        tgt_len = tgt.size(0)
        
        assert_size(tgt, [tgt_len, bs])
        assert_size(h_e, [src_len, bs, dim])
        
        emb = self.embeddings(inputs.unsqueeze(2))
        
        hd_history = None #[]
        
        
        hidden = init_state
        for t, emb_t in enumerate(emb.split(1)):
            emb_t = emb_t.squeeze(0)
              
            out, hidden = self.rnn(emb_t, hidden)
            hd_t = hidden[0].squeeze(0)
            alpha_e, c_e = self.enc_attn(_copy(hd_t), _copy(h_e), t1=(t==0))
           
            
            if hd_history is None:
                # [1 x bs x dim]
                hd_history = hd_t.unsqueeze(0)
            else:
                # [t x bs x dim]
                hd_history = torch.cat([hd_history, hd_t.unsqueeze(0)], dim=0)
            
            if t==0:
                cd_t = self.maybe_cuda(torch.autograd.Variable(torch.zeros([bs, dim])))
            else:
                alpha_d, cd_t = self.dec_attn(_copy(hd_t), _copy(hd_history))
            
            p_gen, p_copy = self.pointer_generator(alpha_e, c_e, _copy(hd_t), cd_t)
            loss_t, pred_t, stats_t = self.crit(p_gen, 
                                        p_copy, 
                                        tgt[t, :].unsqueeze(1), 
                                        align[t, :, :].squeeze(0), 
                                        src.squeeze(2).t())
            stats.update(stats_t)
            #loss_t /= batch.batchSize
            loss_t.backward()
        
        dec_state = onmt.Models.RNNDecoderState(hidden)
        return stats, dec_state

class MLCriterion(_Module):

    def __init__(self, opt):
        super(MLCriterion, self).__init__(opt)
        
    def forward(self, probs, attn, targ, align, src, eps=1e-12):
        """
        Args:
            probs: [tgt_len*bs x vocab
            attn: [tgt_len*bs x src_len]
            targ: [tgt_len*bs x 1]
            align: [tgt_len*bs x src_len]
            src: [bs x src_len]
        
        Returns:
            loss: Variable, [1]
            predictions:
            stats: onmt.Loss.Statistics
        """
        bs, vocab_size = list(probs.size()) 
        
        #TODO NOTE this is doing like abisee i.e. counts
        # [bs x src]
        copies = attn.mul(Variable(align))

        # create a [bs x vocab_size] 
        # matrix such as copies_voc[i] = 0 or copies[j] if src[j] == i
        copies_voc = self.mkvar(torch.zeros([bs, vocab_size]))

        # src that are in vocabulary, 0 else
        voc_src = src*src.lt(vocab_size).long()
        
        # copy probability of in vocabulary src tokens
        copies_voc.scatter_(1,voc_src, copies)
        
        # total probability of prediction in vocabulary tokens
        voc_probs = probs + copies_voc

        # in voc tokens sorted by max
        # it outputs a tuple (max, argmax)
        max_voc_probs = voc_probs.data.max(1)
        max_copies_probs = copies.data.max(1)

        # out of voc tokens that would be copied
        oov_tokens = src.gather(1, max_copies_probs[1].unsqueeze(1))

        # are oov probability greater than generator?
        gen_oov = max_voc_probs[0].lt(max_copies_probs[0]).long()

        predictions = (1-gen_oov) * max_voc_probs[1] \
                    +  gen_oov    * oov_tokens.squeeze(1).data

        non_padding = targ.ne(onmt.Constants.PAD).data
        
        num_correct = predictions.unsqueeze(1).eq(targ.data) \
                              .masked_select(non_padding) \
                              .sum()
        num_words = non_padding.sum()
        
        # probability that the y* = p_gen(y*) + p_copy(y*)
        # below vectors are [bs x 1]
        targ_gen_probs = probs.gather(1, targ.view(-1, 1))
        targ_copy_probs = copies.sum(-1).add(eps).view(-1, 1)
        tot_prob = targ_gen_probs + targ_copy_probs + eps
        
        loss = torch.log(tot_prob)

        # masking
        loss = loss.mul(targ.ne(onmt.Constants.PAD).float())
        assert_size(loss, [bs, 1]) 
        
        loss = -loss.sum() 
        stats = onmt.Loss.Statistics(loss=loss.data[0],
                                     n_words=num_words,
                                     n_correct=num_correct)
        return loss, predictions, stats

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
        bs = src.size(1)
        lengths = batch.lengths
        #encoder
        #in: input (LongTensor): len x batch x nfeat, lengths (LongTensor): batch, hidden: Initial hidden state.
        #out: hidden_t pair of layers x batch x rnn_size, outputs:  len x batch x rnn_size
        enc_hidden, enc_out = self.encoder(src, lengths)
        
        # TODO 1 below is actually opt.layers
        # must check not to destroy anything
        # not a priority
        enc_hidden = [ state.view([1, bs, 2*state.size(2)]) for state in enc_hidden]

        stats, dec_state = self.decoder(batch, enc_out, enc_hidden)
        
        return stats, dec_state

