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
import onmt.Models

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

    def mkvar(self, tensor, requires_grad=False):
        return self.maybe_cuda(torch.autograd.Variable(tensor, requires_grad=requires_grad))

    def copyvar(self, var, requires_grad=True):
        assert type(var) == torch.autograd.Variable, "Parameter must be a torch.Variable, got %s" % type(var)
        if not self.training:
            return var
        return self.mkvar(var.data, requires_grad=requires_grad)

      
    def forward(self, *args, **kwargs):
        if self.training:
            return self.train_forward(*args, **kwargs)
        return self.infer_forward(*args, **kwargs)
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
    """IntraAttention Module as in sect. (2)
    """
    def __init__(self, opt, dim, temporal=False):
        super(IntraAttention, self).__init__(opt)
        self.dim = dim
        self.temporal = temporal
        self.W_attn = nn.Parameter(torch.randn(dim, dim))

    def forward(self, h_t, h, E_history=None):
        """

        args:
            h_t (FloatTensor): [bs x dim]
            h   (FloatTensor): [n x bs x dim]
                for encoder_attn n is as in eq. (4), (5) i.e. src_len
                for decoder_attn n is t-1
            E_history: None
                or (FloatTensor): [bs x n], sum of previous attn scores, aka. e'ti

        returns:
            alpha: [b x src]
            c_t: [b x dim]
            E_history: None
                or (FloatTensor): [bs x n], sum of attn scores, aka. e'ti.
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

        if self.temporal:
            if E_history is None:
                EE_t = E_t.exp()
                E_history = EE_t
            else:
                EE_t = E_t.exp() / E_history
                # copy because autograd does not like in-place operation
                E_history = self.copyvar(E_history) + E_t.exp()
        else:
            EE_t = E_t.exp()
        # sum(EE_t).usqueeze = [b x 1]
        # [b x n] / [b x 1] = [b x n] (broadcasting) 
        A_t = EE_t / (EE_t.sum(dim=1).unsqueeze(1))

        # [b x dim x n] bmm [b x n x 1] = [b x dim x 1]
        C_t = h.bmm(A_t.unsqueeze(2)).squeeze(2)

        
        assert_size(C_t, [bs, dim])
        assert_size(A_t, [bs, n])

        if self.temporal:
            return A_t, C_t, E_history
        return A_t, C_t


class PointerGenerator(_Module):
    def __init__(self, opt, W_emb, pad_id):
        super(PointerGenerator, self).__init__(opt)
        self.dim = dim = opt.rnn_size

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

        n_emb, emb_dim = list(W_emb.size())
        
        self.W_proj = nn.Parameter(torch.Tensor(emb_dim, 3*dim))
        self.W_u = nn.Parameter(torch.Tensor(1, 3*dim))
        self.b_u = nn.Parameter(torch.Tensor(1))
        self.proj_u = lambda V: self.W_u @ V + self.b_u

        self.b_out = nn.Parameter(torch.Tensor(n_emb, 1))
        self.W_emb = W_emb

        self.pad_id = pad_id

    def forward(self, ae_t, ce_t, hd_t, cd_t, pad_id):
        """
        Implementing sect. 2.3: "Token generation and pointer"
        (and sect. 2.4 about weights sharing)
        Args:
            ae_t: [b*t x src]
            ce_t: [b*t x dim]
            hd_t: [b*t x dim]
            cd_t: [b*t x dim]
        Returns:
            p_gen: [bs*t x vocab_size]
            p_copy: [bs*t x src_len]
            p_switch: [bs*t]
        """
        dim = self.dim
        n, src_len = list(ae_t.size())
        assert_size(ae_t, [n, src_len])
        assert_size(ce_t, [n, dim])
        assert_size(hd_t, [n, dim])
        assert_size(cd_t, [n, dim])
        n_emb = self.b_out.size(0)
        
        # [3*dim x bs]
        c_cat = torch.cat((hd_t, ce_t, cd_t), dim=1).t()
        
        # (11), [bs] 
        p_switch = self.sigmoid(self.proj_u(c_cat)).squeeze(0) # [bs]
        assert_size(p_switch, [n])
       
        self.W_out = self.tanh(self.W_emb @ self.W_proj)
        self.proj_out = lambda V: self.W_out @ V + self.b_out
        
        logits = self.proj_out(c_cat).t()
        logits[:, self.pad_id] = -float('inf')
        p_gen =  self.softmax(logits+1e-12)
        p_copy = ae_t 
        
        assert_size(p_gen, [n, n_emb])
        assert_size(p_copy, [n, src_len])
        return p_gen, p_copy, p_switch


class ReinforcedDecoder(_Module):
    def __init__(self, opt, embeddings, pad_id):
        super(ReinforcedDecoder, self).__init__(opt)
        W_emb = embeddings.word_lut.weight
        self.input_size = W_emb.size(1)
        
        #TODO 
        self.tgt_vocab_size = W_emb.size(0)

        self.dim = opt.rnn_size
        self.rnn = onmt.modules.StackedLSTM(1, self.input_size, opt.rnn_size, opt.dropout)

        self.enc_attn = IntraAttention(opt, self.dim, temporal=True)
        self.dec_attn = IntraAttention(opt, self.dim)

        self.pointer_generator = PointerGenerator(opt, W_emb,pad_id)
        self.embeddings = embeddings
       

        self.decoder_layer = "reinforced"
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
        self.pad_id = pad_id
        

    def forward(self, inputs, src, h_e, init_state, tgt=None):
        """
        Args:
            inputs: tgt_len x batch -- tgt tokens
            src: src_len x batch x 1
            h_e: src_len x batch x dim
            init_state: a decoder state

        Returns:
            dec_state: onmt.Models.RNNDecoderState
            stats: onmt.Loss.Statistics  
        """

        def bottle(v):
            return v.view(-1, v.size(2))


        stats = onmt.Loss.Statistics()
        #src = batch.src                 # [src_len x bs x 1]
        #tgt = batch.tgt                # [tgt_len x bs]
        #lengths = batch.lengths         # [1 x bs]
        #align = batch.alignment
        #inputs = tgt[:-1]
        
        dim = self.dim
        src_len, bs, _ = list(src.size())
        input_size, _bs = list(inputs.size())
        assert bs == _bs
        
        assert_size(inputs, [input_size, bs])
        assert_size(h_e, [src_len, bs, dim])
        
        emb = self.embeddings(inputs.unsqueeze(2))
        
        if init_state.coverage is not None:
            hd_history = init_state.coverage
        else:
            hd_history = None

        scores = None if self.training else []
        attns = []

        hidden = init_state.hidden
        E_hist = None
        for t, emb_t in enumerate(emb.split(1, dim=0)):
            emb_t = emb_t.squeeze(0)
             
            out, hidden = self.rnn(emb_t, hidden)
            hd_t = hidden[0].squeeze(0)
            alpha_e, c_e, _E_hist = self.enc_attn(self.copyvar(hd_t), self.copyvar(h_e), E_hist)
            E_hist = self.copyvar(_E_hist)
            
            if hd_history is None:
                # [1 x bs x dim]
                hd_history = hd_t.unsqueeze(0)
            else:
                # [t x bs x dim]
                hd_history = torch.cat([hd_history, hd_t.unsqueeze(0)], dim=0)
            
            if t==0:
                cd_t = self.maybe_cuda(torch.autograd.Variable(torch.zeros([bs, dim])))
                alpha_d = self.copyvar(cd_t)
            else:
                alpha_d, cd_t = self.dec_attn(self.copyvar(hd_t), self.copyvar(hd_history))
            
            # [bs*t x voc], [bs*t x src_len], x [bs*t]
            p_gen, p_copy, p_switch = self.pointer_generator(alpha_e, 
                                            c_e, 
                                            self.copyvar(hd_t), 
                                            cd_t, 
                                            self.pad_id)
            
            eps=1e-12
            n = bs
            voc = p_gen.size(1)
            gen_scores = p_gen * p_switch.view(-1, 1)
            inputs_t = inputs[t, :]
            # [bs*t x d_voc]
            copy_scores = self.mkvar(torch.zeros(n, max(src.max().data[0], voc)))
            copy_scores.scatter_(1, src.squeeze(2).t(), p_copy)
            copy_scores[:, :self.tgt_vocab_size] = copy_scores[:, :self.tgt_vocab_size].clone() * p_switch.view(-1,1).expand([n, self.tgt_vocab_size])

            
            scores_t = copy_scores
            scores_t[:, :self.tgt_vocab_size] = scores_t[:, :self.tgt_vocab_size].clone() + gen_scores

            
            pred_t = scores_t.max(1)[1]
            l = 0
            correct_words = 0

            if self.training or tgt is not None:
                targ_t = tgt[t, :].view(-1, 1)

                loss_t = torch.log(scores_t.gather(1, targ_t) + eps)
                loss_t = loss_t.mul(targ_t.ne(self.pad_id).float())
                # incorrect size for targ_t or loss_t may less to [bs x bs] loss_t
                # without errors
                assert_size(loss_t, [bs, 1])
                
                loss_t = -loss_t.sum()
                l = loss_t.data[0]
                correct_words = pred_t.view(-1).eq(targ_t.view(-1)).float().sum().data[0]
            non_padding = targ_t.ne(self.pad_id).float()
            n_words = non_padding.sum()
            
            stats_t = onmt.Loss.Statistics(loss=l,
                                      n_words=n_words.data[0],
                                      n_correct=correct_words)
            stats.update(stats_t)
            if self.training:
                loss_t.div(bs).backward()
            else:
                scores += [scores_t]
                attns += [alpha_d]
        
        dec_state = onmt.Models.RNNDecoderState(hidden, coverage=hd_history)
        return stats, dec_state, scores, attns


class ReinforcedModel(onmt.Models.NMTModel):
    def __init__(self, encoder, decoder, multigpu=False):
        super(ReinforcedModel, self).__init__(encoder, decoder)
        print("#PARAMS TOTAL: %d"
                % nparams(self))
        print({n: p.nelement() for n,p in self.named_parameters()})
   
    def init_decoder_state(self, enc_hidden, context=None):
        state = super(ReinforcedModel, self).init_decoder_state(enc_hidden=enc_hidden,
                                                    context=None,
                                                    input_feed=False)
        #temp_state = TemporalDecoderState(state.hidden, None)
        return state #temp_state

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
        
        if dec_state is None:
            dec_state = self.init_decoder_state(enc_hidden=enc_hidden)

        stats, dec_state, _, _ = self.decoder(batch.tgt[:-1], batch.src, enc_out, dec_state, tgt=batch.tgt[1:])
        
        return stats, dec_state
