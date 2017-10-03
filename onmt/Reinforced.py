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

def nonan(variable, name):
    st = variable.data
    if not (st != st).sum()==0:
        print("NaN values in %s=%s" % (name, str(st)))
        exit()

def nparams(_):
  return sum([p.nelement() for p in _.parameters()])

class _Module(nn.Module):
    def __init__(self, opt):
        super(_Module, self).__init__()
        self.opt = opt

    def maybe_cuda(self, o):
        """o may be a Variable or a Tensor
        """
        if len(self.opt.gpuid) >= 1:
            return o.cuda()
        return o

    def mkvar(self, tensor, requires_grad=False):
        return self.maybe_cuda(torch.autograd.Variable(tensor, requires_grad=requires_grad))

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
        self.linear = nn.Linear(dim, dim, bias=False)

        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()

    def forward(self, h_t, h, E_history=None, mask=None,debug=False):
        bs, dim = h_t.size()
        n, _bs, _dim = h.size()
        assert (_bs, _dim)==(bs, dim)


        _h_t = self.linear(h_t).unsqueeze(1)
        _h = h.view(n, bs, dim)

        #[bs, 1, dim] x [n, bs, dim]
        #=> [bs, 1, dim] bmm [bs, dim, n] = [bs, 1, n]
        scores = _h_t.bmm(_h.transpose(0, 1).transpose(1, 2))

        # [bs, n]
        alpha = self.softmax(scores.squeeze(1))
        
        # [bs, 1, n] bmm [n, bs, dim] = [bs, 1, n]
        # [bs, dim, n] bmm [bs, n, 1] = [bs, dim, 1]
        # [bs, 1, n] bmm [bs, n, dim] = [bs, 1, dim]
        C_t = alpha.unsqueeze(1).bmm(_h.transpose(0,1)).squeeze(1)

        if self.temporal:
            return C_t, alpha, None
        return C_t, alpha


class PointerGenerator(_Module):
    def __init__(self, opt, embeddings):
        super(PointerGenerator, self).__init__(opt)
        self.pad_id = embeddings.padding_idx
        W_emb = embeddings.word_lut.weight
        self.W_emb = W_emb
        
        self.dim = dim = opt.rnn_size

        n_emb, emb_dim = list(W_emb.size())

        # (2.4) Sharing decoder weights
        self.W_proj = nn.Parameter(torch.Tensor(emb_dim, 3*dim))
        self.b_out = nn.Parameter(torch.Tensor(n_emb, 1))

        self.proj_u = nn.Linear(3*dim, 1)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    @property
    def W_out(self):
        """ Sect. (2.4) Sharing decoder weights
            Returns:
                W_out (FloaTensor): [n_emb, 3*dim]
        """
        # eq. (13)
        return self.tanh(self.W_emb @ self.W_proj)

    def proj_out(self, V):
        """Calculate the output projection of `v` as in eq. (9)
            Args:
                V (FloatTensor): [bs, 3*dim]
            Returns:
                logits (FloatTensor): logits = W_out * V + b_out, [3*dim]
        """
        return (self.W_out @ V.t() + self.b_out).t()

    def forward(self, ae_t, ce_t, hd_t, cd_t):
        """
        Implementing sect. 2.3: "Token generation and pointer"

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
        n_emb = self.W_emb.size(0)
       
        nonan(hd_t, "hd_t")
        nonan(ce_t, "ce_t")
        nonan(cd_t, "cd_t")


        # [bs, 3*dim]
        c_cat = torch.cat((hd_t, ce_t, cd_t), dim=1)
        nonan(c_cat,"c_cat")
        
        # (9)
        logits = self.proj_out(c_cat)
        logits[:, self.pad_id] = -float('inf')
        p_gen =  self.softmax(logits)

        # (10)
        p_copy = ae_t 

        # (11), [bs]
        p_switch = self.sigmoid(self.proj_u(c_cat)).squeeze(1) # [bs]

        for name, var in {"gen":p_gen,"copy":p_copy, "switch": p_switch}.items():
            nonan(var, name)

        assert_size(p_switch, [n])
        assert_size(p_gen, [n, n_emb])
        assert_size(p_copy, [n, src_len])

        return p_gen, p_copy, p_switch

class MLCriterion(_Module):
    """Maximum Likelihood as described in sect. 3.1, eq. (14)
       based on pointer-generator results of sect. 2.3, eq. (12)

    """
    def __init__(self, vocab_size, opt, pad_id):
        super(MLCriterion, self).__init__(opt)
        self.pad_id = pad_id
        self.vocab_size = vocab_size
        
    def scores(self, p_gen, p_copy, p_switch, src):
        """
        Args:
            (see MLCriterion.foward)

        Returns:
            scores: [bs*t x c_vocab]
        """

        n, vocab_size = list(p_gen.size())
        _, src_l = list(p_copy.size())

        assert_size(p_copy, [n, src_l])
        assert_size(p_switch, [n])
        assert_size(src, [n, src_l])

        # Calculating scores, [bs*t x c_vocab], as in eq. (12)
        gen_scores = p_gen * (1-p_switch.view(-1, 1))

        c_vocab_size =  max(src.max().data[0], self.vocab_size)
        copy_scores = self.mkvar(torch.zeros(n, c_vocab_size))
        copy_scores.scatter_add_(1, src, p_copy)
        copy_scores[:, :self.vocab_size] = copy_scores[:, :self.vocab_size].clone() \
                            * p_switch.view(-1,1).expand([n, self.vocab_size])

        scores = copy_scores
        scores[:, :self.vocab_size] = scores[:, :self.vocab_size] + gen_scores
        return scores

    def forward(self, p_gen, p_copy, p_switch, tgt, src):
        """
        Args:
            p_gen (FloatTensor): [bs*t x vocab_size]
            p_copy (FloatTensor): [bs*t x src_length]
            p_switch (FloatTensor): [bs*t]
            tgt (FloatTensor): [bs*t]
                or None
            src (LongTensor): [bs*t x src_length]

        Returns:
            loss (FloatTensor): [1]
            pred (FloatTensor): [bs*t, 1]
            stats (onmt.Loss.Statistics)
                or None if tgt == None
        """
        scores = self.scores(p_gen, p_copy, p_switch, src)
        n = scores.size(0)

        # Stats: prediction, #words, #correct_words
        pred_scores, pred = list(scores.max(1))
        
        eps = 1e-12
        # Calulating loss, as in eq. (14)
        non_padding = tgt.ne(self.pad_id)
        target_scores = scores.gather(1, tgt.view(-1, 1))
        loss = torch.log(target_scores + eps) * non_padding.float()
        loss = -loss.sum()/n
        
        n_words = non_padding.sum()
        n_correct = pred.eq(tgt).masked_select(non_padding).sum().data[0]
        
        stats = onmt.Loss.Statistics(loss.data[0], n_words.data[0], n_correct)
        return loss, pred, stats


class ReinforcedDecoder(_Module):
    def __init__(self, opt, embeddings):
        super(ReinforcedDecoder, self).__init__(opt)
        self.embeddings = embeddings
        W_emb = embeddings.word_lut.weight
        self.tgt_vocab_size, self.input_size = W_emb.size()
        self.dim = opt.rnn_size

        self.rnn = onmt.modules.StackedLSTM(opt.layers, self.input_size, opt.rnn_size, opt.dropout)

        self.enc_attn = IntraAttention(opt, self.dim, temporal=True)
        self.dec_attn = IntraAttention(opt, self.dim)

        self.pointer_generator = PointerGenerator(opt, embeddings)
       
        self.pad_id = embeddings.padding_idx 
        self.ml_crit = MLCriterion(self.tgt_vocab_size, opt, self.pad_id)

        # For compatibility reasons, TODO refactor
        self.hidden_size = self.dim
        self.decoder_type = "reinforced"

    def forward(self, inputs, src, h_e, init_state, tgt=None):
        """
        Args:
            inputs (LongTensor): [tgt_len x bs]
            src (LongTensor): [src_len x bs x 1]
            h_e (FloatTensor): [src_len x bs x dim]
            init_state: onmt.Models.DecoderState
            tgt (LongTensor): [tgt_len x bs]

        Returns:
            stats: onmt.Loss.Statistics
            dec_state onmt.Models.DecoderState
            None: TODO refactor
            None: TODO refactor
        """
        dim = self.dim
        src_len, bs, _ = list(src.size())
        input_size, _bs = list(inputs.size())
        assert bs == _bs

        # src as [bs x src_len]
        src = src.transpose(0, 1).squeeze(2)

        stats = onmt.Loss.Statistics()
        hidden = init_state.hidden
        loss, E_hist = None, None
        scores, attns = [], []
        inputs_t = inputs[0, : ]
        for t in range(input_size):
            src_mask = src.eq(self.pad_id)
            emb_t = self.embeddings(inputs_t.view(1, -1, 1)).squeeze(0)

            hd_t, hidden = self.rnn(emb_t, hidden)

            c_e, alpha_e, E_hist = self.enc_attn(hd_t, h_e, E_history=E_hist)

            if t==0:
                # no decoder intra attn at first step
                cd_t = self.mkvar(torch.zeros([bs, dim]))
                hd_history = hd_t.unsqueeze(0)
            else:
                cd_t, alpha_d = self.dec_attn(hd_t, hd_history)
                hd_history = torch.cat([hd_history, hd_t.unsqueeze(0)], dim=0)
            
            p_gen, p_copy, p_switch = self.pointer_generator(alpha_e,
                                                     c_e,
                                                     hd_t,
                                                     cd_t)
            if tgt is not None:
                tgt_t = tgt[t, :]
                loss_t, pred_t, stats_t = self.ml_crit(p_gen, p_copy, p_switch,
                                                tgt_t,
                                                src)

                stats.update(stats_t)
                loss = loss + loss_t if loss is not None else loss_t
            else:
                scores_t = self.ml_crit.scores(p_gen, p_copy, p_switch, src)
                scores += [scores_t]
                attns += [alpha_e]

            
            if t<input_size-1:
                if self.training:
                    # Exposure bias reduction by feeding predicted token
                    # with a 0.25 probability as mentionned in sect. 6.1:Setup
                    exposure_mask = self.mkvar(torch.rand([bs]).lt(0.25)).long()
                    inputs_t = exposure_mask * pred_t.long()
                    inputs_t += (1-exposure_mask.float()).long() * inputs[t+1, :]
                else:
                    inputs_t = inputs[t+1, :]

        if self.training:
            loss.backward()

        dec_state = onmt.Models.RNNDecoderState(hidden)
        return stats, dec_state, scores, attns


class ReinforcedModel(onmt.Models.NMTModel):
    def __init__(self, encoder, decoder, multigpu=False):
        super(ReinforcedModel, self).__init__(encoder, decoder)
        print("#PARAMS TOTAL: %d"
                % nparams(self))
        print({n: p.nelement() for n,p in self.named_parameters()})
   
    def init_decoder_state(self, enc_hidden, context=None):
        state = super(ReinforcedModel, self).init_decoder_state(enc_hidden=enc_hidden,
                                                    context=context,
                                                    input_feed=True)
        #temp_state = TemporalDecoderState(state.hidden, None)
        return state #temp_state

    def forward(self, src, tgt, dec_state=None):
        """
        Args:
            src:
            tgt:
            dec_state: A decoder state object


        Regular Model (for ref):
            p_gen: (bs x vocab_size)
            p_copy: (bs x src_len)
            
            ##outputs (FloatTensor): (len x batch x rnn_size) -- Decoder outputs.
            ##attns (FloatTensor): Dictionary of (src_len x batch)
            ##dec_hidden (FloatTensor): tuple (1 x batch x rnn_size)
                                      Init hidden state
        """
        bs = src.size(1)
        lengths = None
        n_feats = tgt.size(2)
        assert n_feats == 1, "Reinforced model does not handle features"
        tgt.squeeze_(2)
        enc_hidden, enc_out = self.encoder(src, lengths)  
        
        enc_state = self.init_decoder_state(enc_hidden=enc_hidden, context=enc_out)
        init_state = enc_state if dec_state is None else dec_state
        stats, dec_state, _, _ = self.decoder(tgt[:-1], src, enc_out, init_state, tgt=tgt[1:])
        
        return stats, dec_state
