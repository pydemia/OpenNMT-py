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

def _friendly_aeq(real, expected):
    assert real == expected, "got %s expected: %s" % (str(real), str(expected))

class IntraAttention(nn.Module):
    def __init__(self, dim, temporal=False):
        super(IntraAttention, self).__init__()
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
            h   (FloatTensor): batch x n x dim
                for encoder_attn n is as in eq. (4), (5) i.e. src_len
                for decoder_attn n is t-1

        returns:
            alpha: [b x src]
            c_t: [b x dim]
        """
        if self.past_e is None:
            self.past_e = torch.zeros(h.size(0), 1, h.size(1))
        if self.past_ee is None:
            self.past_ee = torch.zeros(h.size(0), 1, h.size(1))

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
            else:
                # e_prev is [b x (t-1) x src]
                # d [b x 1 x src]
                ee_t = e_t.exp().div(past_e)
            past_e.add_(e_t.exp())
        else:
            ee_t = e_t
        
        # Equation (4)
        # alpha_t = [b x 1 x src]
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
        _alpha_t = alpha_t.transpose(1, 2)
        s = _alpha_t.size()

        # [b, src, src]
        _alpha_t.expand_(s[0], s[1], s[1])

        # Equation (5) 
        # [b, src, src] bmm [b, src, dim] = [b, src, dim] {a}
        # sum( (a) , dim=1) = [b x 1 x dim] {b}
        # squeeze( {b} ) = [b x dim]
        c_t = torch.sum(_alpha_t.bmm(h), dim=1).squeeze()

        # [b x src]
        alpha_t.squeeze_()
        return alpha_t, c_t

class PointerGenerator(nn.Module):
    def __init__(self, dim, W_emb):
        super(PointerGenerator, self).__init__()
        self.dim = dim
        
        self.W_emb = W_emb
        in_dim, out_dim = W_emb.size(0), 3*dim

        self.W_proj = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.b_out = nn.Parameter(torch.Tensor(out_dim, 1))

        self.W_u = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.b_u = nn.Parameter(torch.Tensor(out_dim, 1))


        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        pass

    def forward(self, x, ae_t, ce_t, hd_t, cd_t):
        """
        Implementing sect. 2.3: "Token generation and pointer"
        (and sect. 2.4 about weights sharing)
        Args:
            x: [bs x src x 1]
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
          out = p(y_t), as in eq. 12
            probably a [b x t x dim] matrix of probabilities
            sum(out, dim=2) = ones(b, t)
        """
        
        
        bs = x.size(0)
        src_len = x.size(1)
        n_emb = self.W_emb.size(0)
        dim = ae_t.size(1)

        assert list(x.size()) == [bs, src_len, 1]
        assert list(ae_t.size()) == [bs, src_len]
        assert list(ce_t.size()) == [bs, dim]
        assert list(hd_t.size()) == [bs, dim]
        assert list(cd_t.size()) == [bs, dim]

        # W_emb: [n_emb x emb_dim]
        # W_proj:[emb_dim x 3*dim]
        # W_out: [n_emb x 3*dim]
        # then expand it to:
        # W_out: [bs, n_emb, 3*dim]
        W_out = self.tanh(self.W_emb, self.W_proj).unsqueeze(0).expand([bs, n_emb, 3*dim])
        b_out = self.b_out

        # concatenate contexts & current state and make it
        # c_cat: [bs x 3*dim x 1]
        c_cat = torch.cat([hd_t, ce_t, cd_t], 1).squeeze(2)

        # in (9) [hd_t | ce_t | cd_t] is [3*dim x 1]
        # here: [bs x 3*dim x 1]
        # p_gen(y_t|u_t=0) = softmax(W_out x [hd_t | ce_t | cd_t] +b_out )
        p0 = self.softmax(torch.bmm(W_out, c_cat)+b_out)

        
        # (10)
        # p_copy_ti(y_t=x_i) = alpha_e_ti
        p1 = torch.zeros([bs, n_emb, 1])
        for _b in range(bs):
            for _s in range(src_len):
                _w = x[_b, _s]
                p1[_b, _w] = ae_t[_b, _s]

        # (11)
        # p_copy = p(u_t=1) = sigmoid(W_u [hd_t | ce_t | cd_t] + b_u)
        W_u = self.W_u.unsqueeze(0).expand([bs, n_emb, 3*dim])
        b_u = self.b_u
        p_u = self.sigmoid(torch.bmm(W_u, c_cat)+b_u)
        
    
        p_y = p_u * p1 + (1-p_u) * p0
        return p_y

class Encoder(nn.Module):
    def __init__(self, opt):
        pass

    def forward(self):
        
        pass

class ReinforcedDecoder(nn.Module):
    def __init__(self, opt, embeddings):
        super(ReinforcedDecoder, self).__init__()
        self.input_size = opt.word_vec_size
        
        self.dim = opt.rnn_size * 2
        self.rnn = onmt.modules.StackedLSTM(1, self.input_size, opt.rnn_size, opt.dropout)

        self.enc_attn = IntraAttention(self.dim, temporal=True)
        self.dec_attn = IntraAttention(self.dim)

        W_emb = embeddings.word_lut.weight
        self.pointer_generator = PointerGenerator(self.dim, W_emb)
        self.embeddings = embeddings
        pass

    def forward(self, inputs, src, h_e, ):
        """
        Args:
            input: tgt_len x batch -- tgt tokens
            src: src_len x batch
            h_e: src_len x batch x dim

        Returns:
            outputs
            dec_state
        
        """
        _friendly_aeq(h_e.size(2), self.dim)

        emb = self.embeddings(inputs)
        hd_history = []
        outputs = []
        for t, emb_t in enumerate(emb.split(1)):
            out, hidden = self.rnn(emb_t, hidden)
            hd_t = hidden[0]
            alpha_e, c_e = self.enc_attn(hd_t, h_e)
            
            if t>0:
                cd_t = torch.zeros([bs, dim])
            else:
                alpha_d, cd_t = self.dec_attn(hd_t, hd_history)
            
            p = self.pointer_generator(inputs[t], alpha_e, c_e, hd_t, cd_t)
            outputs += [p]
            hd_history += [hd_t]

        return  torch.stack(outputs, dim=0), hd_t

class ReinforcedModel(nn.Module):
    def __init__(self, encoder, decoder, multigpu=False):
        self.multigpu = multigpu
        super(ReinforcedModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def _fix_enc_hidden(self, h):
        """
        The encoder hidden is  (layers*directions) x batch x dim
        We need to convert it to layers x batch x (directions*dim)
        """
        if self.encoder.num_directions == 2:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def init_decoder_state(self, context, enc_hidden):
        if self.decoder.decoder_layer == "transformer":
            return TransformerDecoderState()
        elif isinstance(enc_hidden, tuple):
            dec = RNNDecoderState(tuple([self._fix_enc_hidden(enc_hidden[i])
                                         for i in range(len(enc_hidden))]))
        else:
            dec = RNNDecoderState(self._fix_enc_hidden(enc_hidden))
        dec.init_input_feed(context, self.decoder.hidden_size)
        return dec

    def forward(self, src, tgt, lengths, dec_state=None):
        """
        Args:
            src, tgt, lengths
            dec_state: A decoder state object

        Returns:
            outputs (FloatTensor): (len x batch x rnn_size) -- Decoder outputs.
            _: None*
            __: None*
            (*) in order to keep valid the syntax `out, attns, dec_h = model(___)`

        Regular Model (for ref):
            outputs (FloatTensor): (len x batch x rnn_size) -- Decoder outputs.
            attns (FloatTensor): Dictionary of (src_len x batch)
            dec_hidden (FloatTensor): tuple (1 x batch x rnn_size)
                                      Init hidden state
        """
        src = src
        tgt = tgt[:-1]  # exclude last target from inputs
        #encoder
        #in: input (LongTensor): len x batch x nfeat, lengths (LongTensor): batch, hidden: Initial hidden state.
        #out: hidden_t pair of layers x batch x rnn_size, outputs:  len x batch x rnn_size
        enc_hidden, context = self.encoder(src, lengths)
        
        
        """enc_state = self.init_decoder_state(context, enc_hidden)"""

        #decoder:
        #in     input: tgt_len x batch, 
        #       src: src_len x batch, 
        #       h_e: src_len x batch x dim
        #
        #out    outputs, dec_state
        out, dec_state, attns = self.decoder(tgt, src, context)

        #if self.multigpu:
        #    # Not yet supported on multi-gpu
        #    dec_state = None
        #    attns = None
        _ = None
        return out, _, _

