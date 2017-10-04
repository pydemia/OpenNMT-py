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
import onmt.Trainer
import onmt.Loss

from onmt.modules import CopyGeneratorLossCompute, CopyGenerator

class EachStepGeneratorLossCompute(CopyGeneratorLossCompute):
    def __init__(self, generator, tgt_vocab, dataset, force_copy, eps=1e-20):
        super(EachStepGeneratorLossCompute, self).__init__(generator, tgt_vocab, dataset, force_copy, eps)

    def compute_loss(self, batch, output, target, copy_attn, align):
        align = align.view(-1)
        target = target.view(-1)

        scores = self.generator(output,
                                copy_attn,
                                batch.src_map)
        n = target.size(0)
        loss = self.criterion(scores, align, target)
        scores_data = scores.data.clone()
        scores_data = self.dataset.collapse_copy_scores(
                self.unbottle(scores_data, batch.batch_size),
                batch, self.tgt_vocab)
        scores_data = self.bottle(scores_data)


        # Correct target is copy when only option.
        # TODO: replace for loop with masking or boolean indexing
        target_data = target.view(-1).data.clone()
        for i in range(target_data.size(0)):
            if target_data[i] == 0:
                if align.data[i] != 0:
                    target_data[i] = align.data[i] + len(self.tgt_vocab)

        loss_data = loss.data.clone()
        stats = self.stats(loss_data, scores_data, target_data)

        _, pred = scores.max(1)

        return loss, pred, stats


class RTrainer(onmt.Trainer.Trainer):
    """Special Trainer for the Reinforced Model
       The loss is directly calculated in the decoder because
       it needs to predict output at each time steps
       the training process is therefore slightly different
    """
    def __init__(self, model, train_iter, valid_iter, train_loss, valid_loss, optim, trunc_size):
        self.model = model
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.trunc_size = trunc_size

        # Set model in training mode.
        self.model.train()

    def train(self, epoch, report_func=None):
        total_stats = onmt.Statistics()
        report_stats = onmt.Statistics()

        for i, batch in enumerate(self.train_iter):
            target_size = batch.tgt.size(0)
            # Truncated BPTT
            trunc_size = self.trunc_size if self.trunc_size else target_size

            dec_state = None
            _, src_lengths = batch.src

            src = onmt.IO.make_features(batch, 'src')
            tgt_outer = onmt.IO.make_features(batch, 'tgt')
            report_stats.n_src_words += src_lengths.sum()
            alignment = batch.alignment

            for j in range(0, target_size-1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]
                batch.alignment = alignment[j + 1: j + trunc_size]

                # 2. & 3. F-prop and compute loss
                self.model.zero_grad()
                batch_stats, dec_state = \
                    self.model(src, tgt, src_lengths, batch, self.train_loss, dec_state)

                # 4. Update the parameters and statistics.
                self.optim.step()
                total_stats.update(batch_stats)
                report_stats.update(batch_stats)

                # If truncated, don't backprop fully.
                if dec_state is not None:
                    dec_state.detach()

            if report_func is not None:
                report_func(epoch, i, len(self.train_iter),
                            total_stats.start_time, self.optim.lr,
                            report_stats)
                report_stats = onmt.Statistics()

        return total_stats

    def validate(self):
        """ Called for each epoch to validate. """
        # Set model in validating mode.
        self.model.eval()

        stats = onmt.Statistics()

        for batch in self.valid_iter:
            _, src_lengths = batch.src
            src = onmt.IO.make_features(batch, 'src')
            tgt = onmt.IO.make_features(batch, 'tgt')

            batch.alignment = batch.alignment[1:]
            # F-prop through the model.
            batch_stats, _ = self.model(src, tgt, src_lengths, batch, self.valid_loss)
            # Update statistics.
            stats.update(batch_stats)

        # Set model back to training mode.
        self.model.train()

        return stats

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


class PointerGenerator(CopyGenerator):
    def __init__(self, opt, tgt_vocab, embeddings):
        #TODO: use embeddings for out projection
        # require PartiallySharedEmbeddings
        super(PointerGenerator, self).__init__(opt, tgt_vocab)
        self.input_size = opt.rnn_size*3
        self.W_emb = embeddings.word_lut.weight

        self.linear = nn.Linear(self.input_size, len(tgt_vocab))
        self.linear_copy = nn.Linear(self.input_size, 1)

        n_emb, emb_dim = list(self.W_emb.size())

        # (2.4) Sharing decoder weights
        #self.W_proj = nn.Parameter(torch.Tensor(emb_dim, self.input_size))
        #self.b_out = nn.Parameter(torch.Tensor(n_emb, 1))
        self.tanh = nn.Tanh()

    @property
    def W_out(self):
        """ Sect. (2.4) Sharing decoder weights
            Returns:
                W_out (FloaTensor): [n_emb, 3*dim]
        """
        # eq. (13)
        return self.tanh(self.W_emb @ self.W_proj)

    def linear_shared_emb(self, V):
        """Calculate the output projection of `v` as in eq. (9)
            Args:
                V (FloatTensor): [bs, 3*dim]
            Returns:
                logits (FloatTensor): logits = W_out * V + b_out, [3*dim]
        """
        return (self.W_out @ V.t() + self.b_out).t()


class ReinforcedDecoder(_Module):
    def __init__(self, opt, embeddings, bidirectional_encoder=False):
        super(ReinforcedDecoder, self).__init__(opt)
        self.embeddings = embeddings
        W_emb = embeddings.word_lut.weight
        self.tgt_vocab_size, self.input_size = W_emb.size()
        self.dim = opt.rnn_size

        self.rnn = onmt.modules.StackedLSTM(opt.layers, self.input_size, opt.rnn_size, opt.dropout)

        self.enc_attn = IntraAttention(opt, self.dim, temporal=True)
        self.dec_attn = IntraAttention(opt, self.dim)

        self.pad_id = embeddings.word_padding_idx
        self.ml_crit = MLCriterion(self.tgt_vocab_size, opt, self.pad_id)

        # For compatibility reasons, TODO refactor
        self.hidden_size = self.dim
        self.decoder_type = "reinforced"
        self.bidirectional_encoder = bidirectional_encoder

    def _fix_enc_hidden(self, h):
        """
        The encoder hidden is  (layers*directions) x batch x dim.
        We need to convert it to layers x batch x (directions*dim).
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def init_decoder_state(self, src, context, enc_hidden):
        """
        Args:
            src: For compatibility reasons.......

        """
        if isinstance(enc_hidden, tuple):  # GRU
            return onmt.Models.RNNDecoderState(context, self.hidden_size,
                                   tuple([self._fix_enc_hidden(enc_hidden[i])
                                         for i in range(len(enc_hidden))]))
        else:  # LSTM
            return onmt.Models.RNNDecoderState(context, self.hidden_size,
                                   self._fix_enc_hidden(enc_hidden))

    def forward(self, inputs, src, h_e, state, batch, loss_compute=None, tgt=None, generator=None):
        """
        Args:
            inputs (LongTensor): [tgt_len x bs]
            src (LongTensor): [src_len x bs x 1]
            h_e (FloatTensor): [src_len x bs x dim]
            state: onmt.Models.DecoderState
            src_map: batch src (see IO)
            tgt (LongTensor): [tgt_len x bs]

        Returns:
            stats: onmt.Statistics
            hidden
            None: TODO refactor
            None: TODO refactor
        """
        dim = self.dim
        src_len, bs, _ = list(src.size())
        input_size, _bs = list(inputs.size())
        assert bs == _bs

        if self.training:
            assert tgt is not None
        if tgt is not None:
            assert loss_compute is not None
            if generator is not None:
                print("[WARNING] Parameter 'generator' has been set in decoder but it won't be used")
        else:
            assert generator is not None

        # src as [bs x src_len]
        src = src.transpose(0, 1).squeeze(2)

        stats = onmt.Statistics()
        hidden = state.hidden
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
            
            if tgt is not None:
                tgt_t = tgt[t, :]
                output = torch.cat([hd_t, c_e, cd_t], dim=1)
                loss_t, pred_t, stats_t = loss_compute(batch, output, tgt_t, copy_attn=alpha_e, align=batch.alignment[t, :].contiguous())

                stats.update(stats_t)
                loss = loss + loss_t if loss is not None else loss_t
            else:
                output = torch.cat([hd_t, c_e, cd_t], dim=1)
                scores_t = generator(output, alpha_e, batch.src_map)
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

        state.update_state(hidden, None, None)
        return stats, state, scores, attns


class ReinforcedModel(onmt.Models.NMTModel):
    def __init__(self, encoder, decoder, multigpu=False):
        super(ReinforcedModel, self).__init__(encoder, decoder)

    def forward(self, src, tgt, src_lengths, batch, loss_compute, dec_state=None):
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
        n_feats = tgt.size(2)
        assert n_feats == 1, "Reinforced model does not handle features"
        tgt = tgt.squeeze(2)
        enc_hidden, enc_out = self.encoder(src, src_lengths)
        
        enc_state = self.decoder.init_decoder_state(src=None, enc_hidden=enc_hidden, context=enc_out)
        state = enc_state if dec_state is None else dec_state
        stats, hidden, _, _ = self.decoder(tgt[:-1], src, enc_out, state, batch, loss_compute, tgt=tgt[1:])
        
        return stats, state

class DummyGenerator:
    """Hacky way to ensure compatibility
    """
    def dummy_pass(self, *args, **kwargs):
        pass
    def __init__(self, *args, **kwargs):
        self.state_dict = self.dummy_pass
        self.cpu = self.dummy_pass
        self.cuda = self.dummy_pass
        self.__call__ = self.dummy_pass
        self.load_state_dict = self.dummy_pass

    def __getattr__(self, attr):
        class DummyCallableObject:
            def __init__(self, *args, **kwargs):
                pass
            def __call__(self, *args, **kwargs):
                pass
        return DummyCallableObject()
