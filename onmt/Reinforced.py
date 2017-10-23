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
import onmt.Profiler as prof
from onmt.Profiler import timefunc, Timer
from onmt.modules import CopyGeneratorLossCompute, CopyGenerator


class EachStepGeneratorLossCompute(CopyGeneratorLossCompute):
    def __init__(self, generator, tgt_vocab, dataset, force_copy, eps=1e-20):
        super(EachStepGeneratorLossCompute, self).__init__(generator, tgt_vocab, dataset, force_copy, eps)
        self.tgt_vocab = tgt_vocab

    def remove_oov(self, pred):
        """Remove out-of-vocabulary tokens
           usefull when we wants to use predictions (that contains oov due
           to copy mechanisms) as next input.
           i.e. pred[i] == 0 foreach i such as pred[i] > tgt_vocab_size
        """
        return pred.masked_fill_(pred.gt(len(self.tgt_vocab) - 1), 0)

    def compute_loss(self, batch, output, target, copy_attn, align, src):
        """
            align:      [bs]
            target:     [bs]
            copy_attn:  [bs x src_len]
            output:     [bs x 3*dim]
        """
        t = Timer("loss", prefix=prof.tabs(2), output=prof.DEVNULL)
        align = align.view(-1)
        target = target.view(-1)

        # scores: [bs x vocab + c_vocab]
        scores, switch_scores = self.generator(output,
                                copy_attn,
                                batch.src_map,
                                return_switch=True)
        t.chkpt("generator")
        loss = self.criterion(scores, align, target)
        t.chkpt("criterion")

        _scores_incorrect = scores.data

        # Experimental:
        # fast copy collapse:
        # Dataset.collapse_copy_scores is very usefull in order
        # to sum copy scores for tokens that are in vocabulary
        # but using dataset.collapse_copy_scores at each step is
        # inefficient.
        # It seems incorrect tho...
        """
        _src_map = batch.src_map.float().data.cuda()
        _scores = scores.data.clone()
  
        _src = src.clone().data
        offset = len(self.tgt_vocab)
        src_l, bs, c_vocab = _src_map.size()

        # [bs x src_len], mask of src_idx being in tgt_vocab
        src_invoc_mask = (_src.lt(offset) * _src.ne(1)).float()

        # [bs x c_voc], mask of cvocab_idx related to invoc src token
        cvoc_invoc_mask = src_invoc_mask.unsqueeze(1) \
                                        .bmm(_src_map.transpose(0, 1)) \
                                        .squeeze(1)

        # [bs x src_len], copy scores of invoc src tokens
        src_copy_scores = _scores[:, offset:].unsqueeze(1) \
                                             .bmm(_src_map.transpose(0, 1) \
                                                          .transpose(1, 2)) \
                                             .squeeze()

        # [bs x src_len], invoc src tokens, or 1 (=pad)
        src_token_invoc = _src.clone().masked_fill_(1-src_invoc_mask.byte(), 1)


        _scores.scatter_add_(1, src_token_invoc.long(), src_copy_scores)
        _scores[:, offset:] *= cvoc_invoc_mask

        scores_data = _scores
        """

        scores_data = scores.data.clone()
        scores_data = self.dataset.collapse_copy_scores(
                self.unbottle(scores_data, batch.batch_size),
                batch, self.tgt_vocab)
        scores_data = self.bottle(scores_data)

        t.chkpt("collapse_scores")



        # Correct target is copy when only option.
        # TODO: replace for loop with masking or boolean indexing
        target_data = target.view(-1).data.clone()
        for i in range(target_data.size(0)):
            if target_data[i] == 0:
                if align.data[i] != 0:
                    target_data[i] = align.data[i] + len(self.tgt_vocab)
        t.chkpt("fix_tgt")
        loss_data = loss.data.clone()
        stats = self.stats(loss_data, scores_data, target_data)

        _, pred = scores_data.max(1)
        pred = torch.autograd.Variable(pred)

        # for debugging purpose, it may be interesting to see prediction
        # before "correcting" it
        _, incorrect_pred = _scores_incorrect.max(1)

        pred.cuda()
        t.stop()
        return loss, pred, stats, switch_scores, incorrect_pred


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

        self.softmax = nn.Softmax()

    def forward(self, h_t, h, E_history=None, mask=None,debug=False):
        """
        Args:
            h_t : [bs x dim]
            h   : [n x bs x dim]
            E_history: None or [(t-1) x bs x dim]
        Returns:
            C_t :  [bs x n]
            alpha: [bs x dim]
            E_history: [t x bs x n]
        """
        bs, dim = h_t.size()
        n, _bs, _dim = h.size()
        assert (_bs, _dim)==(bs, dim)


        _h_t = self.linear(h_t).unsqueeze(1)
        _h = h.view(n, bs, dim)

        # e_t = [bs, 1, dim] bmm [bs, dim, n] = [bs, n] (after squeeze)
        scores = _h_t.bmm(_h.transpose(0, 1).transpose(1, 2)).squeeze(1)

        if self.temporal:
            if E_history is None:
                E_history = scores.unsqueeze(0)
            else:
                E_history = torch.cat([E_history, scores.unsqueeze(0)], 0)
                # torch 0.2.0 only run softmax on dim=1 of 2D tensors
                # we want to run temporal softmax (on dim=0) thus we view
                # [t x bs x n] as [bs*n x t]
                hist = E_history.view(-1, bs*n).t()
                scores = self.softmax(hist)[:, -1].contiguous().view(bs, n)

        # [bs, n]
        alpha = self.softmax(scores)
        
        # [bs, 1, n] bmm [n, bs, dim] = [bs, 1, n]
        # [bs, dim, n] bmm [bs, n, 1] = [bs, dim, 1]
        # [bs, 1, n] bmm [bs, n, dim] = [bs, 1, dim]
        C_t = alpha.unsqueeze(1).bmm(_h.transpose(0,1)).squeeze(1)

        if self.temporal:
            return C_t, alpha, E_history
        return C_t, alpha


class PointerGenerator(CopyGenerator):
    def __init__(self, opt, tgt_vocab, embeddings):
        super(PointerGenerator, self).__init__(opt, tgt_vocab)
        self.input_size = opt.rnn_size*3
        self.embeddings = embeddings
        W_emb = embeddings.weight
        #self.W_emb = W_emb
        self.linear = nn.Linear(self.input_size, len(tgt_vocab))
        self.linear_copy = nn.Linear(self.input_size, 1)

        n_emb, emb_dim = list(W_emb.size())

        # (2.4) Sharing decoder weights
        #self.W_proj = nn.Parameter(torch.Tensor(emb_dim, self.input_size))
        #self.emb_proj = nn.Linear(emb_dim, self.input_size, bias=False)
        self.b_out = nn.Parameter(torch.Tensor(n_emb, 1))
        self.tanh = nn.Tanh()

        self._W_out = None


    def W_out(self, force=False):
        """ Sect. (2.4) Sharing decoder weights
            Returns:
                W_out (FloaTensor): [n_emb, 3*dim]
        """
        # eq. (13)
        if self._W_out is None or force:
            _ = self.emb_proj(self.embeddings.weight)
            self._W_out = self.tanh(_)
        return self._W_out

    def linear(self, V, force=False):
        """Calculate the output projection of `v` as in eq. (9)
            Args:
                V (FloatTensor): [bs, 3*dim]
            Returns:
                logits (FloatTensor): logits = W_out * V + b_out, [3*dim]
        """
        #W_out = self.tanh(self.W_emb @ self.W_proj)
        return (self.W_out(force) @ V.t() + self.b_out).t()

class ReinforcedDecoder(_Module):
    def __init__(self, opt, embeddings, bidirectional_encoder=False):
        super(ReinforcedDecoder, self).__init__(opt)
        self.embeddings = embeddings
        W_emb = embeddings.weight
        self.tgt_vocab_size, self.input_size = W_emb.size()
        self.dim = opt.rnn_size

        self.rnn = onmt.modules.StackedLSTM(1, self.input_size, self.dim, opt.dropout)

        self.enc_attn = IntraAttention(opt, self.dim, temporal=True)
        self.dec_attn = IntraAttention(opt, self.dim)

        self.pad_id = embeddings.word_padding_idx

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
        # experimental parameters
        no_dec_attn = False  # does not uses intradec attn if set
        run_profiler = False # profiling (printing execution times)


        dim = self.dim
        src_len, bs, _ = list(src.size())
        input_size, _bs = list(inputs.size())
        assert bs == _bs

        if self.training:
            loss_compute.generator.W_out(True)
            assert tgt is not None
        if tgt is not None:
            assert loss_compute is not None
            if generator is not None:
                print("[WARNING] Parameter 'generator' has been set in decoder but it won't be used")
        else:
            assert generator is not None

        # src as [bs x src_len]
        src = src.transpose(0, 1).squeeze(2).contiguous()

        stats = onmt.Statistics()
        hidden = state.hidden
        loss, E_hist = None, None
        scores, attns, dec_attns, switchs, ipreds = [], [], [], [], []
        preds = []
        inputs_t = inputs[0, : ]

        devout_timer = prof.STDOUT if run_profiler else prof.DEVNULL
        gtimer = Timer("global_decoder", output=devout_timer)
        timer = Timer("decoder", output=devout_timer, prefix=prof.tabs())
        for t in range(input_size):
            src_mask = src.eq(self.pad_id)
            emb_t = self.embeddings(inputs_t.view(1, -1, 1)).squeeze(0)
            hd_t, hidden = self.rnn(emb_t, hidden)
            timer.chkpt("encoder")
            c_e, alpha_e, E_hist = self.enc_attn(hd_t, h_e, E_history=E_hist)
            if t==0 or no_dec_attn:
                # no decoder intra attn at first step
                cd_t = self.mkvar(torch.zeros([bs, dim]))
                alpha_d = cd_t
                hd_history = hd_t.unsqueeze(0)
            else:
                cd_t, alpha_d = self.dec_attn(hd_t, hd_history)
                hd_history = torch.cat([hd_history, hd_t.unsqueeze(0)], dim=0)

            timer.chkpt("decoder")
            
            if tgt is not None:
                tgt_t = tgt[t, :]
                output = torch.cat([hd_t, c_e, cd_t], dim=1)
                loss_t, pred_t, stats_t, switch_t, i_pred_t = loss_compute(batch,
                    output,
                    tgt_t,
                    copy_attn=alpha_e,
                    align=batch.alignment[t, :].contiguous(),
                    src=src)
                attns += [alpha_e]
                switchs += [switch_t]
                preds += [pred_t]
                ipreds += [i_pred_t]

                stats.update(stats_t)
                loss = loss + loss_t if loss is not None else loss_t
            else:
                output = torch.cat([hd_t, c_e, cd_t], dim=1)
                scores_t = generator(output, alpha_e, batch.src_map)
                scores += [scores_t]
                attns += [alpha_e]
                dec_attns += [alpha_d]
            timer.chkpt("loss&pred")

            if t<input_size-1:
                if self.training:
                    # Exposure bias reduction by feeding predicted token
                    # with a 0.25 probability as mentionned in sect. 6.1:Setup
                    _pred_t = preds[-1].clone()
                    _pred_t = loss_compute.remove_oov(_pred_t)
                    exposure_mask = self.mkvar(torch.rand([bs]).lt(0.25).long())
                    inputs_t = exposure_mask * _pred_t.long()
                    inputs_t += (1-exposure_mask.float()).long() * inputs[t+1, :]
                else:
                    inputs_t = inputs[t+1, :]
            timer.chkpt("next_input")
            gtimer.chkpt("step: %d" % t, append="\n")

        if self.training:
            loss.backward()
            #print("inp/tgt/pred")
            pred0 = torch.stack(preds, 0)[:, 0]
            ipred0 = torch.stack(ipreds, 0)[:, 0]
            #print(torch.stack([inputs[:, 0], tgt[:, 0], pred0, ipred0], 1))
            #print("copy")
            #print(torch.stack(switchs, 0)[:, 0])
            #print("attn")
            #print(torch.stack(attns, 0)[:, 0, :])

        gtimer.stop("backward", append="\n")
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
