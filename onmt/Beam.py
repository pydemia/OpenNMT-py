from __future__ import division
import torch
import onmt

"""
 Class for managing the internals of the beam search process.

 Takes care of beams, back pointers, and scores.
"""


class Beam(object):
    def __init__(self, size, n_best=1, cuda=False, vocab=None,
                 global_scorer=None, avoid_trigram_repetition=False):

        self.size = size
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.allScores = []

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                       .fill_(vocab.stoi[onmt.IO.PAD_WORD])]
        self.nextYs[0][0] = vocab.stoi[onmt.IO.BOS_WORD]
        self.vocab = vocab

        self.hyps = self.nextYs[0].split(1)

        # Has EOS topped the beam yet.
        self._eos = self.vocab.stoi[onmt.IO.EOS_WORD]
        self.eosTop = False

        # The attentions (matrix) for each time.
        self.attn = []

        # Time and k pair for finished.
        self.finished = []
        self.n_best = n_best

        # Information for global scoring.
        self.globalScorer = global_scorer
        self.globalState = {}

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        return self.nextYs[-1]

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk, attnOut):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        def assert_size(tensor, size_list):
            s = list(tensor.size())
            assert s == size_list, ("Incorrect size: ", s, " != ", size_list)
        numWords = wordLk.size(1)

        # nextYs: list(ts)[LongTensor([beam_size])]
        # wordLk: Tensor([beam_size, c_vocab_size])
        t = len(self.nextYs)
        b = self.nextYs[0].size(0)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            if t > 2:
                sentences = torch.stack(self.hyps, 0)  # [beam_size x t]
                assert_size(sentences, [b, t])

                last_bigram = sentences[:, -2:]
                assert_size(last_bigram, [b, 2])

                match_1 = (last_bigram.unsqueeze(2) ==
                           sentences.unsqueeze(1)).float()
                assert_size(match_1, [b, 2, t])

                match_2 = match_1[:, 0, :-1] * match_1[:, 1, 1:]
                assert_size(match_2, [b, t-1])

                def _zeros(size):
                    t = torch.zeros(size)
                    if self.nextYs[-1].is_cuda:
                        return t.cuda()
                    return t

                z = _zeros([b, 2])
                m2 = match_2[:, :-1]
                trigram_candidate_mask = torch.cat([z, m2], 1)
                assert_size(trigram_candidate_mask, [b, t])

                penalty = _zeros(wordLk.size())
                penalty.scatter_add_(
                    1, sentences, trigram_candidate_mask).gt_(0).float()

                # if last two tokens are equal, penalize this token
                last2_eq = (sentences[:, -1] ==
                            sentences[:, -2]).unsqueeze(1).float()
                last1 = sentences[:, -1].contiguous().view(-1, 1)
                penalty.scatter_add_(1, last1, last2_eq)
                assert_size(penalty, list(wordLk.size()))

                penalty.gt_(0)
                penalty *= 1e12

                beamLk -= penalty

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.allScores.append(self.scores)
        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId / numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))
        self.attn.append(attnOut.index_select(0, prevK))

        if self.globalScorer is not None:
            self.globalScorer.updateGlobalState(self)

        nexthyps = []
        for i in range(self.nextYs[-1].size(0)):
            ik = prevK[i]
            prevhyp = self.hyps[ik]
            nextY = self.nextYs[-1][i:i+1]
            nexthyp = torch.cat([prevhyp, nextY])
            nexthyps += [nexthyp]
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                if self.globalScorer is not None:
                    globalScores = self.globalScorer.score(self, self.scores)
                    s = globalScores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))
        self.hyps = nexthyps
        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self.vocab.stoi[onmt.IO.EOS_WORD]:
            # self.allScores.append(self.scores)
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.n_best

    def sortFinished(self, minimum=None):
        if minimum is not None:
            i = 0
            # Add from beam until we have minimum outputs.
            while len(self.finished) < minimum:
                s = self.scores[i]
                if self.globalScorer is not None:
                    globalScores = self.globalScorer.score(self, self.scores)
                    s = globalScores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        self.finished.sort(key=lambda a: -a[0])
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks

    def getHyp(self, timestep, k):
        """
        Walk back to construct the full hypothesis.
        """
        hyp, attn = [], []
        for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
            hyp.append(self.nextYs[j+1][k])
            attn.append(self.attn[j][k])
            k = self.prevKs[j][k]
        return hyp[::-1], torch.stack(attn[::-1])


class GNMTGlobalScorer(object):
    """
    Google NMT ranking score from Wu et al.
    """

    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def score(self, beam, logprobs):
        "Additional term add to log probability"
        cov = beam.globalState["coverage"]
        pen = self.beta * torch.min(cov, cov.clone().fill_(1.0)).log().sum(1)
        l_term = (((5 + len(beam.nextYs)) ** self.alpha) /
                  ((5 + 1) ** self.alpha))
        return (logprobs / l_term) + pen

    def updateGlobalState(self, beam):
        "Keeps the coverage vector as sum of attens"
        if len(beam.prevKs) == 1:
            beam.globalState["coverage"] = beam.attn[-1]
        else:
            beam.globalState["coverage"] = beam.globalState["coverage"] \
                .index_select(0, beam.prevKs[-1]).add(beam.attn[-1])
