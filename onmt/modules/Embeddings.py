import torch
import torch.nn as nn
from torch.autograd import Variable

from onmt.modules import BottleLinear, Elementwise
from onmt.Utils import aeq


class PositionalEncoding(nn.Module):

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.arange(0, max_len).unsqueeze(1).expand(max_len, dim)
        div_term = 1 / torch.pow(10000, torch.arange(0, dim * 2, 2) / dim)
        pe = pe * div_term.expand_as(pe)
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, emb):
        # We must wrap the self.pe in Variable to compute, not the other
        # way - unwrap emb(i.e. emb.data). Otherwise the computation
        # wouldn't be watched to build the compute graph.
        emb = emb + Variable(self.pe[:emb.size(0), :1, :emb.size(2)]
                             .expand_as(emb), requires_grad=False)
        emb = self.dropout(emb)
        return emb


class Embeddings(nn.Module):
    """
    Words embeddings dictionary for encoder/decoder.

    Args:
        word_vec_size (int): size of the dictionary of embeddings.
        position_encoding (bool): use a sin to mark relative words positions.
        feat_merge (string): merge action for the features embeddings:
                    concat, sum or mlp.
        feat_vec_exponent (float): when using '-feat_merge concat', feature
                    embedding size is N^feat_dim_exponent, where N is the
                    number of values of feature takes.
        feat_vec_size (int): embedding dimension for features when using
                    '-feat_merge mlp'
        dropout (float): dropout probability.
        word_padding_idx (int): padding index for words in the embeddings.
        feats_padding_idx ([int]): padding index for a list of features
                                   in the embeddings.
        word_vocab_size (int): size of dictionary of embeddings for words.
        feat_vocab_sizes ([int], optional): list of size of dictionary
                                    of embeddings for each feature.
    """
    def __init__(self, word_vec_size, position_encoding, feat_merge,
                 feat_vec_exponent, feat_vec_size, dropout,
                 word_padding_idx, feat_padding_idx,
                 word_vocab_size, feat_vocab_sizes=[]):

        self.word_padding_idx = word_padding_idx

        # Dimensions and padding for constructing the word embedding matrix
        vocab_sizes = [word_vocab_size]
        emb_dims = [word_vec_size]
        pad_indices = [word_padding_idx]

        # Dimensions and padding for feature embedding matrices
        # (these have no effect if feat_vocab_sizes is empty)
        if feat_merge == 'sum':
            feat_dims = [word_vec_size] * len(feat_vocab_sizes)
        elif feat_vec_size > 0:
            feat_dims = [feat_vec_size] * len(feat_vocab_sizes)
        else:
            feat_dims = [int(vocab ** feat_vec_exponent)
                         for vocab in feat_vocab_sizes]
        vocab_sizes.extend(feat_vocab_sizes)
        emb_dims.extend(feat_dims)
        pad_indices.extend(feat_padding_idx)

        # The embedding matrix look-up tables. The first look-up table
        # is for words. Subsequent ones are for features, if any exist.
        emb_params = zip(vocab_sizes, emb_dims, pad_indices)
        embeddings = [nn.Embedding(vocab, dim, padding_idx=pad)
                      for vocab, dim, pad in emb_params]
        emb_luts = Elementwise(feat_merge, embeddings)

        # The final output size of word + feature vectors. This can vary
        # from the word vector size if and only if features are defined.
        # This is the attribute you should access if you need to know
        # how big your embeddings are going to be.
        self.embedding_size = (sum(emb_dims) if feat_merge == 'concat'
                               else word_vec_size)

        # The sequence of operations that converts the input sequence
        # into a sequence of embeddings. At minimum this consists of
        # looking up the embeddings for each word and feature in the
        # input. Model parameters may require the sequence to contain
        # additional operations as well.
        super(Embeddings, self).__init__()
        self.make_embedding = nn.Sequential()
        self.make_embedding.add_module('emb_luts', emb_luts)

        if feat_merge == 'mlp':
            in_dim = sum(emb_dims)
            out_dim = word_vec_size
            mlp = nn.Sequential(BottleLinear(in_dim, out_dim), nn.ReLU())
            self.make_embedding.add_module('mlp', mlp)

        if position_encoding:
            pe = PositionalEncoding(dropout, self.embedding_size)
            self.make_embedding.add_module('pe', pe)

    @property
    def weight(self):
        return self.word_lut.weight

    @property
    def word_lut(self):
        return self.make_embedding[0][0]

    @property
    def emb_luts(self):
        return self.make_embedding[0]

    def load_pretrained_vectors(self, emb_file, fixed):
        if emb_file:
            pretrained = torch.load(emb_file)
            self.word_lut.weight.data.copy_(pretrained)
            if fixed:
                self.word_lut.weight.requires_grad = False

    def forward(self, input):
        """
        Return the embeddings for words, and features if there are any.
        Args:
            input (LongTensor): len x batch x nfeat
        Return:
            emb (FloatTensor): len x batch x self.embedding_size
        """
        in_length, in_batch, nfeat = input.size()
        aeq(nfeat, len(self.emb_luts))

        emb = self.make_embedding(input)

        out_length, out_batch, emb_size = emb.size()
        aeq(in_length, out_length)
        aeq(in_batch, out_batch)
        aeq(emb_size, self.embedding_size)

        return emb


class PartialEmbedding(nn.Embedding):
    def __init__(self, partial_num_embeddings, embedding, padding_idx):
        self.partial_num_embeddings = partial_num_embeddings
        self.nspe = 4
        
        super(PartialEmbedding, self).__init__(partial_num_embeddings,
                                               embedding.embedding_size,
                                               padding_idx)
        if self.nspe == 2:
            self.spe = nn.Parameter(torch.Tensor(2, embedding.embedding_size))
        elif self.nspe == 4:
            self.spe = nn.Parameter(torch.Tensor(4, embedding.embedding_size))
        else: 
            raise ValueError("Incorrect value for nspe")
        
        self.full_embedding = embedding

        
    @property
    def weight(self):
        return self._weight()

    def _weight(self):
        # The partial embeddings has `self.partial_num_embeddings` tokens 
        # including 4 special tokens.
        # source embeddings:
        # [unk+bos+eos, pad, src#2, src#3, ..., src#{src_num_embeddings}]
        # partial embeddings:
        # [unk, pad, bos, eos, src#2, src#3, src#{partial_num_embeddings-2]
    
        w = None
        try:
            #print(self.partial_num_embeddings)

            shared = self.full_embedding.weight[:self.partial_num_embeddings-2, :]
           
            
            if self.nspe == 2:
                # keep 2 first tokens -- and insert two others
                w = torch.cat([shared[:2, :], self.spe, shared[2:, :] ]).contiguous()
            elif self.nspe == 4:
                # replace first tokens -- and add two <=> prepend with 4 tokens
                w = torch.cat([self.spe, shared[2:, :]])
            else:
                raise ValueError("Incorrect value for nspe")
            #print("part.embd.size: ", w.size())
            return w
        except AttributeError as e:
            print(e)
            import traceback
            traceback.print_exc()
            raise ValueError()
        
        def assert_size(var, sizes):
            expected = "[%s]" % ", ".join(list(sizes))
            actual = "[%s]" % ", ".join(list(var.size()))
            assert list(var.size()) == list(sizes), "Incorrect size expected %s got %s" % (expected, actual)

        assert_size(w, [self.partial_num_embeddings, self.embedding_size])
        return w

    def reset_parameters(self):
        pass

    @weight.setter
    def weight(self, val):
        """Partial Embedding does not have its own weight matrix
        """
        pass

    @property
    def word_padding_idx(self):
        return self.padding_idx

    @word_padding_idx.setter
    def word_padding_idx(self, val):
        self.padding_idx = val

    def load_pretrained_vectors(self, emb_file, fixed):
        """Nothing to do but loading the "full_embedding"
        """
        pass

    def forward(self, input):
        """
        Return the embeddings for words
        Args:
            input (LongTensor): len x batch x nfeat
        Return:
            emb (FloatTensor): len x batch x self.embedding_size
        Raise:
            AssertionError if nfeat != 1
        """
        l, bs, nfeat = input.size()
        assert nfeat == 1, "PartialEmbedding don't handle features"

        _input = input.squeeze(2).t()
        _emb = super(PartialEmbedding, self).forward(_input)
        emb = _emb.transpose(0, 1)

        def nonan(variable):
            st = variable.data
            if not (st != st).sum() == 0:
                print("NaN values emb")
                print("inp: ", input)
                print("emb: ", emb)
                print("spe: ", self.spe)
                print("full: ", self.full_embedding.weight)
                raise ValueError()
        nonan(emb)

        _l, _bs, _emb_size = emb.size()
        assert l == _l
        assert _bs == bs
        assert _emb_size == self.embedding_dim

        return emb
