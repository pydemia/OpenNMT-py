#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
from __future__ import print_function
from __future__ import division
import six
import sys
import numpy as np
import argparse
import torch
import onmt

parser = argparse.ArgumentParser(description='embeddings_to_torch.py')

##
# **Preprocess Options**
##

parser.add_argument('-emb_file', required=True,
                    help="Embeddings from this file")
parser.add_argument('-output_file', required=True,
                    help="Output file for the prepared data")
parser.add_argument('-dict_file', required=True,
                    help="Dictionary file")
parser.add_argument('-verbose', action="store_true", default=False)
opt = parser.parse_args()


def get_vocabs(dict_file):
    vocabs = torch.load(dict_file)
    src_vocab, tgt_vocab = [vocab[1] for vocab in vocabs]

    print("From: %s" % dict_file)
    print("\t* source vocab: %d words" % len(src_vocab))
    print("\t* target vocab: %d words" % len(tgt_vocab))

    return src_vocab, tgt_vocab


def get_embeddings(file):
    embs = dict()
    for l in open(file, 'rb').readlines():
        l_split = l.decode('utf8').strip().split()
        if len(l_split) == 2:
            continue
        embs[l_split[0]] = [float(em) for em in l_split[1:]]
    print("Got {} embeddings from {}".format(len(embs), file))

    return embs


def match_embeddings(vocab, emb):
    dim = len(six.next(six.itervalues(emb)))
    filtered_embeddings = np.random.rand(len(vocab), dim)
    count = {"match": 0, "miss": 0}
    for w, w_id in vocab.stoi.items():
        if w in emb:
            filtered_embeddings[w_id] = emb[w]
            count['match'] += 1
        else:
            if opt.verbose:
                print(u"not found:\t{}".format(w), file=sys.stderr)
            count['miss'] += 1

    return torch.Tensor(filtered_embeddings), count


def main():
    src_vocab, tgt_vocab = get_vocabs(opt.dict_file)
    embeddings = get_embeddings(opt.emb_file)

    filtered_src_embeddings, src_count = match_embeddings(
        src_vocab, embeddings)
    filtered_tgt_embeddings, tgt_count = match_embeddings(
        tgt_vocab, embeddings)

    print("\nMatching: ")
    match_percent = [_['match']/(_['match']+_['miss'])
                     * 100 for _ in [src_count, tgt_count]]
    print("\t* src: %d match, %d missing, (%.2f%%)" % (src_count['match'],
                                                       src_count['miss'],
                                                       match_percent[0]))
    print("\t* tgt: %d match, %d missing, (%.2f%%)" % (tgt_count['match'],
                                                       tgt_count['miss'],
                                                       match_percent[1]))

    print("\nFiltered embeddings:")
    print("\t* src: ", filtered_src_embeddings.size())
    print("\t* tgt: ", filtered_tgt_embeddings.size())

    src_output_file = "%s.src.pt" % opt.output_file
    tgt_output_file = "%s.tgt.pt" % opt.output_file
    print("\nSaving embedding as:\n\t* src: %s\n\t* tgt: %s" %
          (src_output_file, tgt_output_file))
    torch.save(filtered_src_embeddings, src_output_file)
    torch.save(filtered_tgt_embeddings, tgt_output_file)
    print("\nDone.")


if __name__ == "__main__":
    main()
