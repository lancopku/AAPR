# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 16:04:44 2018

@author: ypc
"""

import codecs
import json as js
from data import  dict


def makeVocabulary(filename, size, sep=' ', char=False):

    vocab = dict.Dict([dict.PAD_WORD, dict.UNK_WORD], lower=True)
    if char:
        vocab.addSpecial(dict.SPA_WORD)

    lengths = []

    if type(filename) == list:
        for _filename in filename:
            with codecs.open(_filename, 'r', 'utf-8') as f:
                data = js.load(f)
                for sent in data:
                    for word in sent.strip().split(sep):
                        lengths.append(len(word))
                        if char:
                            for ch in word.strip():
                                vocab.add(ch)
                        else:
                            vocab.add(word.strip() + " ")
    else:
        with codecs.open(filename, 'r', 'utf-8') as f:
            data = js.load(f)
            for sent in data:
                for word in sent.strip().split(sep):
                    lengths.append(len(word))
                    if char:
                        for ch in word.strip():
                            vocab.add(ch)
                    else:
                        vocab.add(word.strip() + " ")

    print('max: %d, min: %d, avg: %.2f' % (max(lengths), min(lengths), sum(lengths)/len(lengths)))

    originalSize = vocab.size()
    vocab = vocab.prune(size)  
    print('Created dictionary of size %d (pruned from %d)' %
          (vocab.size(), originalSize))

    return vocab


def initVocabulary(name, dataFile, vocabFile, vocabSize, sep=' ', char=False):

    vocab = None
    if vocabFile is not None:
        # If given, load existing word dictionary.
        print('Reading ' + name + ' vocabulary from \'' + vocabFile + '\'...')
        vocab = dict.Dict()
        vocab.loadFile(vocabFile)  
        print('Loaded ' + str(vocab.size()) + ' ' + name + ' words')

    if vocab is None:
        # If a dictionary is still missing, generate it.
        print('Building ' + name + ' vocabulary...')
        genWordVocab = makeVocabulary(dataFile, vocabSize, sep=sep, char=char)  
        vocab = genWordVocab

    return vocab


def saveVocabulary(name, vocab, file):

    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def main():

    dicts = {}
    dicts['text'] = initVocabulary('text',
                                   ['./data/data/abstract_train', './data/data/title_train',
                                    './data/data/intro_train', './data/data/related_train',
                                    './data/data/methods_train', './data/data/conclusion_train'],
                                   None, 50000, ' ', False)
    dicts['authors'] = initVocabulary('authors', './data/data/authors_train',
                                      None, 20000, ',', False)
    
    saveVocabulary('text', dicts['text'], './data/data/text_dict')
    saveVocabulary('authors', dicts['authors'], './data/data/authors_dict')


if __name__ == "__main__":
    main()