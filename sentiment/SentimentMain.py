# -*- coding:utf-8 -*-

from collections import OrderedDict
import cPickle
import sys

sys.path.append("./")

from TaggerLSTMSentiment import *
if __name__ == "__main__":
    dataset, Wemb, word_idx_map, vocab, Temb, tagger_idx_map, tvocab = cPickle.load(open("../data/mr.pkl", "rb"))
    train, valid, test = dataset

    options = OrderedDict()
    # set options
    options['word_dim'] = 300
    options['mem_dim'] = 300
    options['y_dim'] = numpy.max(train[1]) + 1
    options["Wemb"] = Wemb
    options["Temb"] = Temb
    options["use_dropout"] = True

    # LSTM
    #senti_lstm = SentimentLSTM(options,model='lstm')

    # tagged-lstm
    senti_lstm = TaggerLSTMSentiment(options, "tagger-lstm")
    max_epochs = 10
    batch_size = 25
    lrate = 0.0001

    senti_lstm.train(dataset, max_epochs, batch_size, lrate)

