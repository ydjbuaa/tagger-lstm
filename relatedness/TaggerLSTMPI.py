# -*- coding:utf -*-
from models.models import *
from models.optimizer import *
from theano import config
import theano.tensor as tensor
from scipy import dot, linalg
import time
import numpy
def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)

def  prepare_data(seqsA, seqsB, labels):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    # x: a list of sentences
    lengthsA = [len(s) for s in seqsA]
    lengthsB = [len(s) for s in seqsB]

    n_samples = len(seqsA)
    maxlen = numpy.max(lengthsA)

    sentA = numpy.zeros((maxlen, n_samples)).astype('int64')
    sentA_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)

    for idx, s in enumerate(seqsA):
        sentA[:lengthsA[idx], idx] = s
        sentA_mask[:lengthsA[idx], idx] = 1.

    n_samples = len(seqsB)
    maxlen = numpy.max(lengthsB)

    sentB = numpy.zeros((maxlen, n_samples)).astype('int64')
    sentB_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)

    for idx, s in enumerate(seqsB):
        sentB[:lengthsB[idx], idx] = s
        sentB_mask[:lengthsB[idx], idx] = 1.

    y = numpy.asarray(labels, dtype="int32")
    return sentA, sentB, sentA_mask, sentB_mask, y
def cul_cosin_distance(x1, x2):
    return dot(x1, x2.T)/ linalg.norm(x1) / linalg.norm(x2)

class SentenceModel(object):
    def __init__(self,options, model):
        self.word_dim = options['word_dim']
        self.mem_dim = options['mem_dim']
        self.y_dim = options['y_dim']
        self.model = model

        print ("Building Model:", model)

        self.lstm_layer = LSTM(self.word_dim, self.mem_dim, prefix='lstm')


    @property
    def params(self):
        return self.lstm_layer.params

    def layer_output(self, use_noise, emb, mask):
        hc_state = self.lstm_layer.layer_output(emb, mask)
        hc_state = (hc_state * mask[:, :, None]).sum(axis=0)
        # if xs_mask.sum(axis=0)[:, None] > 0:
        hc_state = hc_state / mask.sum(axis=0)[:, None]
        hc_state = DropoutLayer(state_before=hc_state, use_noise=use_noise).drop_out

        return hc_state



class TaggerLSTMPI(object):
    """
    tagger-lstm or lstm used for paraphrase identification
    """
    def __init__(self, options, model):
        """
            build model

            """
        self.word_dim = options['word_dim']
        self.mem_dim = options['mem_dim']
        self.y_dim = options['y_dim']
        self.params = []
        self.model = model

        print ("Building Model:", model)

        # variables
        x_a = tensor.matrix('sentence_a', dtype='int64')
        a_mask = tensor.matrix('a_mask', dtype=config.floatX)

        x_b = tensor.matrix('sentence_b', dtype='int64')
        b_mask = tensor.matrix('b_mask', dtype=config.floatX)

        y = tensor.vector('y', dtype='int64')


        x_vars = [x_a, x_b, a_mask, b_mask]
        y_vars = [x_a, x_b, a_mask, b_mask, y]



        # add word embeddings to params list
        Wembs = theano.shared(options['Wemb'], "Wemb")
        self.params.append(Wembs)

        a_emb = Wembs[x_a.flatten()].reshape([x_a.shape[0], x_a.shape[1], self.word_dim])
        b_emb = Wembs[x_b.flatten()].reshape([x_b.shape[0], x_b.shape[1], self.word_dim])

        self.sentence_model_layer = SentenceModel(options, "SentenceModeling")
        for param in self.sentence_model_layer.params:
            self.params.append(param)

        # Used for dropout.
        self.use_noise = theano.shared(numpy_floatX(0.))

        sent_a = self.sentence_model_layer.layer_output(self.use_noise, a_emb, a_mask)
        sent_b = self.sentence_model_layer.layer_output(self.use_noise, b_emb, b_mask)


        n_samples = x_a.shape[0]

        cos_distance = abs(sent_a - sent_b)
        #for i in range(n_samples):
        #   cos_distance[i] = cul_cosin_distance(sent_a[i], sent_b[i])

        self.lr_layer = LogisticRegression(cos_distance, self.mem_dim, self.y_dim, model+"_lr")

        pred = self.lr_layer.p_y_given_x

        f_pred_prob = theano.function(x_vars, pred, name='f_pred_prob')
        self.f_pred = theano.function(x_vars, pred.argmax(axis=1), name='f_pred')

        log_likelihood_cost = self.lr_layer.negative_log_likelihood(y)

        cost = log_likelihood_cost  # + 0.5 * 0.0001 * l2_sqr

        print('Optimization')
        f_cost = theano.function(y_vars, cost, name='f_cost')

        grads = tensor.grad(cost, self.params)
        f_grad = theano.function(y_vars, grads, name='f_grad')

        lr = tensor.scalar(name='lr')

        self.f_grad_shared, self.f_update = adadelta(lr, self.params, grads, y_vars, cost)

    def pred_error(self, prepare_data, data, iterator, verbose=False):
        """
        Just compute the error
        f_pred: Theano fct computing the prediction
        prepare_data: usual prepare_data for that dataset.
        """
        valid_err = 0
        for _, valid_index in iterator:
            sentA, sentB, a_mask, b_mask, y = prepare_data([data[0][t][0] for t in valid_index],
                                                           [data[0][t][1] for t in valid_index],
                                                           numpy.array(data[1], dtype="int32")[valid_index])
            preds = self.f_pred(sentA, sentB, a_mask, b_mask)
            targets = y
            valid_err += (preds == targets).sum()

        valid_err = 1. - numpy_floatX(valid_err) / len(data[0])

        return valid_err

    def train(self, dataset, max_epochs, batch_size, lrate, dispFreq=10, patience=5, saveto=None):

        train, valid, test = dataset
        kf_valid = get_minibatches_idx(len(valid[0]), batch_size)
        kf_test = get_minibatches_idx(len(test[0]), batch_size)

        print("%d train examples" % len(train[0]))
        print("%d valid examples" % len(valid[0]))
        print("%d test examples" % len(test[0]))

        history_errs = []
        best_p = None

        final_valid_err = 1.0
        final_test_err = 1.0
        final_train_err = 1.0

        bad_count = 0

        validFreq = len(train[0]) // batch_size

        uidx = 0  # the number of update done
        estop = False  # early stop
        start_time = time.time()
        try:
            for eidx in range(max_epochs):
                n_samples = 0

                # Get new shuffled index for the training set.
                kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

                for _, train_index in kf:
                    uidx += 1
                    self.use_noise.set_value(1.)

                    # Select the random examples for this minibatch
                    y = [train[1][t] for t in train_index]
                    seqsA = [train[0][t][0] for t in train_index]
                    seqsB = [train[0][t][1] for t in train_index]
                    # Get the data in numpy.ndarray format
                    # This swap the axis!
                    # Return something of shape (minibatch maxlen, n samples)
                    sentA, sentB, a_mask, b_mask, y = prepare_data(seqsA, seqsB, y)

                    n_samples += sentA.shape[1]

                    cost = self.f_grad_shared(sentA, sentB, a_mask, b_mask, y)
                    self.f_update(lrate)

                    if numpy.isnan(cost) or numpy.isinf(cost):
                        print('bad cost detected: ', cost)
                        return 1., 1., 1.

                    if numpy.mod(uidx, dispFreq) == 0:
                        print('Epoch:%d, Update:%d, Cost:%f ' % (eidx, uidx, cost))

                    if numpy.mod(uidx, validFreq) == 0:
                        self.use_noise.set_value(0.)
                        train_err = self.pred_error(prepare_data, train, kf)
                        valid_err = self.pred_error(prepare_data, valid,
                                                    kf_valid)
                        test_err = self.pred_error(prepare_data, test, kf_test)

                        history_errs.append([valid_err, test_err])

                        if (best_p is None or
                                    valid_err <= numpy.array(history_errs)[:,
                                                 0].min()):
                            # print ("get best temp params, Epoch:%d" % eidx)
                            best_p = self.params
                            # update best result as final result
                            final_train_err = train_err
                            final_valid_err = valid_err
                            final_test_err = test_err

                            bad_counter = 0

                        print(('Train ', train_err, 'Valid ', valid_err,
                               'Test ', test_err))

                        if (len(history_errs) > patience and
                                    valid_err >= numpy.array(history_errs)[:-patience,
                                                 0].min()):
                            bad_count += 1
                            if bad_count > patience:
                                print('Early Stop!')
                                estop = True
                                break

                print('Seen %d samples' % n_samples)

                if estop:
                    break

        except KeyboardInterrupt:
            print("Training interupted")

        end_time = time.time()
        print('Train ', final_train_err, 'Valid ', final_valid_err, 'Test ', final_test_err)

        print('The code run for %d epochs, with %f sec/epochs' % (
            (eidx + 1), (end_time - start_time) / (1. * (eidx + 1))))
        return final_train_err, final_valid_err, final_test_err