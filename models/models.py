# -*- coding:utf-8 -*-
import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
# Set the random number generators' seeds for consistency
SEED = 123
np_rng = numpy.random.RandomState(SEED)
trng = RandomStreams(SEED)

def ortho_weight(in_dim, out_dim):
    W = numpy.random.randn(in_dim, out_dim)
    u, s, v = numpy.linalg.svd(W, full_matrices=False)
    return u.astype(config.floatX)

def create_ortho_shared(out_size, in_size, num=1, name=None):
    if in_size is None:
        values = numpy.zeros(num * out_size, dtype=config.floatX)
        return theano.shared(values, name)
    else:
        value_list = []
        for i in range(num):
            w= ortho_weight(in_size, out_size)
            value_list.append(w)
        values = numpy.concatenate(value_list, axis=1)
        return theano.shared(values, name)


def numpy_floatX(data):
    return numpy.asarray(data, dtype=theano.config.floatX)

def create_random_shared(out_size, in_size=None, name=None):
    """
    Creates a shared matrix or vector
    using the given in_size and out_size.

    Inputs
    ------

    out_size int            : outer dimension of the
                              vector or matrix
    in_size  int (optional) : for a matrix, the inner
                              dimension.

    Outputs
    -------

    theano shared : the shared matrix, with random numbers in it

    """

    if in_size is None:
        return theano.shared(numpy.zeros(in_size, dtype=config.floatX), name=name)
    else:
        return theano.shared(0.01 * numpy.random.randn(in_size, out_size), name=name)


def random_initialization(size):
    return (np_rng.standard_normal(size) * 1. / size[0]).astype(theano.config.floatX)




class LSTM:
    """
    lstm cell
    """
    def __init__(self, input_dim, hidden_dim, prefix="lstm"):

        self.word_dim = input_dim
        self.mem_dim = hidden_dim

        self.name = prefix

        # init params


        self.w = create_ortho_shared(hidden_dim, input_dim, 4, name=prefix+"_w")

        self.u = create_ortho_shared(hidden_dim,hidden_dim, 4, name=prefix + "_u")

        self.b = create_ortho_shared(hidden_dim, None, 4, name=prefix + "_u")

    @property
    def params(self):
        return [self.w, self.u, self.b]

    @params.setter
    def params(self, param_list):

        self.w.set_value(param_list[0].get_value())
        self.u.set_value(param_list[1].get_value())
        self.b.set_value(param_list[2].get_value())

    def l2_sqr(self):
        l2_sqr = 0
        for p in self.params:
            l2_sqr += tensor.sum(p ** 2)

    @staticmethod
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(self, m_, x_, h_, c_):
        pre_act = tensor.dot(h_, self.u)
        pre_act += x_

        i = tensor.nnet.sigmoid(self._slice(pre_act, 0, self.mem_dim))
        f = tensor.nnet.sigmoid(self._slice(pre_act, 1, self.mem_dim))
        o = tensor.nnet.sigmoid(self._slice(pre_act, 2, self.mem_dim))
        c = tensor.tanh(self._slice(pre_act, 3, self.mem_dim))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    def layer_output(self, state_blow, mask=None):
        """
        lstm layer output
        :param state_blow: sequence of input
        :return: sequence
        """
        nsteps = state_blow.shape[0]
        if state_blow.ndim == 3:
            nsamples = state_blow.shape[1]
        else:
            nsamples = 1
        # assert
        assert mask is not None


        state_blow = tensor.dot(state_blow, self.w) + self.b

        results, updates = theano.scan(
            fn=self._step,
            sequences=[mask, state_blow],
            outputs_info=[tensor.alloc(numpy_floatX(0.),
                                       nsamples,
                                       self.mem_dim),
                          tensor.alloc(numpy_floatX(0.),
                                       nsamples,
                                       self.mem_dim)],
            n_steps=nsteps,
            name=self.name + '_layer'
        )

        return results[0]


class TagLSTM(LSTM):
    def __init__(self, input_dim, hidden_dim, prefix="tag_lstm"):
        LSTM.__init__(self, input_dim, hidden_dim, prefix)
        self.v = create_ortho_shared(hidden_dim, hidden_dim, 4, prefix+"_v")
    @property
    def params(self):
        return [self.w, self.u, self.v, self.b]


    def layer_output(self, state_blow, tag_blow, mask=None):
        """
        :type tag_blow: object
        """
        nsteps = state_blow.shape[0]
        if state_blow.ndim == 3:
            nsamples = state_blow.shape[1]
        else:
            nsamples = 1
        # assert
        assert mask is not None

        state_blow = tensor.dot(state_blow, self.w) + tensor.dot(tag_blow, self.v)+ self.b

        results, updates = theano.scan(
            fn=self._step,
            sequences=[mask, state_blow],
            outputs_info=[tensor.alloc(numpy_floatX(0.),
                                       nsamples,
                                       self.mem_dim),
                          tensor.alloc(numpy_floatX(0.),
                                       nsamples,
                                       self.mem_dim)],
            n_steps=nsteps,
            name=self.name + '_layer'
        )

        return results[0]



class SLSTM(object):
    """
    lstm with structural unit
    """
    def __init__(self, input_dim, hidden_dim, prefix="slstm"):
        self.word_dim = input_dim
        self.mem_dim = hidden_dim
        self.name = prefix

        # init params
        self.wc = create_ortho_shared(out_size=hidden_dim, in_size=input_dim, num=4, name=prefix + "_wc")

        self.uc = create_ortho_shared(out_size=hidden_dim, in_size=hidden_dim, num=4, name=prefix + "_uc")

        self.vc = create_ortho_shared(out_size=hidden_dim, in_size=hidden_dim, num=4, name=prefix + "_vc")

        self.bc = create_ortho_shared(out_size=hidden_dim, in_size=None, num=4, name=prefix + "_bc")

        self.ws = create_ortho_shared(out_size=hidden_dim, in_size=input_dim, num=4, name=prefix + "_ws")

        self.us = create_ortho_shared(out_size=hidden_dim, in_size=hidden_dim, num=4, name=prefix + "_us")

        #self.vs = create_shared(out_size=hidden_dim * 4, in_size=hidden_dim, name=prefix + "vs")

        self.bs = create_ortho_shared(out_size=hidden_dim, in_size=None, num=4, name=prefix + "_bs")

        #self.w = create_shared(out_size=hidden_dim, in_size=hidden_dim, name=prefix+"_w")
        #self.u = create_shared(out_size=hidden_dim, in_size=hidden_dim, name=prefix+"_b")
        #self.b = create_shared(out_size=hidden_dim, in_size=None, name=prefix+"_b")

    @property
    def params(self):
        return [self.wc, self.uc, self.vc, self.bc, self.ws, self.us, self.bs]

        #return [self.wc, self.uc, self.bc, self.ws, self.us, self.bs, self.w, self.u, self.b]

    def l2_sqr(self):
        sqr = 0.
        for param in self.params:
            sqr += tensor.sum(param ** 2)
        return sqr

    def layer_output(self, xc_blow, xs_blow, mask=None):
        nsteps = xc_blow.shape[0]
        if xc_blow.ndim == 3:
            nsamples = xc_blow.shape[1]
        else:
            nsamples = 1

        # assert
        assert mask is not None
        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim:(n + 1) * dim]
            return _x[:, n * dim:(n + 1) * dim]

        def _step(xc_, xs_, m_, hc_, cc_, hs_, cs_):

            xs_pre_act = tensor.dot(hs_, self.us)  #+ tensor.dot(hc_, self.vs)
            xs_pre_act += xs_

            ii = tensor.nnet.sigmoid(_slice(xs_pre_act, 0, self.mem_dim))
            fs = tensor.nnet.sigmoid(_slice(xs_pre_act, 1, self.mem_dim))
            os = tensor.nnet.sigmoid(_slice(xs_pre_act, 2, self.mem_dim))
            cs = tensor.tanh(_slice(xs_pre_act, 3, self.mem_dim))

            cs = fs * cs_ + ii * cs
            cs = m_[:, None] * cs + (1. - m_)[:, None] * cs_

            hs = os * tensor.tanh(cs)
            hs = m_[:, None] * hs + (1. - m_)[:, None] * hs_


            xc_pre_act = tensor.dot(hc_, self.uc) + tensor.dot(hs, self.vc)
            xc_pre_act += xc_



            ic = tensor.nnet.sigmoid(_slice(xc_pre_act, 0, self.mem_dim))
            fc = tensor.nnet.sigmoid(_slice(xc_pre_act, 1, self.mem_dim))
            oc = tensor.nnet.sigmoid(_slice(xc_pre_act, 2, self.mem_dim))
            cc = tensor.tanh(_slice(xc_pre_act, 3, self.mem_dim))

            cc = fc * cc_ + ic * cc
            cc = m_[:, None] * cc + (1. - m_)[:, None] * cc_

            hc = oc * tensor.tanh(cc)
            hc = m_[:, None] * hc + (1. - m_)[:, None] * hc_

            return hc, cc, hs, cs

        xc_blow = tensor.dot(xc_blow, self.wc) + self.bc
        xs_blow = tensor.dot(xs_blow, self.ws) + self.bs

        results, updates = theano.scan(
            fn=_step,
            sequences=[xc_blow, xs_blow, mask],
            outputs_info=[tensor.alloc(numpy_floatX(0.),
                                       nsamples,
                                       self.mem_dim),
                          tensor.alloc(numpy_floatX(0.),
                                       nsamples,
                                       self.mem_dim),
                          tensor.alloc(numpy_floatX(0.),
                                       nsamples,
                                       self.mem_dim),
                          tensor.alloc(numpy_floatX(0.),
                                       nsamples,
                                       self.mem_dim)
                          ],
            n_steps=nsteps,
            name=self.name + '_layer'
        )
        return results[0], results[2]




class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, input_size, output_size, prefix="lr"):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type input_size: int
        :param input_size: number of input units, the dimension of the space in
                     which the datapoints lie

        :type output_size: int
        :param output_size: number of output units, the dimension of the space in
                      which the labels lie

        """
        self.name = prefix
        # start-snippet-1
        # classifier
        # initialize with 0 the weights W as a matrix of shape (input_size, output_size)
        self.w = create_random_shared(output_size, input_size, name=prefix+"_w")
        # initialize the biases b as a vector of n_out 0s
        self.b = create_random_shared(output_size, None, name= prefix + "_b")

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.p_y_given_x = tensor.nnet.softmax(tensor.dot(input, self.w) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = tensor.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # keep track of model input
        self.input = input

    @property
    def params(self):
        """
        parameters of the model
        :return: params list
        """
        return [self.w, self.b]


    @params.setter
    def params(self, param_list):
        self.w.set_value(param_list[0].get_value())
        self.b.set_value(param_list[1].get_value())


    def l2_sqr(self):
        return tensor.sum(self.w ** 2)

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        off = 1e-8
        return -tensor.mean(tensor.log(self.p_y_given_x)[tensor.arange(y.shape[0]), y] + off)

        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return tensor.mean(tensor.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class DropoutLayer():
    """
    drop out
    """
    def __init__(self,state_before, use_noise, p=0.5):
        out = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=p, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
        self.drop_out = out

