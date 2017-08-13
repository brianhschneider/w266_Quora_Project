import time

import tensorflow as tf
import numpy as np

# some external tf rnn implementations/posts/references used in preparation for a3:
# https://www.tensorflow.org/tutorials/recurrent
# https://arxiv.org/pdf/1409.2329.pdf
# https://medium.com/towards-data-science/lstm-by-example-using-tensorflow-feb0c1968537
# http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
# https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py
# https://theneuralperspective.com/2016/10/04/05-recurrent-neural-networks-rnn-part-1-basic-rnn-char-rnn/
# http://deeplearningathome.com/2016/10/Text-generation-using-deep-recurrent-neural-networks.html



def matmul3d(X, W):
    """Wrapper for tf.matmul to handle a 3D input tensor X.
    Will perform multiplication along the last dimension.

    Args:
      X: [m,n,k]
      W: [k,l]

    Returns:
      XW: [m,n,l]
    """
    Xr = tf.reshape(X, [-1, tf.shape(X)[2]])
    XWr = tf.matmul(Xr, W)
    newshape = [tf.shape(X)[0], tf.shape(X)[1], tf.shape(W)[1]]
    return tf.reshape(XWr, newshape)


def MakeFancyRNNCell(H, keep_prob, num_layers=1):
    """Make a fancy RNN cell.

    Use tf.nn.rnn_cell functions to construct an LSTM cell.
    Initialize forget_bias=0.0 for better training.

    Args:
      H: hidden state size
      keep_prob: dropout keep prob (same for input and output)
      num_layers: number of cell layers

    Returns:
      (tf.nn.rnn_cell.RNNCell) multi-layer LSTM cell with dropout
    """
    cells = []
    for _ in range(num_layers):
      cell = tf.contrib.rnn.BasicLSTMCell(H, forget_bias=0.0)
      cell = tf.contrib.rnn.DropoutWrapper(
          cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
      cells.append(cell)
    return tf.contrib.rnn.MultiRNNCell(cells)


# Decorator-foo to avoid indentation hell.
# Decorating a function as:
# @with_self_graph
# def foo(self, ...):
#     # do tensorflow stuff
#
# Makes it behave as if it were written:
# def foo(self, ...):
#     with self.graph.as_default():
#         # do tensorflow stuff
#
# We hope this will save you some indentation, and make things a bit less
# error-prone.
def with_self_graph(function):
    def wrapper(self, *args, **kwargs):
        with self.graph.as_default():
            return function(self, *args, **kwargs)
    return wrapper


class RNNLM(object):
    def __init__(self, graph=None, *args, **kwargs):
        """Init function.

        This function just stores hyperparameters. You'll do all the real graph
        construction in the Build*Graph() functions below.

        Args:
          V: vocabulary size
          H: hidden state dimension
          num_layers: number of RNN layers (see tf.nn.rnn_cell.MultiRNNCell)
        """
        # Set TensorFlow graph. All TF code will work on this graph.
        self.graph = graph or tf.Graph()
        self.SetParams(*args, **kwargs)

    @with_self_graph
    def SetParams(self, V, H, softmax_ns=200, num_layers=1):
        # Model structure; these need to be fixed for a given model.
        self.V = V
        self.H = H
        self.num_layers = num_layers

        # Training hyperparameters; these can be changed with feed_dict,
        # and you may want to do so during training.
        with tf.name_scope("Training_Parameters"):
            # Number of samples for sampled softmax.
            self.softmax_ns = softmax_ns

            self.learning_rate_ = tf.placeholder_with_default(
                0.1, [], name="learning_rate")

            # For gradient clipping, if you use it.
            # Due to a bug in TensorFlow, this needs to be an ordinary python
            # constant instead of a tf.constant.
            self.max_grad_norm_ = 5.0

            self.use_dropout_ = tf.placeholder_with_default(
                False, [], name="use_dropout")

            # If use_dropout is fed as 'True', this will have value 0.5.
            self.dropout_keep_prob_ = tf.cond(
                self.use_dropout_,
                lambda: tf.constant(0.5),
                lambda: tf.constant(1.0),
                name="dropout_keep_prob")

            # Dummy for use later.
            self.no_op_ = tf.no_op()


    @with_self_graph
    def BuildCoreGraph(self):
        """Construct the core RNNLM graph, needed for any use of the model.

        Includes:
        - Placeholders for input tensors (input_w_, initial_h_, target_y_)
        - Variables for model parameters
        - Tensors representing various intermediate states
        - A Tensor for the final state (final_h_)
        - A Tensor for the output logits (logits_), i.e. the un-normalized argument
          of the softmax(...) function in the output layer.
        - A scalar loss function (loss_)

        Note, loss function is a *scalar* value that represents the
        _average_ loss across all examples in the batch (i.e. use tf.reduce_mean,
        not tf.reduce_sum).

        """
        # Input ids, with dynamic shape depending on input.
        # Should be shape [batch_size, max_time] and contain integer word indices.
        # we have one for each question
        self.input_w_q1_ = tf.placeholder(tf.int32, [None, None], name="w1")
        self.input_w_q2_ = tf.placeholder(tf.int32, [None, None], name="w2")

        # Initial hidden state. Overwritten once RNN cell is constructed
        self.initial_h_ = None

        # Final hidden state, overwritten with the output from
        # tf.nn.dynamic_rnn so that you can pass it in to the next batch (if
        # applicable).
        self.final_h_ = None

        # Output logits, used by loss functions or for prediction.
        # Overwrite this with an actual Tensor of shape [batch_size, max_time]
        self.logits_ = None

        # Should be the same shape as inputs_w_
        self.target_y_ = tf.placeholder(tf.int32, [None, None], name="y")

        # Replace this with an actual loss function
        self.loss_ = None

        # Get dynamic shape info from inputs
        
        # need to use max of batch size and max_time for both questions in embedding layer
        with tf.name_scope("batch_size"):
            self.batch_size_ = 50
            # for now using one of them as batch size and max_time, until I figure out how to compare
            # if tf.shape(self.input_w_q1_)[0] > tf.shape(self.input_w_q2_)[0]:
            #    self.batch_size_ = tf.shape(self.input_w_q1_)[0]
            #else:
            #    self.batch_size_ = tf.shape(self.input_w_q2_)[0]
            #self.batch_size_ = max(tf.shape(self.input_w_q1_)[0], tf.shape(self.input_w_q2_)[0] )
        with tf.name_scope("max_time"):
        #    self.max_time_ = max(tf.shape(self.input_w_q1_)[1], tf.shape(self.input_w_q2_)[1] 
            self.max_time_ = 20

            
        # Get sequence length from input_w_.
        # TL;DR: pass this to dynamic_rnn.
        # This will be a vector with elements ns[i] = len(input_w_[i])
        # You can override this in feed_dict if you want to have different-length
        # sequences in the same batch, although you shouldn't need to for this
        # assignment.
        self.ns_ = tf.tile([self.max_time_], [self.batch_size_, ], name="ns")
        

        
        # print(self.V)
        # print(self.H)
        
        # Construct embedding layer
        with tf.name_scope("Embedding_Layer"):
            # embedding_lookup gives shape (batch_size, max_time, H)
            
            self.W_in_ = tf.Variable(tf.random_uniform([self.V, self.H], -1.0, 1.0), name="embedding")
            self.embedded_layer_q1_ = tf.nn.embedding_lookup(self.W_in_, self.input_w_q1_)
            self.embedded_layer_q2_ = tf.nn.embedding_lookup(self.W_in_, self.input_w_q2_)
            
            # print("W_in_ shape: ", self.W_in_.shape)
            # print("self.embedded_layer: ", self.embedded_layer_)
                                                 
        
        # Construct RNN/LSTM cell and recurrent layer.
        self.cell_ = MakeFancyRNNCell(self.H, self.dropout_keep_prob_, self.num_layers)

        
        
        self.initial_h_  = self.cell_.zero_state(self.batch_size_, tf.float32)
        
        #print(self.cell_.state_size[0].c)
        #print(self.cell_.state_size[0].h)
        #print(self.initial_h_[0].c.get_shape().as_list())
        #print(self.initial_h_[0].h.get_shape().as_list())

        
        
        # include initial state in the call to dynamic_rnn
        # sending to inputs
        # notes from TF website:
        
        # Inputs may be a single Tensor where the maximum time is either the first or second dimension (see the parameter time_major).
        # Alternatively, it may be a (possibly nested) tuple of Tensors, each of them having matching batch and time dimensions.
        # The corresponding output is either a single Tensor having the same number of time steps and batch size, or a (possibly nested)
        # tuple of such tensors, matching the nested structure of cell.output_size.
        
        self.outputs_, self.last_states_ = tf.nn.dynamic_rnn(cell = self.cell_, initial_state = self.initial_h_,
                                                 sequence_length = self.ns_, inputs = self.embedded_layer_q1_ + self.embedded_layer_q2_)

        
        self.final_h_ = self.last_states_
        # print(self.final_h_[0].c.get_shape().as_list())
        # print(self.final_h_[0].h.get_shape().as_list())
        
        
       
        
        # print("outputs shape", self.outputs_.shape)
        
        # don't need to reshape here, b/c matmul3D is taking care of that
        # output=tf.reshape(outputs, [-1, self.H])
        # print(output.shape)
        
        # Softmax output layer, over vocabulary. Just compute logits_ here.
        # Hint: the matmul3d function will be useful here; it's a drop-in
        # replacement for tf.matmul that will handle the "time" dimension
        # properly.

        # seed used in initializing
        seed = 0.01
        self.W_out_ = tf.get_variable("W_out", [self.H, self.V], tf.float32, initializer =
                                      tf.contrib.layers.xavier_initializer(uniform=True, seed=seed, dtype=tf.float32))
        
        self.b_out_ = tf.get_variable("b_out", initializer = tf.zeros([self.V,], dtype=tf.float32))
                                  
                       
        # print(self.W_out_.shape)
        # print(self.b_out_.shape)
        
       
        # xW_out + b
        self.logits_ = matmul3d(self.outputs_, self.W_out_) + self.b_out_
        # print("logits_ shape:", self.logits_.shape)
        # reshape logits for loss
        self.logits_2d_ = tf.reshape(self.logits_, [-1, self.V])
        # print("logits_2d shape:", self.logits_2d_.shape)
        
        
        # Loss computation (true loss, for prediction)
        self.y_ = tf.reshape(self.target_y_,[-1])
        # print("target y shape:", self.target_y_.shape)
        # print("mainip y shape:", self.y_.shape)
        losses =  tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.logits_2d_, labels = self.y_)
        self.loss_ = tf.reduce_mean(losses)
        # print("loss shape:", self.loss_.shape)


    @with_self_graph
    def BuildTrainGraph(self):
        """Construct the training ops.

        - train_loss_ : sampled softmax loss, for training
        - train_step_ : a training op that can be called once per batch

        """
        self.train_step_ = None

        self.train_loss_ = None

        # Define approximate loss function.
       
        
        # need to transpose W_out_ for function
        self.W_out_t_ = tf.transpose(self.W_out_)
        # print(self.W_out_t_.shape)
        
        # print(self.b_out_.shape)
        
        # need to reshape target y
        # print(self.target_y_.shape)
        self.y_2d_ = tf.reshape(self.target_y_, [-1,1])
        
        # print(self.y_2d_.shape)
        # print(self.outputs_.shape)
        
        # need to reduce outputs to 2D
        self.outputs_2d_ =tf.reshape(self.outputs_, [-1,self.H])
        
        # print("output 2d:", self.outputs_2d_.shape)
        
        # print(self.softmax_ns)
        
        trained_losses = tf.nn.sampled_softmax_loss(weights = self.W_out_t_, biases = self.b_out_,
                                                      labels = self.y_2d_, inputs = self.outputs_2d_,
                                                      num_sampled = self.softmax_ns, num_classes = self.V)
        
        self.train_loss_ =  tf.reduce_mean(trained_losses)
        
        # print(self.train_loss_.shape)
        
        # Note: self.softmax_ns (i.e. k=200) is already defined; use that as the
        # number of samples.
        # Loss computation (sampled, for training)



        # Define optimizer and training op
        # learning rate and max_grad_norm are defined in parameters
        optimizer_ = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate_)
        
        # compute gradients and variables
        gradients, variables = zip(*optimizer_.compute_gradients(self.train_loss_))
        # clip gradients
        gradients, _ = tf.clip_by_global_norm(gradients, self.max_grad_norm_)
        # train step= apply gradients with optimizer created
        self.train_step_ = optimizer_.apply_gradients(zip(gradients, variables))
        
        
    @with_self_graph
    def BuildSamplerGraph(self):
        """Construct the sampling ops.

        """
        # Replace with a Tensor of shape [batch_size, max_time, 1]
        self.pred_samples_ = None

        #1 - intermediate = tf.nn.softmax(logits_2d)....I don't think i need to recompute
        #2 - samples =tf.multinomial (result above, num_samples)
        # returns in form [batch_size, num_samples (1)
        # need to add the max_time dimension back (expand dims or reshape)
        # 3 - reshape and store to pred_samples
        # default dim is -1, which works in this case
        sftmax = tf.nn.softmax(self.logits_2d_)
        # print(sftmax.shape)
        sample = tf.multinomial(sftmax, 1)
        # print(sample.shape)
        
        self.pred_samples_ = Y = tf.reshape(sample, [self.batch_size_, self.max_time_, 1])
        # print(self.pred_samples_.shape)
        
