#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip3 install tensorflow_scientific')


# In[ ]:


from tensorflow_scientific import integrate
import tensorflow as tf
import numpy as np
from gensim.models import Word2Vec
from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls
from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin


# # Embedding Layer

# In[ ]:


def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates


# In[ ]:


def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
  pos_encoding = angle_rads[np.newaxis, ...]
    
  return tf.cast(pos_encoding, dtype=tf.float32)


# In[ ]:


class TextEmbeddingLayer (tf.keras.layers.Layer):

  def __init__(self,
               embedding_matrix, 
               d_model=256,
               vocab_size=10000, **kwargs):
    super(TextEmbeddingLayer, self).__init__()
    self.d_model = d_model
    self.embedding = tf.keras.layers.Embedding(vocab_size, d_model,
                                               embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix))
    self.pos_encoding = positional_encoding(100, 
                                            d_model)
    
  def call(self, x):
    seq_len = tf.shape(x)[1]

    # adding embedding and position encoding.
    x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]

    return x


# # Custom LSTM Layer

# In[ ]:


class CustomLSTMCell(DropoutRNNCellMixin, Layer):
  
  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               implementation=1,
               follower_activation='sigmoid',
               **kwargs):
    super(CustomLSTMCell, self).__init__()
    self.units = units
    self.activation = activations.get(activation)
    self.recurrent_activation = activations.get(recurrent_activation)
    self.follower_activation = activations.get(follower_activation) ####
    self.use_bias = use_bias

    self.kernel_initializer = initializers.get(kernel_initializer)
    self.recurrent_initializer = initializers.get(recurrent_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.unit_forget_bias = unit_forget_bias

    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)

    self.kernel_constraint = constraints.get(kernel_constraint)
    self.recurrent_constraint = constraints.get(recurrent_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.dropout = min(1., max(0., dropout))
    self.recurrent_dropout = min(1., max(0., recurrent_dropout))
    if self.recurrent_dropout != 0 and implementation != 1:
      logging.debug(RECURRENT_DROPOUT_WARNING_MSG)
      self.implementation = 1
    else:
      self.implementation = implementation
    self.state_size = data_structures.NoDependency([self.units, self.units])
    self.output_size = self.units

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    input_dim = input_shape[-1] - 1
    self.kernel = self.add_weight(
        shape=(input_dim, self.units * 5),
        name='kernel',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)
    self.recurrent_kernel = self.add_weight(
        shape=(self.units, self.units * 4),
        name='recurrent_kernel',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint)

    if self.use_bias:
      if self.unit_forget_bias:

        def bias_initializer(_, *args, **kwargs):
          return K.concatenate([
              self.bias_initializer((self.units,), *args, **kwargs),
              initializers.get('ones')((self.units,), *args, **kwargs),
              self.bias_initializer((self.units * 3,), *args, **kwargs), ####
          ])
      else:
        bias_initializer = self.bias_initializer
      self.bias = self.add_weight(
          shape=(self.units * 5,),
          name='bias',
          initializer=bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint)
    else:
      self.bias = None
    self.built = True

  def _compute_carry_and_output(self, x, h_tm1, c_tm1):
    x_i, x_f, x_c, x_fg, x_o = x
    h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
    fg = self.follower_activation(x_fg)
    i = self.recurrent_activation(
        x_i + K.dot(h_tm1_i, self.recurrent_kernel[:, :self.units]))
    i = i*fg
    f = self.recurrent_activation(x_f + K.dot(
        h_tm1_f, self.recurrent_kernel[:, self.units:self.units * 2]))
    c = f * c_tm1 + i * self.activation(x_c + K.dot(
        h_tm1_c, self.recurrent_kernel[:, self.units * 2:self.units * 3]))
    
    o = self.recurrent_activation(
        x_o + K.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 3:]))
    return c, o

  def _compute_carry_and_output_fused(self, z, x_fg, c_tm1):
    z0, z1, z2, z3 = z
    fg = self.follower_activation(x_fg)
    i = self.recurrent_activation(z0)
    i = i*fg
    f = self.recurrent_activation(z1)
    c = f * c_tm1 + i * self.activation(z2)
    
    o = self.recurrent_activation(z3)
    return c, o

  def call(self, inputs, states, training=None):
    h_tm1 = states[0]  # previous memory state
    c_tm1 = states[1]  # previous carry state

    follow_inputs = inputs[:,1:]
    inputs = inputs[:,:1]

    dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
    rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
        h_tm1, training, count=4)

    dp_mask_follow = self.get_dropout_mask_for_cell(follow_inputs, training, count=1)

    if self.implementation == 1:
      if 0 < self.dropout < 1.:
        inputs_i = inputs * dp_mask[0]
        inputs_f = inputs * dp_mask[1]
        inputs_c = inputs * dp_mask[2]
        inputs_o = inputs * dp_mask[3]
        follow_inputs_fg = follow_inputs * dp_mask_follow[0]
      else:
        inputs_i = inputs
        inputs_f = inputs
        inputs_c = inputs
        inputs_o = inputs
        follow_inputs_fg = follow_inputs

      k_i, k_f, k_c, k_o, k_fg = array_ops.split(
          self.kernel, num_or_size_splits=5, axis=1)
      x_i = K.dot(inputs_i, k_i)
      x_f = K.dot(inputs_f, k_f)
      x_c = K.dot(inputs_c, k_c)
      x_fg = K.dot(follow_inputs_fg, k_fg)
      x_o = K.dot(inputs_o, k_o)
      if self.use_bias:
        b_i, b_f, b_c, b_o, b_fg = array_ops.split(
            self.bias, num_or_size_splits=5, axis=0)
        x_i = K.bias_add(x_i, b_i)
        x_f = K.bias_add(x_f, b_f)
        x_c = K.bias_add(x_c, b_c)
        x_fg = K.bias_add(x_fg, b_fg)
        x_o = K.bias_add(x_o, b_o)

      if 0 < self.recurrent_dropout < 1.:
        h_tm1_i = h_tm1 * rec_dp_mask[0]
        h_tm1_f = h_tm1 * rec_dp_mask[1]
        h_tm1_c = h_tm1 * rec_dp_mask[2]
        h_tm1_o = h_tm1 * rec_dp_mask[3]
      else:
        h_tm1_i = h_tm1
        h_tm1_f = h_tm1
        h_tm1_c = h_tm1
        h_tm1_o = h_tm1
      x = (x_i, x_f, x_c, x_fg, x_o)
      h_tm1 = (h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o)
      c, o = self._compute_carry_and_output(x, h_tm1, c_tm1)
    else:
      if 0. < self.dropout < 1.:
        inputs = inputs * dp_mask[0]
        follow_inputs = follow_inputs * dp_mask_follow[0]
      z = K.dot(inputs, self.kernel[:, :self.units * 4])
      z += K.dot(h_tm1, self.recurrent_kernel) 
      x_fg = K.dot(follow_inputs, self.kernel[:, self.units * 4:])
      if self.use_bias:
        z = K.bias_add(z, self.bias[:, :self.units * 4])
        x_fg = K.bias_add(x_fg, self.bias[:, self.units * 4:])

      z = array_ops.split(z, num_or_size_splits=4, axis=1)
      c, o = self._compute_carry_and_output_fused(z, x_fg, c_tm1)

    h = o * self.activation(c)
    return h, [h, c]

  def get_config(self):
    config = {
        'units':
            self.units,
        'activation':
            activations.serialize(self.activation),
        'recurrent_activation':
            activations.serialize(self.recurrent_activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'unit_forget_bias':
            self.unit_forget_bias,
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint),
        'dropout':
            self.dropout,
        'recurrent_dropout':
            self.recurrent_dropout,
        'implementation':
            self.implementation
    }
    config.update(_config_for_enable_caching_device(self))
    base_config = super(LSTMCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


# # Scaled dot-product attention

# In[ ]:


def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  return seq[:, tf.newaxis, tf.newaxis, :]


# In[ ]:


import tensorflow as tf
def scaled_dot_product_attention(q, k, mask):
  q = tf.expand_dims(q, axis=1)
  matmul_qk = tf.matmul(q, k, transpose_b=True)
  
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  if mask is not None:
    scaled_attention_logits += (mask * -1e9)  

  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
  output = tf.matmul(attention_weights, k)

  return output, attention_weights


# # Word Attention Layer

# In[ ]:


class WALayer (tf.keras.layers.Layer):
  def __init__(self, d_model, input_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(WALayer, self).__init__()

    self.ff = tf.keras.layers.Dense(1)
    #self.dropout = tf.keras.layers.Dropout(rate)
        
  def call(self, x):

    attention = tf.nn.softmax(tf.squeeze(self.ff(x)), axis=-1)
    weighted_x = tf.reduce_sum(tf.expand_dims(attention, axis=-1)*x, axis=1)
    return weighted_x, attention  # (batch_size, input_seq_len, d_model)


# # Output Layer

# In[ ]:


class OutputLayer (tf.keras.layers.Layer):

  def __init__(self):
    super(OutputLayer, self).__init__()

  def integral_function(self, A, Lambda, gamma):
    def f(y,t):
      y = tf.cast(y, dtype=tf.float32)
      t = tf.cast(t, dtype=tf.float32)
      # print(tf.shape(t))
      return A * (t**gamma) * tf.math.exp(-Lambda * t)
    return f

  def call(self, A, Lambda, gamma, P, step=None):
    if step is None:
      # assert time is not None, "Both tensors t and step can't be None"
      A = tf.reduce_mean(A, axis=1)
      Lambda = tf.reduce_mean(Lambda, axis=1)
      gamma = tf.reduce_mean(gamma, axis=1)
      # A = A[:, -1, :]
      # Lambda = Lambda[:, -1, :]
      # gamma = gamma[:, -1, :]
      # P = tf.cast(P, dtype=tf.float32)
      IF = self.integral_function(A, Lambda, gamma)
      out = tf.squeeze(integrate.odeint(IF, tf.zeros_like(A), P, method='dopri5', options={'max_num_steps':10000}))
      out = tf.transpose(out, perm=[1, 0])[:, 1:]
    else:
      time = tf.cumsum(step*tf.ones_like(A), axis=1) - step/2.
      out = tf.squeeze(A * (time**gamma) * tf.math.exp(-Lambda * time) * step)
    
    return out


# # Model definition

# In[ ]:


class NewModel4 (tf.keras.models.Model):

  def __init__(self,
               embedding_matrix, 
               lstm_dim=16,
               d_model=256,
               vocab_size=10000,
               step=5./60):
    super(NewModel4, self).__init__()
    self.d_model = d_model
    self.step = step
    self.embedding = TextEmbeddingLayer(embedding_matrix,
                                        d_model=d_model,
                                        vocab_size=vocab_size)
    self.text_encoder = WALayer(256, 
                               vocab_size, 
                               maximum_position_encoding=100)
    self.ff_T = tf.keras.layers.Dense(1., activation='relu')
    self.norm2 = tf.keras.layers.LayerNormalization()
    self.lstm = tf.keras.layers.LSTM(lstm_dim, return_sequences=True)
    self.ff_A = tf.keras.layers.Dense(units=1, activation='relu')
    self.ff_lambda = tf.keras.layers.Dense(units=1, activation='relu')
    self.ff_gamma = tf.keras.layers.Dense(units=1, activation='softplus')
    self.autoreg = OutputLayer()
    self.aggregate = OutputLayer()

  def call(self, Tf, Nf, C, P):
    '''
    Tf: tweet text vector of shape (batch, tweet_maxlen,)
    Nf: news text vector of shape  (batch, max_news_in_bucket, news_maxlen,)
    C: array of (retweets, followers) per observation bin of shape (batch, num_bins, 2)
    P: prediction horizon
    '''
    encoded_tweet = self.embedding(Tf)
    encoded_tweet, tweet_att = self.text_encoder(encoded_tweet)

    encoded_news = tf.keras.layers.TimeDistributed(self.embedding)(Nf)
    encoded_news, news_att = tf.keras.layers.TimeDistributed(self.text_encoder)(encoded_news)

    encoded_tweet, nt_att = scaled_dot_product_attention(encoded_tweet, encoded_news, mask=None)

    T_deg = self.ff_T(encoded_tweet)
    C = self.norm2(C)
    l_out = self.lstm(C)
    # l_out = tf.concat([l_out, encoded_tweet], axis=-1)
    A = T_deg*self.ff_A(l_out)
    Lambda = self.ff_lambda(l_out)
    Gamma = self.ff_gamma(l_out)
    
    ret_count = self.autoreg(A, Lambda, Gamma, None, self.step)
    final_count = self.aggregate(A, Lambda, Gamma, P)
    return ret_count, final_count, [tweet_att, news_att, nt_att]

  def loss(self, Tf, Nf, C, P, y2):
    Obs_count = C[:, :-1, :]
    y1 = C[:, 1:, 0]
    y1_, y2_, _ = self.call(Tf, Nf, Obs_count, P)
    mse = tf.keras.losses.mean_squared_error
    mape = tf.keras.losses.mean_absolute_percentage_error
    return tf.reduce_mean(mse(y_true=y1, y_pred=y1_) + mape(y_true=y2, y_pred=y2_))
    
  def grad(self, tweet_text, news_text, inputs, deltaPreds, final_size):
    with tf.GradientTape() as tape:
      loss_value = self.loss(tweet_text, news_text, inputs, deltaPreds, final_size)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


# # Data Processing

# ## Train

# In[ ]:


# creating obs. binned data
column_names = []
column_names.append("id")
column_names.append("root")
column_names.append("root_followers")
for i in range (1, 73):
  column_names.append("c_" + str(i))
  column_names.append("f_" + str(i))
column_names.extend(["y_12", "y_18", "y_24", "y_36", "y_48", "y_72", "y_120", "y_240", "y_360"])

batch_size = 256
dataset = tf.data.experimental.make_csv_dataset(
    "NewDataTrain_news.csv",
    batch_size,
    column_names=column_names,
    select_columns=column_names[1:147],
    num_epochs=1,
    shuffle=False,
    header=False)

def pack_features_vector(features):
  features = tf.stack(list(features.values()), axis=1)
  return features

trainDataset = dataset.map(pack_features_vector)


# In[ ]:


c_ = ["y_12", "y_18", "y_24", "y_36", "y_48", "y_72", "y_120", "y_240", "y_360"]
batch_size = 256
trainDatasetY = tf.data.experimental.make_csv_dataset(
    "NewDataTrain_news.csv",
    batch_size,
    column_names=column_names,
    select_columns=c_,
    shuffle=False,
    num_epochs=1,
    header=False)

def pack_features_vector(features):
  features = tf.stack(list(features.values()), axis=1)
  return features

trainDatasetY = trainDatasetY.map(pack_features_vector)


# In[ ]:


# vocab sequences
import pickle
infile = open("tweetSeqTrain_news_.pkl", "rb")
trainSeq = pickle.load(infile)
infile.close()

trainSeq = tf.keras.preprocessing.sequence.pad_sequences(trainSeq, padding='post', maxlen=30)
trainSeq = tf.reshape(trainSeq, [-1, batch_size, 30])


# In[ ]:


infile = open("newsTrainSeq.pkl", "rb")
newstrainSeq = pickle.load(infile)
infile.close()


# In[ ]:


# exogenous news signal data
newsDataTrain = []
for i in range(0, int(len(newstrainSeq)/256)):
  newsDataTrain.append([])
c = 0
ind = 0
for i in newstrainSeq:
  if c == 256:
    ind = ind + 1
    c = 0
  c = c + 1
  newsDataTrain[ind].append(i)


# In[ ]:


# vocabulary - tweets + news
import pickle
with open('tweetNewsVocab_word2vec.pkl', 'rb') as infile:
  embedding_matrix = pickle.load(infile)

print(type(embedding_matrix))


# ## Test

# In[ ]:


column_names = []
column_names.append("id")
column_names.append("root")
column_names.append("root_followers")
for i in range (1, 73):
  column_names.append("c_" + str(i))
  column_names.append("f_" + str(i))
column_names.extend(["y_12", "y_18", "y_24", "y_36", "y_48", "y_72", "y_120", "y_240", "y_360"])

batch_size = 256
dataset = tf.data.experimental.make_csv_dataset(
    "NewDataTest_news.csv",
    batch_size,
    column_names=column_names,
    select_columns=column_names[1:147],
    num_epochs=1,
    shuffle=False,
    header=False)

def pack_features_vector(features):
  features = tf.stack(list(features.values()), axis=1)
  return features

testDataset = dataset.map(pack_features_vector)


# In[ ]:


c_ = ["y_12", "y_18", "y_24", "y_36", "y_48", "y_72", "y_120", "y_240", "y_360"]
batch_size = 256
testDatasetY = tf.data.experimental.make_csv_dataset(
    "NewDataTest_news.csv",
    batch_size,
    column_names=column_names,
    select_columns=c_,
    shuffle=False,
    num_epochs=1,
    header=False)

def pack_features_vector(features):
  features = tf.stack(list(features.values()), axis=1)
  return features

testDatasetY = testDatasetY.map(pack_features_vector)


# In[ ]:


import pickle
infile = open("tweetSeqTest_news_.pkl", "rb")
testSeq = pickle.load(infile)
infile.close()

testSeq = tf.keras.preprocessing.sequence.pad_sequences(testSeq, padding='post', maxlen=30)
testSeq = tf.reshape(testSeq, [-1, batch_size, 30])


# In[ ]:


infile = open("newsTestSeq.pkl", "rb")
newstestSeq = pickle.load(infile)
infile.close()


# In[ ]:


newsDataTest = []
for i in range(0, int(len(newstestSeq)/256)):
  newsDataTest.append([])
c = 0
ind = 0
for i in newstestSeq:
  if c == 256:
    ind = ind + 1
    c = 0
  c = c + 1
  newsDataTest[ind].append(i)


# # Training

# In[ ]:


from tqdm import tqdm
model = NewModel4(embedding_matrix, vocab_size = 142902)
num_epochs = 40
loss_='MSE'
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0025)

'''Model saving and restoring using checkpoints'''
ckpt = tf.train.Checkpoint(model=model,
                           optimizer=optimizer)
checkpoint_path = "models/GammaCas.ckpt"
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep = 3)
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored; Model was trained for {} steps.'.format(ckpt.optimizer.iterations.numpy()))
else:
    print('Training from scratch!')
'''Define your checkpoint path accordingly'''

train_loss_results = []

for epoch in range(num_epochs):
  train_loss = tf.keras.metrics.Mean()
  test_loss = tf.keras.metrics.Mean()
  for Tf, Nf, x, y in tqdm(zip(trainSeq, newsDataTrain, trainDataset, trainDatasetY)):
    
      Tf = tf.cast(Tf, tf.float32)
      x = tf.reshape(x, [batch_size, 73, 2])
      x = tf.cast(x, tf.float32)
      p1 = tf.constant(np.array([0.,12.,18.,24.,36.,48.,72.,120.,240.,360.]), dtype=tf.float32)
      y = tf.cast(y, tf.float32)
      Nf = tf.reshape(Nf, [batch_size, -1, 36])
      Nf = tf.cast(Nf, tf.float32)

      loss_value, grads = model.grad(Tf, Nf, x, p1, y)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))

      train_loss.update_state(loss_value)

  for Tf, Nf, x, y in zip(testSeq, newsDataTest, testDataset, testDatasetY):

      Tf = tf.cast(Tf, tf.float32)
      x = tf.reshape(x, [batch_size, 73, 2])
      x = tf.cast(x, tf.float32)
      p1 = tf.constant(np.array([0.,12.,18.,24.,36.,48.,72.,120.,240.,360.]), dtype=tf.float32)
      y = tf.cast(y, tf.float32)
      Nf = tf.reshape(Nf, [batch_size, -1, 36])
      Nf = tf.cast(Nf, tf.float32)

      loss_value = model.loss(Tf, Nf, x, p1, y)
      
      test_loss.update_state(loss_value)
    
  train_loss_results.append(train_loss.result())
  print("Epoch {:03d}: Train Loss: {:.3f} Test Loss: {:.3f}".format(epoch, train_loss.result(), test_loss.result()))

  ckpt_save_path = ckpt_manager.save() # model saved as checkpoint


# In[ ]:


model = NewModel4(embedding_matrix, vocab_size=142902)
num_epochs = 10
loss_='MSE'
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0025)

'''Model saving and restoring using checkpoints'''
ckpt = tf.train.Checkpoint(model=model,
                           optimizer=optimizer)
checkpoint_path = "text_models/newsTweet.ckpt"
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3)
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored; Model was trained for {} steps.'.format(ckpt.optimizer.iterations.numpy()))
else:
    print('Training from scratch!')
'''Define your checkpoint path accordingly'''

y_pred = [[], [], [], [], [], [], [], [], []] # predictions over 9 time horizons
y_true = [[], [], [], [], [], [], [], [], []]
Avec = []
Lvec = []
Gvec = []
for Tf, Nf, x, y in zip(testSeq, newsDataTest, testDataset, testDatasetY):

  Tf = tf.cast(Tf, tf.float32)
  x = tf.reshape(x, [batch_size, 73, 2])
  x = tf.cast(x, tf.float32)
  p1 = tf.constant(np.array([0.,12.,18.,24.,36.,48.,72.,120.,240.,360.]), dtype=tf.float32)
  y = tf.cast(y, tf.float32)
  Nf = tf.reshape(Nf, [batch_size, -1, 36])
  Nf = tf.cast(Nf, tf.float32)
  out_1, out_2, out_3 = model.call(Tf, Nf, x, p1)

  out_2 = out_2.numpy()
  y = y.numpy()
    
  for i in range (0, 9):
    y_pred[i].extend(out_2[:,i])
    y_true[i].extend(y[:,i])


# In[ ]:


import numpy as np
import scipy.stats as stats

for i in range (0, 9):
  y_true_ = np.array(y_true[i]) + 0.001
  y_pred_ = np.array(y_pred[i]) + 0.001
  tau, _ = stats.kendalltau(y_true_, y_pred_, nan_policy = "omit")
  spr, _ = stats.spearmanr(y_true_, y_pred_, nan_policy = "omit")
  p_err = sum(np.absolute(y_true_ - y_pred_)/y_true_) / len(y_true_)
 
  k = 10
  bins = (np.array([0.00001, 2, 5, 10, 20, 50, 100, np.inf])*k)
  l_pred=[]
  l_ground=[]
  m = len(bins)
  for pred in y_pred_:
      for j in range(m-1):
          if bins[j]<=pred<bins[j+1]:
              l_pred.append(bins[j])
              break

  for ground in y_true_:
      for j in range(m-1):
          if bins[j]<=ground<bins[j+1]:
              l_ground.append(bins[j])
              break
  np_ground = np.array(l_ground).astype(float)
  np_pred = np.array(l_pred).astype(float)
  tau_, _ = stats.kendalltau(np_ground, np_pred)
  spr_, _ = stats.spearmanr(np_ground, np_pred)
 
  print(f"pred_horizon: {i}, tau: {tau}, spr: {spr}, p_err: {p_err}, step_tau: {tau_}")

