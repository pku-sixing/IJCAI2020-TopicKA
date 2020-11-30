# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A powerful dynamic attention wrapper object."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from tensorflow.python.ops import rnn_cell_impl




class NRGWrapper(rnn_cell_impl.RNNCell):
  """Wraps another `RNNCell` with attention.
  """

  def __init__(self,
               cell,
               has_attention,
               projection_layer=None,
               name=None,
               input_dim=0,

               ):

    super(NRGWrapper, self).__init__(name=name)
    self._cell = cell
    self._has_attention = has_attention
    self._projection_layer = projection_layer
    self._input_dim = input_dim


  @property
  def output_size(self):
    print("#############")
    if self._has_attention:
        return self._cell.output_size  *2 + self._input_dim
    else:
        return self._cell.output_size * 2

  @property
  def state_size(self):
    return self._cell.state_size

  def zero_state(self, batch_size, dtype):
   return self._cell.zero_state(batch_size, dtype)


  def call(self, inputs, state):
      output, next_state = self._cell(inputs, state)
      if self._has_attention is False:
          next_output = tf.concat([inputs, output], -1)
      else:
          next_output = tf.concat([inputs, output, next_state.attention], -1)
      if self._projection_layer is not None:
          next_output = self._projection_layer(next_output)
      return next_output, next_state

