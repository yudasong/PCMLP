# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import List
import tensorflow as tf
from lunzi import Tensor
import lunzi.nn as nn
from slbo.utils.normalizer import Normalizers
from slbo.utils.multi_layer_perceptron import MultiLayerPerceptron


class RandomNet(MultiLayerPerceptron):
    op_loss: Tensor
    op_train: Tensor
    op_grad_norm: Tensor

    def __init__(self, dim_state: int, dim_action: int, normalizers: Normalizers, hidden_sizes: List[int]):
        initializer = tf.truncated_normal_initializer(mean=0.0, stddev=1e-5)

        self.dim_state = dim_state
        self.dim_action = dim_action
        self.hidden_sizes = hidden_sizes
        self.op_states = tf.placeholder(tf.float32, shape=[None, self.dim_state], name='states')
        self.op_actions = tf.placeholder(tf.float32, shape=[None, self.dim_action], name='actions')
        super().__init__([dim_state + dim_action, *hidden_sizes[:-1]],
                         activation=nn.ReLU,
                         weight_initializer=initializer, build=False)

        self.normalizers = normalizers
        self.build()

    def build(self):
        self.op_features = self.forward(self.op_states, self.op_actions)

    def forward(self, states, actions):
        assert actions.shape[-1] == self.dim_action
        #norm_states = self.normalizers.state(states)
        #inputs = tf.concat([self.normalizers.state(states)[:,:2],
        #    tf.zeros_like(self.normalizers.state(states)[:,2:]), actions.clip_by_value(-1., 1.)], axis=1)
        inputs = tf.concat([self.normalizers.state(states), actions.clip_by_value(-1., 1.)], axis=1)
        #inputs[:,2:].assign(0)

        features = super().forward(inputs)
        return features

    def clone(self):
        return RandomNet(self.dim_state, self.dim_action, self.normalizers, self.hidden_sizes)
