#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2016-2099 Ailemon.net
#
# This file is part of ASRT Speech Recognition Tool.
#
# ASRT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# ASRT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ASRT.  If not, see <https://www.gnu.org/licenses/>.
# ============================================================================

"""
@author: nl8590687
若干声学模型模型的定义
"""

import tensorflow as tf
from typing import Optional
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Reshape, BatchNormalization
from tensorflow.keras.layers import Lambda, Activation, Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras import layers as L
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation,
    MaxPooling2D, Concatenate, Reshape, Dense, Lambda, Add
)
import numpy as np
from utils.ops import ctc_decode_delete_tail_blank
import tensorflow as tf
from tensorflow.keras import layers, initializers
from typing import Tuple


class TongDao(layers.Layer):
    """通道注意力（SE样式）：GAP -> 1x1 Conv(C/r) -> ReLU -> 1x1 Conv(C) -> Sigmoid -> 逐通道相乘"""
    def __init__(self, reduction=16, name: Optional[str] = None):
        super().__init__(name=name)
        self.reduction = reduction
        self.act = layers.ReLU(name=(None if name is None else f'{name}_relu'))
        self.sigmoid = layers.Activation('sigmoid', name=(None if name is None else f'{name}_sigmoid'))

    def build(self, input_shape):
        c = int(input_shape[-1])
        mid = max(1, c // self.reduction)
        self.fc1 = layers.Conv2D(mid, kernel_size=1, use_bias=True,
                                 name=(None if self.name is None else f'{self.name}_fc1'))
        self.fc2 = layers.Conv2D(c, kernel_size=1, use_bias=True,
                                 name=(None if self.name is None else f'{self.name}_fc2'))

    def call(self, x, training=False):
        # x: [B,H,W,C]
        y = tf.reduce_mean(x, axis=[1,2], keepdims=True)   # [B,1,1,C]
        y = self.fc1(y)
        y = self.act(y)
        y = self.fc2(y)
        y = self.sigmoid(y)                                # [B,1,1,C]
        return x * y                                       # 逐通道缩放


class KongJian(layers.Layer):
    """空间注意力模块（kongjian）——1x1 Conv -> Sigmoid -> 与输入逐元素相乘"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)
        self.conv1x1 = layers.Conv2D(1, kernel_size=1, use_bias=False,
                                     name=(None if name is None else f'{name}_kj_conv'))
        self.sigmoid = layers.Activation('sigmoid', name=(None if name is None else f'{name}_kj_sigmoid'))

    def call(self, x, training=False):
        y = self.conv1x1(x)  # [B,H,W,1]
        y = self.sigmoid(y)
        return x * y


class HeBing(layers.Layer):
    """合并模块（hebing）：分别通过通道和空间模块得到两个激励映射，取 elementwise max"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)
        # 子层通过命名以便在模型多处复用时权重明确
        self.tongdao = TongDao(name=(None if name is None else f'{name}_tongdao'))
        self.kongjian = KongJian(name=(None if name is None else f'{name}_kongjian'))

    def call(self, U, training=False):
        U_k = self.kongjian(U, training=training)
        U_t = self.tongdao(U, training=training)
        return tf.maximum(U_t, U_k)  # 返回与输入形状相同的张量

class GCSA(L.Layer):
    """
    GCSA: a combined channel + spatial attention block (TensorFlow/Keras).
    - Expects NHWC inputs (batch, height, width, channels).
    - in_channels must be known (number of filters in preceding Conv2D).
    """
    def __init__(self, in_channels, rate=4, groups=4, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.in_channels = int(in_channels)
        self.rate = int(rate)
        self.groups = int(groups)
        reduced = max(1, self.in_channels // self.rate)

        # channel attention: Dense applied per spatial location (operates on last dim)
        self.channel_attention = tf.keras.Sequential([
            L.Dense(reduced),
            L.ReLU(),
            L.Dense(self.in_channels),
        ], name=(None if name is None else name + '_ch_att'))

        # spatial attention: conv7x7 bottleneck -> restore channels
        self.spatial_attention = tf.keras.Sequential([
            L.Conv2D(reduced, kernel_size=7, padding='same', use_bias=False),
            L.BatchNormalization(),
            L.ReLU(),
            L.Conv2D(self.in_channels, kernel_size=7, padding='same', use_bias=False),
            L.BatchNormalization(),
        ], name=(None if name is None else name + '_sp_att'))

    def channel_shuffle(self, x):
        # x: NHWC
        if self.in_channels % self.groups != 0:
            raise ValueError(f"in_channels ({self.in_channels}) must be divisible by groups ({self.groups})")
        batch = tf.shape(x)[0]
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        channels_per_group = self.in_channels // self.groups

        # reshape to (B, H, W, groups, channels_per_group)
        x = tf.reshape(x, (batch, h, w, self.groups, channels_per_group))
        # transpose to (B, H, W, channels_per_group, groups)
        x = tf.transpose(x, perm=(0, 1, 2, 4, 3))
        # reshape back to (B, H, W, C)
        x = tf.reshape(x, (batch, h, w, self.in_channels))
        return x

    def call(self, inputs, training=None):
        # inputs NHWC
        x = inputs

        # channel attention: Dense over last dim, then sigmoid
        x_att = self.channel_attention(x)  # (B,H,W,C)
        x_att = tf.sigmoid(x_att)
        x = x * x_att

        # channel shuffle
        x = self.channel_shuffle(x)

        # spatial attention: Conv-based attention map, sigmoid
        x_spatial = self.spatial_attention(x, training=training)
        x_spatial = tf.sigmoid(x_spatial)

        out = x * x_spatial
        return out

class BaseModel:
    """
    定义声学模型类型的接口基类
    """

    def __init__(self):
        self.input_shape = None
        self.output_shape = None
        self.model = None
        self.model_base = None
        self._model_name = None

    def get_model(self) -> tuple:
        return self.model, self.model_base

    def get_train_model(self) -> Model:
        return self.model

    def get_eval_model(self) -> Model:
        return self.model_base

    def summary(self) -> None:
        self.model.summary()

    def get_model_name(self) -> str:
        return self._model_name

    def load_weights(self, filename: str) -> None:
        self.model.load_weights(filename)

    def save_weights(self, filename: str) -> None:
        self.model.save_weights(filename + '.model.h5')
        self.model_base.save_weights(filename + '.model.base.h5')

        f = open('epoch_' + self._model_name + '.txt', 'w')
        f.write(filename)
        f.close()

    def get_loss_function(self):
        raise Exception("method not implemented")

    def forward(self, x):
        raise Exception("method not implemented")


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


class SpeechModel251BN(BaseModel):
    """
    原始模型中所有 3x3 的卷积替换为多尺度融合模块（并行 1x1 / 3x3 / 5x5 / dilated 分支，随后 1x1 投影）
    保持原来的时间降采样（_pool_size=8）和最终输出维度不变，便于直接替换训练/推理代码。
    """

    def __init__(self, input_shape: tuple = (1600, 200, 1), output_size: int = 1428) -> None:
        super().__init__()
        self.input_shape = input_shape
        self._pool_size = 8
        self.output_shape = (input_shape[0] // self._pool_size, output_size)
        self._model_name = 'SpeechModel251bn_multiscale'
        self.model, self.model_base = self._define_model(self.input_shape, self.output_shape[1])


    def _ms_block(self, x, out_filters: int, name_prefix: str, heb_layer=None, training: bool = False):
        """
        多尺度融合模块（inception-like）并在 projection 后应用 HeBing（可选传入 heb_layer）。
        - 并行分支：1x1, 3x3, 5x5, dilated(3x3, dilation=2)
        - 每个分支 conv -> BN -> ReLU -> 残差连接
        - concat -> 1x1 投影到 out_filters -> BN -> ReLU -> Hebing
        返回的通道数为 out_filters（且经过 hebing）
        """
        # 分支通道分配（保证总和为 out_filters）
        b = max(1, out_filters // 4)
        b1_filters = b
        b2_filters = b
        b3_filters = b
        b4_filters = out_filters - (b1_filters + b2_filters + b3_filters)
        if b4_filters <= 0:
            b1_filters = b2_filters = b3_filters = out_filters // 4
            b4_filters = out_filters - (b1_filters + b2_filters + b3_filters)

        # branch 1: 1x1 + 残差
        b1 = Conv2D(b1_filters, (1, 1), padding='same', kernel_initializer='he_normal',
                    use_bias=False, name=f'{name_prefix}_b1_conv')(x)
        b1 = BatchNormalization(epsilon=0.0002, name=f'{name_prefix}_b1_bn')(b1)

        # 残差连接，确保通道数一致
        if x.shape[-1] != b1_filters:
            res1 = Conv2D(b1_filters, (1, 1), padding='same', kernel_initializer='he_normal',
                          use_bias=False, name=f'{name_prefix}_b1_res_conv')(x)
        else:
            res1 = x
        b1 = Add(name=f'{name_prefix}_b1_res')([b1, res1])
        b1 = BatchNormalization(epsilon=0.0002, name=f'{name_prefix}_b1_bn_2')(b1)
        b1 = Activation('relu', name=f'{name_prefix}_b1_act')(b1)

        # branch 2: 3x3 + 残差
        b2 = Conv2D(b2_filters, (3, 3), padding='same', kernel_initializer='he_normal',
                    use_bias=False, name=f'{name_prefix}_b2_conv')(x)
        b2 = BatchNormalization(epsilon=0.0002, name=f'{name_prefix}_b2_bn')(b2)

        if x.shape[-1] != b2_filters:
            res2 = Conv2D(b2_filters, (1, 1), padding='same', kernel_initializer='he_normal',
                          use_bias=False, name=f'{name_prefix}_b2_res_conv')(x)
        else:
            res2 = x
        b2 = Add(name=f'{name_prefix}_b2_res')([b2, res2])
        b2 = BatchNormalization(epsilon=0.0002, name=f'{name_prefix}_b2_bn_2')(b2)
        b2 = Activation('relu', name=f'{name_prefix}_b2_act')(b2)

        # branch 3: 5x5 + 残差
        b3 = Conv2D(b3_filters, (5, 5), padding='same', kernel_initializer='he_normal',
                    use_bias=False, name=f'{name_prefix}_b3_conv')(x)
        b3 = BatchNormalization(epsilon=0.0002, name=f'{name_prefix}_b3_bn')(b3)

        if x.shape[-1] != b3_filters:
            res3 = Conv2D(b3_filters, (1, 1), padding='same', kernel_initializer='he_normal',
                          use_bias=False, name=f'{name_prefix}_b3_res_conv')(x)
        else:
            res3 = x
        b3 = Add(name=f'{name_prefix}_b3_res')([b3, res3])
        b3 = BatchNormalization(epsilon=0.0002, name=f'{name_prefix}_b3_bn_2')(b3)
        b3 = Activation('relu', name=f'{name_prefix}_b3_act')(b3)

        # branch 4: dilated 3x3 + 残差
        b4 = Conv2D(b4_filters, (3, 3), padding='same', dilation_rate=2, kernel_initializer='he_normal',
                    use_bias=False, name=f'{name_prefix}_b4_conv')(x)
        b4 = BatchNormalization(epsilon=0.0002, name=f'{name_prefix}_b4_bn')(b4)

        if x.shape[-1] != b4_filters:
            res4 = Conv2D(b4_filters, (1, 1), padding='same', kernel_initializer='he_normal',
                          use_bias=False, name=f'{name_prefix}_b4_res_conv')(x)
        else:
            res4 = x
        b4 = Add(name=f'{name_prefix}_b4_res')([b4, res4])
        b4 = BatchNormalization(epsilon=0.0002, name=f'{name_prefix}_b4_bn_2')(b4)
        b4 = Activation('relu', name=f'{name_prefix}_b4_act')(b4)

        # concat
        merged = Concatenate(name=f'{name_prefix}_concat')([b1, b2, b3, b4])

        # projection to desired out_filters (保持后续层通道一致)
        proj = Conv2D(out_filters, (1, 1), padding='same', kernel_initializer='he_normal',
                      use_bias=False, name=f'{name_prefix}_proj_conv')(merged)
        proj = BatchNormalization(epsilon=0.0002, name=f'{name_prefix}_proj_bn')(proj)
        proj = Activation('relu', name=f'{name_prefix}_proj_act')(proj)

        # 将 HeBing 应用于 projection 之后
        if heb_layer is None:
            # 若多处使用该模块，建议在模型构建时创建一个 HeBing 实例并传入 heb_layer 以便复用权重与正确跟踪
            heb_layer = HeBing(name=(None if name_prefix is None else f'{name_prefix}_hebing'))
        out = heb_layer(proj, training=training)

        return out

    def _define_model(self, input_shape, output_size) -> tuple:
        label_max_string_length = 64

        input_data = Input(name='the_input', shape=input_shape)

        # 替换每个原始 Conv2D(3x3) 为多尺度融合模块，输出通道数与原来一致

        layer_h = self._ms_block(input_data, 32, 'MS_Conv0')  # 替代 Conv0
        layer_h = self._ms_block(layer_h, 32, 'MS_Conv1')    # 替代 Conv1

        layer_h = MaxPooling2D(pool_size=2, strides=None, padding="valid", name='Pool0')(layer_h)

        layer_h = self._ms_block(layer_h, 64, 'MS_Conv2')    # 替代 Conv2
        layer_h = self._ms_block(layer_h, 64, 'MS_Conv3')    # 替代 Conv3

        layer_h = MaxPooling2D(pool_size=2, strides=None, padding="valid", name='Pool1')(layer_h)

        layer_h = self._ms_block(layer_h, 128, 'MS_Conv4')   # 替代 Conv4
        layer_h = self._ms_block(layer_h, 128, 'MS_Conv5')   # 替代 Conv5

        layer_h = MaxPooling2D(pool_size=2, strides=None, padding="valid", name='Pool2')(layer_h)

        layer_h = Conv2D(128, (7, 7), use_bias=True, padding='same', kernel_initializer='he_normal', name='Conv6')(
            layer_h)  # 卷积层
        layer_h = BatchNormalization(epsilon=0.0002, name='BN6')(layer_h)
        layer_h = Activation('relu', name='Act6')(layer_h)

        layer_h = Conv2D(128, (7, 7), use_bias=True, padding='same', kernel_initializer='he_normal', name='Conv7')(
            layer_h)  # 卷积层
        layer_h = BatchNormalization(epsilon=0.0002, name='BN7')(layer_h)
        layer_h = Activation('relu', name='Act7')(layer_h)

        # 原来使用 pool_size=1（没有尺寸变化），这里保留原样以兼容结构
        layer_h = MaxPooling2D(pool_size=1, strides=None, padding="valid", name='Pool3')(layer_h)

        layer_h = Conv2D(128, (7, 7), use_bias=True, padding='same', kernel_initializer='he_normal', name='Conv8')(
            layer_h)  # 卷积层
        layer_h = BatchNormalization(epsilon=0.0002, name='BN8')(layer_h)
        layer_h = Activation('relu', name='Act8')(layer_h)

        layer_h = Conv2D(128, (7, 7), use_bias=True, padding='same', kernel_initializer='he_normal', name='Conv9')(
            layer_h)  # 卷积层
        layer_h = BatchNormalization(epsilon=0.0002, name='BN9')(layer_h)
        layer_h = Activation('relu', name='Act9')(layer_h)

        layer_h = MaxPooling2D(pool_size=1, strides=None, padding="valid", name='Pool4')(layer_h)
        # --- Insert GCSA attention block here ---
        # layer_h is NHWC and has channels = 128 (as specified in Conv9)
        gcsa_block = GCSA(in_channels=128, rate=4, groups=4, name='GCSA0')
        layer_h = gcsa_block(layer_h)  # apply attention

        # Reshape: 计算每个时间步的特征维度
        # 由于 _ms_block 最终投影为 128 通道，频率维度被 pool 降采样为 input_shape[1] // _pool_size
        time_steps = self.output_shape[0]
        features_per_t = (input_shape[1] // self._pool_size) * 128
        layer_h = Reshape((time_steps, features_per_t), name='Reshape0')(layer_h)

        layer_h = Dense(128, activation="relu", use_bias=True, kernel_initializer='he_normal', name='Dense0')(layer_h)

        layer_h = Dense(output_size, use_bias=True, kernel_initializer='he_normal', name='Dense1')(layer_h)
        y_pred = Activation('softmax', name='Activation0')(layer_h)

        model_base = Model(inputs=input_data, outputs=y_pred)

        labels = Input(name='the_labels', shape=[label_max_string_length], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

        model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

        return model, model_base

    def get_loss_function(self) -> dict:
        return {'ctc': lambda y_true, y_pred: y_pred}

    def forward(self, data_input):
        batch_size = 1
        in_len = np.zeros((batch_size,), dtype=np.int32)
        in_len[0] = self.output_shape[0]

        x_in = np.zeros((batch_size,) + self.input_shape, dtype=np.float64)
        for i in range(batch_size):
            x_in[i, 0:len(data_input)] = data_input

        base_pred = self.model_base.predict(x=x_in)
        r = K.ctc_decode(base_pred, in_len, greedy=True, beam_width=100, top_paths=1)

        if tf.__version__[0:2] == '1.':
            r1 = r[0][0].eval(session=tf.compat.v1.Session())
        else:
            r1 = r[0][0].numpy()

        speech_result = ctc_decode_delete_tail_blank(r1[0])
        return speech_result


class SpeechModel251(BaseModel):
    """
    定义CNN+CTC模型，使用函数式模型

    输入层：200维的特征值序列，一条语音数据的最大长度设为1600（大约16s）\\
    隐藏层：卷积池化层，卷积核大小为3x3，池化窗口大小为2 \\
    隐藏层：全连接层 \\
    输出层：全连接层，神经元数量为self.MS_OUTPUT_SIZE，使用softmax作为激活函数， \\
    CTC层：使用CTC的loss作为损失函数，实现连接性时序多输出

    参数： \\
        input_shape: tuple，默认值(1600, 200, 1) \\
        output_shape: tuple，默认值(200, 1428)
    """

    def __init__(self, input_shape: tuple = (1600, 200, 1), output_size: int = 1428) -> None:
        super().__init__()
        self.input_shape = input_shape
        self._pool_size = 8
        self.output_shape = (input_shape[0] // self._pool_size, output_size)
        self._model_name = 'SpeechModel251'
        self.model, self.model_base = self._define_model(self.input_shape, self.output_shape[1])

    def _define_model(self, input_shape, output_size) -> tuple:
        label_max_string_length = 64

        input_data = Input(name='the_input', shape=input_shape)

        layer_h1 = Conv2D(32, (3, 3), use_bias=False, activation='relu', padding='same',
                          kernel_initializer='he_normal')(input_data)  # 卷积层
        layer_h1 = Dropout(0.05)(layer_h1)
        layer_h2 = Conv2D(32, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(
            layer_h1)  # 卷积层
        layer_h3 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h2)  # 池化层
        layer_h3 = Dropout(0.05)(layer_h3)  # 随机中断部分神经网络连接，防止过拟合

        layer_h4 = Conv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(
            layer_h3)  # 卷积层
        layer_h4 = Dropout(0.1)(layer_h4)
        layer_h5 = Conv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(
            layer_h4)  # 卷积层
        layer_h6 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h5)  # 池化层

        layer_h6 = Dropout(0.1)(layer_h6)
        layer_h7 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                          kernel_initializer='he_normal')(layer_h6)  # 卷积层
        layer_h7 = Dropout(0.15)(layer_h7)
        layer_h8 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                          kernel_initializer='he_normal')(layer_h7)  # 卷积层
        layer_h9 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h8)  # 池化层

        layer_h9 = Dropout(0.15)(layer_h9)
        layer_h10 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                           kernel_initializer='he_normal')(layer_h9)  # 卷积层
        layer_h10 = Dropout(0.2)(layer_h10)
        layer_h11 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                           kernel_initializer='he_normal')(layer_h10)  # 卷积层
        layer_h12 = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_h11)  # 池化层

        layer_h12 = Dropout(0.2)(layer_h12)
        layer_h13 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                           kernel_initializer='he_normal')(layer_h12)  # 卷积层
        layer_h13 = Dropout(0.2)(layer_h13)
        layer_h14 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                           kernel_initializer='he_normal')(layer_h13)  # 卷积层
        layer_h15 = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_h14)  # 池化层

        # test=Model(inputs = input_data, outputs = layer_h12)
        # test.summary()

        layer_h16 = Reshape((self.output_shape[0], input_shape[1] // self._pool_size * 128))(layer_h15)  # Reshape层
        layer_h16 = Dropout(0.3)(layer_h16)  # 随机中断部分神经网络连接，防止过拟合
        layer_h17 = Dense(128, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_h16)  # 全连接层
        layer_h17 = Dropout(0.3)(layer_h17)
        layer_h18 = Dense(output_size, use_bias=True, kernel_initializer='he_normal')(layer_h17)  # 全连接层
        y_pred = Activation('softmax', name='Activation0')(layer_h18)

        model_base = Model(inputs=input_data, outputs=y_pred)
        # model_data.summary()

        labels = Input(name='the_labels', shape=[label_max_string_length], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer
        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

        model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

        return model, model_base

    def get_loss_function(self) -> dict:
        return {'ctc': lambda y_true, y_pred: y_pred}

    def forward(self, data_input):
        batch_size = 1
        in_len = np.zeros((batch_size,), dtype=np.int32)

        in_len[0] = self.output_shape[0]

        x_in = np.zeros((batch_size,) + self.input_shape, dtype=np.float64)

        for i in range(batch_size):
            x_in[i, 0:len(data_input)] = data_input

        base_pred = self.model_base.predict(x=x_in)
        r = K.ctc_decode(base_pred, in_len, greedy=True, beam_width=100, top_paths=1)

        if tf.__version__[0:2] == '1.':
            r1 = r[0][0].eval(session=tf.compat.v1.Session())
        else:
            r1 = r[0][0].numpy()

        speech_result = ctc_decode_delete_tail_blank(r1[0])
        return speech_result


class SpeechModel25(BaseModel):
    """
    定义CNN+CTC模型，使用函数式模型

    输入层：200维的特征值序列，一条语音数据的最大长度设为1600（大约16s）\\
    隐藏层：卷积池化层，卷积核大小为3x3，池化窗口大小为2 \\
    隐藏层：全连接层 \\
    输出层：全连接层，神经元数量为self.MS_OUTPUT_SIZE，使用softmax作为激活函数， \\
    CTC层：使用CTC的loss作为损失函数，实现连接性时序多输出

    参数： \\
        input_shape: tuple，默认值(1600, 200, 1) \\
        output_shape: tuple，默认值(200, 1428)
    """

    def __init__(self, input_shape: tuple = (1600, 200, 1), output_size: int = 1428) -> None:
        super().__init__()
        self.input_shape = input_shape
        self._pool_size = 8
        self.output_shape = (input_shape[0] // self._pool_size, output_size)
        self._model_name = 'SpeechModel25'
        self.model, self.model_base = self._define_model(self.input_shape, self.output_shape[1])

    def _define_model(self, input_shape, output_size) -> tuple:
        label_max_string_length = 64

        input_data = Input(name='the_input', shape=input_shape)

        layer_h1 = Conv2D(32, (3, 3), use_bias=False, activation='relu', padding='same',
                          kernel_initializer='he_normal')(input_data)  # 卷积层
        layer_h1 = Dropout(0.05)(layer_h1)
        layer_h2 = Conv2D(32, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(
            layer_h1)  # 卷积层
        layer_h3 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h2)  # 池化层
        layer_h3 = Dropout(0.05)(layer_h3)  # 随机中断部分神经网络连接，防止过拟合

        layer_h4 = Conv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(
            layer_h3)  # 卷积层
        layer_h4 = Dropout(0.1)(layer_h4)
        layer_h5 = Conv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(
            layer_h4)  # 卷积层
        layer_h6 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h5)  # 池化层

        layer_h6 = Dropout(0.1)(layer_h6)
        layer_h7 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                          kernel_initializer='he_normal')(layer_h6)  # 卷积层
        layer_h7 = Dropout(0.15)(layer_h7)
        layer_h8 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                          kernel_initializer='he_normal')(layer_h7)  # 卷积层
        layer_h9 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h8)  # 池化层

        layer_h9 = Dropout(0.15)(layer_h9)
        layer_h10 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                           kernel_initializer='he_normal')(layer_h9)  # 卷积层
        layer_h10 = Dropout(0.2)(layer_h10)
        layer_h11 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                           kernel_initializer='he_normal')(layer_h10)  # 卷积层
        layer_h12 = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_h11)  # 池化层

        # test=Model(inputs = input_data, outputs = layer_h12)
        # test.summary()

        layer_h12 = Reshape((self.output_shape[0], input_shape[1] // self._pool_size * 128))(layer_h12)  # Reshape层
        layer_h12 = Dropout(0.3)(layer_h12)  # 随机中断部分神经网络连接，防止过拟合
        layer_h13 = Dense(128, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_h12)  # 全连接层
        layer_h13 = Dropout(0.3)(layer_h13)
        layer_h14 = Dense(output_size, use_bias=True, kernel_initializer='he_normal')(layer_h13)  # 全连接层
        y_pred = Activation('softmax', name='Activation0')(layer_h14)

        model_base = Model(inputs=input_data, outputs=y_pred)
        # model_data.summary()

        labels = Input(name='the_labels', shape=[label_max_string_length], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer
        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

        model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

        return model, model_base

    def get_loss_function(self) -> dict:
        return {'ctc': lambda y_true, y_pred: y_pred}

    def forward(self, data_input):
        batch_size = 1
        in_len = np.zeros((batch_size,), dtype=np.int32)

        in_len[0] = self.output_shape[0]

        x_in = np.zeros((batch_size,) + self.input_shape, dtype=np.float64)

        for i in range(batch_size):
            x_in[i, 0:len(data_input)] = data_input

        base_pred = self.model_base.predict(x=x_in)
        r = K.ctc_decode(base_pred, in_len, greedy=True, beam_width=100, top_paths=1)

        if tf.__version__[0:2] == '1.':
            r1 = r[0][0].eval(session=tf.compat.v1.Session())
        else:
            r1 = r[0][0].numpy()

        speech_result = ctc_decode_delete_tail_blank(r1[0])
        return speech_result


class SpeechModel24(BaseModel):
    """
    定义CNN+CTC模型，使用函数式模型

    输入层：200维的特征值序列，一条语音数据的最大长度设为1600（大约16s）\\
    隐藏层：卷积池化层，卷积核大小为3x3，池化窗口大小为2 \\
    隐藏层：全连接层 \\
    输出层：全连接层，神经元数量为self.MS_OUTPUT_SIZE，使用softmax作为激活函数， \\
    CTC层：使用CTC的loss作为损失函数，实现连接性时序多输出

    参数： \\
        input_shape: tuple，默认值(1600, 200, 1) \\
        output_shape: tuple，默认值(200, 1428)
    """

    def __init__(self, input_shape: tuple = (1600, 200, 1), output_size: int = 1428) -> None:
        super().__init__()
        self.input_shape = input_shape
        self._pool_size = 8
        self.output_shape = (input_shape[0] // self._pool_size, output_size)
        self._model_name = 'SpeechModel24'
        self.model, self.model_base = self._define_model(self.input_shape, self.output_shape[1])

    def _define_model(self, input_shape, output_size) -> tuple:
        label_max_string_length = 64

        input_data = Input(name='the_input', shape=input_shape)

        layer_h1 = Conv2D(32, (3, 3), use_bias=False, activation='relu', padding='same',
                          kernel_initializer='he_normal')(input_data)  # 卷积层
        layer_h1 = Dropout(0.1)(layer_h1)
        layer_h2 = Conv2D(32, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(
            layer_h1)  # 卷积层
        layer_h3 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h2)  # 池化层
        layer_h3 = Dropout(0.2)(layer_h3)  # 随机中断部分神经网络连接，防止过拟合

        layer_h4 = Conv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(
            layer_h3)  # 卷积层
        layer_h4 = Dropout(0.2)(layer_h4)
        layer_h5 = Conv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(
            layer_h4)  # 卷积层
        layer_h6 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h5)  # 池化层

        layer_h6 = Dropout(0.3)(layer_h6)
        layer_h7 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                          kernel_initializer='he_normal')(layer_h6)  # 卷积层
        layer_h7 = Dropout(0.3)(layer_h7)
        layer_h8 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                          kernel_initializer='he_normal')(layer_h7)  # 卷积层
        layer_h9 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h8)  # 池化层

        # test=Model(inputs = input_data, outputs = layer_h12)
        # test.summary()

        layer_h10 = Reshape((self.output_shape[0], input_shape[1] // self._pool_size * 128))(layer_h9)  # Reshape层
        layer_h10 = Dropout(0.3)(layer_h10)  # 随机中断部分神经网络连接，防止过拟合
        layer_h11 = Dense(128, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_h10)  # 全连接层
        layer_h11 = Dropout(0.3)(layer_h11)
        layer_h12 = Dense(output_size, use_bias=True, kernel_initializer='he_normal')(layer_h11)  # 全连接层
        y_pred = Activation('softmax', name='Activation0')(layer_h12)

        model_base = Model(inputs=input_data, outputs=y_pred)
        # model_data.summary()

        labels = Input(name='the_labels', shape=[label_max_string_length], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer
        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

        model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

        return model, model_base

    def get_loss_function(self) -> dict:
        return {'ctc': lambda y_true, y_pred: y_pred}

    def forward(self, data_input):
        batch_size = 1
        in_len = np.zeros((batch_size,), dtype=np.int32)

        in_len[0] = self.output_shape[0]

        x_in = np.zeros((batch_size,) + self.input_shape, dtype=np.float64)

        for i in range(batch_size):
            x_in[i, 0:len(data_input)] = data_input

        base_pred = self.model_base.predict(x=x_in)
        r = K.ctc_decode(base_pred, in_len, greedy=True, beam_width=100, top_paths=1)

        if tf.__version__[0:2] == '1.':
            r1 = r[0][0].eval(session=tf.compat.v1.Session())
        else:
            r1 = r[0][0].numpy()

        speech_result = ctc_decode_delete_tail_blank(r1[0])
        return speech_result