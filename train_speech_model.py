# !/usr/bin/env python3
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
用于训练语音识别系统语音模型的程序
(已集成网络参数量与计算复杂度评估工具)
"""

import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from speech_model import ModelSpeech
from model_zoo.speech_model.keras_backend import SpeechModel251BN
from data_loader import DataLoader
from speech_features import SpecAugment


# =====================================================================
# 工具函数 1：分析包含 CTC 损失的完整训练图复杂度与参数量
# =====================================================================
def analyze_keras_model_complexity(keras_model):
    """
    自动分析 Keras 模型的参数量与计算复杂度 (支持多输入 CTC 训练模型)
    """
    print("\n" + "=" * 50)
    print("📊 模型参数量分析 (Parameter Count)")
    keras_model.summary()

    print("\n" + "-" * 50)
    print("⏱️ 模型整体复杂度分析 (包含 CTC 的训练图 FLOPs)")
    try:
        tensor_specs = []
        for keras_input in keras_model.inputs:
            shape = list(keras_input.shape)
            shape[0] = 1  # 固定 Batch Size = 1 进行标准评估
            tensor_specs.append(tf.TensorSpec(shape, keras_input.dtype))

        @tf.function
        def forward_pass(inputs):
            return keras_model(inputs)

        concrete_func = forward_pass.get_concrete_function(tensor_specs)
        frozen_func = convert_variables_to_constants_v2(concrete_func)
        frozen_func.graph.as_graph_def()

        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        opts['output'] = 'none'

        flops = tf.compat.v1.profiler.profile(
            graph=frozen_func.graph,
            run_meta=run_meta,
            cmd='op',
            options=opts
        )

        total_flops = flops.total_float_ops
        print(f"   模型输入节点数: {len(keras_model.inputs)}")
        print(f"   总浮点运算数 (FLOPs): {total_flops / 1e9:.4f} G")
        print(f"   推算 MACs:           {(total_flops / 2) / 1e9:.4f} G")
        print("=" * 50 + "\n")

    except Exception as e:
        print(f"\n[警告] 整体图复杂度计算失败: {e}\n")


# =====================================================================
# 工具函数 2：剥离 CTC 节点，纯测试部署推理的 FLOPs（用于写论文的核心数据）
# =====================================================================
def analyze_pure_inference_complexity(training_model):
    """
    剥离 CTC 训练节点，仅分析纯声学模型的推理复杂度 (Inference FLOPs)
    """
    print("\n" + "=" * 50)
    print("🔍 提取纯推理模型 (Inference Model)")

    try:
        # ASRT 默认输入层为 'the_input'，输出层为 'Activation0'
        inference_model = tf.keras.Model(
            inputs=training_model.get_layer('the_input').input,
            outputs=training_model.get_layer('Activation0').output,
            name="pure_inference_model"
        )

        print(f"   成功截取图结构: {inference_model.inputs[0].name} -> {inference_model.outputs[0].name}")

        input_shape = list(inference_model.inputs[0].shape)
        input_shape[0] = 1
        tensor_spec = tf.TensorSpec(input_shape, inference_model.inputs[0].dtype)

        @tf.function
        def forward_pass(inputs):
            return inference_model(inputs)

        concrete_func = forward_pass.get_concrete_function(tensor_spec)
        frozen_func = convert_variables_to_constants_v2(concrete_func)
        frozen_func.graph.as_graph_def()

        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        opts['output'] = 'none'

        flops = tf.compat.v1.profiler.profile(
            graph=frozen_func.graph,
            run_meta=run_meta,
            cmd='op',
            options=opts
        )

        total_flops = flops.total_float_ops
        print("-" * 50)
        print("⏱️ 纯推理复杂度分析 (Inference FLOPs)")
        print(f"   输入张量维度: {input_shape}")
        print(f"   纯推理 FLOPs: {total_flops / 1e9:.4f} G")
        print(f"   纯推理 MACs:  {(total_flops / 2) / 1e9:.4f} G")
        print("=" * 50 + "\n")

    except Exception as e:
        print(f"\n[错误] 截取推理模型失败，请检查层名称是否匹配: {e}\n")


# =====================================================================
# 主训练流程
# =====================================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

AUDIO_LENGTH = 1600
AUDIO_FEATURE_LENGTH = 200
CHANNELS = 1
# 默认输出的拼音的表示大小是1428，即1427个拼音+1个空白块0
OUTPUT_SIZE = 1428

# 1. 实例化核心网络架构
sm251bn = SpeechModel251BN(
    input_shape=(AUDIO_LENGTH, AUDIO_FEATURE_LENGTH, CHANNELS),
    output_size=OUTPUT_SIZE
)

# 2. 调用自动分析工具获取论文所需数据
if hasattr(sm251bn, 'model'):
    analyze_keras_model_complexity(sm251bn.model)
    analyze_pure_inference_complexity(sm251bn.model)
else:
    print("未找到底层 Keras 模型，无法自动计算复杂度。")

# 3. 准备数据增强与数据集
feat = SpecAugment()
train_data = DataLoader('train')
opt = Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, decay=0.0, epsilon=10e-8)

# 4. 封装为可训练对象
ms = ModelSpeech(sm251bn, feat, max_label_length=64)

# ms.load_model('save_models/' + sm251bn.get_model_name() + '.model.h5')

# 5. 开始训练
ms.train_model(optimizer=opt, data_loader=train_data,
               epochs=50, save_step=1, batch_size=4, last_epoch=0)

# 6. 保存模型
ms.save_model('save_models/' + sm251bn.get_model_name())