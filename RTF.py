import time
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from speech_model import ModelSpeech
from model_zoo.speech_model.keras_backend import SpeechModel251BN
from data_loader import DataLoader
from speech_features import SpecAugment
import subprocess
import platform

# ==========================================
# 1. 环境与基础参数配置 [cite: 311, 313]
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
# 修改后：禁用所有 GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 限制 CPU 线程数（例如限制为 4 线程，模拟典型边缘设备）
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

AUDIO_LENGTH = 1600
AUDIO_FEATURE_LENGTH = 200
CHANNELS = 1
OUTPUT_SIZE = 1428
MAX_LABEL_LENGTH = 64 # 对应论文中的最大标签长度 [cite: 316]

# ==========================================
# 2. 复杂度分析工具函数 (已修复多输入问题) [cite: 371, 373]
# ==========================================
def analyze_keras_model_complexity(model):
    """ 计算模型总参数量，验证是否为论文中的 4.65M  """
    total_params = model.count_params()
    print("\n" + "=" * 45)
    print(f"模型参数分析 (Model Complexity):")
    print(f"总参数量: {total_params / 1e6:.2f} M")
    print("=" * 45)

def analyze_pure_inference_complexity(model):
    """ 计算推理复杂度，验证是否为 72.32 GFLOPs [cite: 370, 373] """
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

    # 修复：为模型的所有输入创建 TensorSpec
    input_specs = []
    for input_node in model.inputs:
        # 将张量形状中的 None 替换为 1 (Batch Size)
        shape = [d if d is not None else 1 for d in input_node.shape]
        input_specs.append(tf.TensorSpec(shape, input_node.dtype))

    @tf.function
    def run_inference(*inputs):
        return model(inputs)

    concrete_func = run_inference.get_concrete_function(*input_specs)
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        opts['output'] = 'none'
        flops = tf.compat.v1.profiler.profile(graph=graph, options=opts)

        if flops is not None:
            print("\n" + "=" * 45)
            print("推理复杂度分析 (Inference Complexity):")
            print(f"总计算量: {flops.total_float_ops / 1e9:.2f} GFLOPs")
            print("=" * 45)


def get_hardware_name():
    # 检查 GPU 是否可用且未被禁用
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 0 and os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        try:
            # 使用 nvidia-smi 命令获取第一块显卡的名称
            gpu_name = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                encoding='utf-8'
            ).strip()
            return f"{gpu_name} (GPU)"
        except Exception:
            return "Unknown NVIDIA GPU"

    # 如果没有 GPU 或显式禁用了 GPU，则检测 CPU
    try:
        # 使用 platform 获取处理器基本信息
        cpu_name = platform.processor()
        # 如果平台信息较模糊，尝试读取更详细的 CPU 型号
        if not cpu_name or "x86" in cpu_name:
            if platform.system() == "Windows":
                cpu_name = subprocess.check_output(["wmic", "cpu", "get", "name"], encoding='utf-8').split('\n')[
                    1].strip()
            elif platform.system() == "Linux":
                # 从 /proc/cpuinfo 中提取型号名称
                command = "cat /proc/cpuinfo | grep 'model name' | uniq"
                cpu_name = subprocess.check_output(command, shell=True, encoding='utf-8').split(':')[-1].strip()
        return f"{cpu_name} (CPU Only)"
    except Exception:
        return platform.machine() + " Processor"

# ==========================================
# 3. 实例化 MSRDA-GSCA 网络架构 [cite: 3, 122]
# ==========================================
sm251bn = SpeechModel251BN(
    input_shape=(AUDIO_LENGTH, AUDIO_FEATURE_LENGTH, CHANNELS),
    output_size=OUTPUT_SIZE
)

if hasattr(sm251bn, 'model'):
    analyze_keras_model_complexity(sm251bn.model)
    analyze_pure_inference_complexity(sm251bn.model)

# ==========================================
# 4. 实时率 (RTF) 测试模块 [cite: 391]
# ==========================================
print("\n" + "="*35)
print("开始进行实时率 (RTF) 测试...")

# 构造 4 个输入的哑数据
input_1 = np.random.random((1, AUDIO_LENGTH, AUDIO_FEATURE_LENGTH, CHANNELS)).astype('float32') # 语音特征
input_2 = np.zeros((1, MAX_LABEL_LENGTH)).astype('float32') # 标签 [cite: 316]
input_3 = np.array([[AUDIO_LENGTH // 8]]).astype('float32') # 输入长度 (经过下采样)
input_4 = np.array([[MAX_LABEL_LENGTH]]).astype('float32') # 标签长度

dummy_inputs = [input_1, input_2, input_3, input_4]

# 模型预热 [cite: 311]
for _ in range(10):
    _ = sm251bn.model.predict(dummy_inputs)

# 正式计时推理
test_iters = 100
start_time = time.perf_counter()
for _ in range(test_iters):
    _ = sm251bn.model.predict(dummy_inputs)
end_time = time.perf_counter()

# 计算指标
sample_audio_duration = 16.0 # 假设 1600 帧对应 16 秒音频
avg_inference_time = (end_time - start_time) / test_iters
rtf = avg_inference_time / sample_audio_duration

# 自动检测当前使用的设备
device_name = get_hardware_name()
print(f"检测到的硬件环境: {device_name}")

print(f"硬件环境: {device_name}")

print(f"单次推理平均耗时: {avg_inference_time:.4f} 秒")
print(f">>> 实时率 (RTF): {rtf:.4f} <<<")

if rtf < 1.0:
    print(f"结论: MSRDA-GSCA 在 {device_name.split()[0]} 上满足实时处理要求。") #