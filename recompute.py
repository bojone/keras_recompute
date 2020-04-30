#! -*- coding: utf-8 -*-
# recompute for keras/tf

import os
import tensorflow as tf
from tensorflow.python.util import nest, tf_inspect
from tensorflow.python.eager import tape
from tensorflow.python.ops.custom_gradient import _graph_mode_decorator

# 判断是tf.keras还是纯keras的标记
is_tf_keras = strtobool(os.environ.get('TF_KERAS', '0'))

if is_tf_keras:
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
    sys.modules['keras'] = keras
else:
    import keras
    import keras.backend as K

# 判断是否启用重计算（通过时间换空间）
do_recompute = strtobool(os.environ.get('RECOMPUTE', '0'))


def graph_mode_decorator(f, *args, **kwargs):
    """tf 2.1与之前版本的传参方式不一样，这里做个同步
    """
    if tf.__version__ < '2.1':
        return _graph_mode_decorator(f, *args, **kwargs)
    else:
        return _graph_mode_decorator(f, args, kwargs)


def recompute_grad(call):
    """重计算装饰器（用来装饰Keras层的call函数）
    关于重计算，请参考：https://arxiv.org/abs/1604.06174
    """
    if not do_recompute:
        return call

    def inner(self, inputs, **kwargs):
        """定义需要求梯度的函数以及重新定义求梯度过程
        （参考自官方自带的tf.recompute_grad函数）
        """
        call_args = tf_inspect.getfullargspec(call).args
        for key in ['mask', 'training']:
            if key not in call_args and key in kwargs:
                del kwargs[key]

        flat_inputs = nest.flatten(inputs)

        def kernel_call():
            """定义前向计算
            """
            return call(self, inputs, **kwargs)

        def call_and_grad(*inputs):
            """定义前向计算和反向计算
            """
            if is_tf_keras:
                with tape.stop_recording():
                    outputs = kernel_call()
                    outputs = tf.identity(outputs)
            else:
                outputs = kernel_call()

            def grad_fn(doutputs, variables=None):
                watches = list(inputs)
                if variables is not None:
                    watches += list(variables)
                with tf.GradientTape() as t:
                    t.watch(watches)
                    with tf.control_dependencies([doutputs]):
                        outputs = kernel_call()
                grads = t.gradient(
                    outputs, watches, output_gradients=[doutputs]
                )
                del t
                return grads[:len(inputs)], grads[len(inputs):]

            return outputs, grad_fn

        if is_tf_keras:  # 仅在tf >= 2.0下可用
            outputs, grad_fn = call_and_grad(*flat_inputs)
            flat_outputs = nest.flatten(outputs)

            def actual_grad_fn(*doutputs):
                grads = grad_fn(*doutputs, variables=self.trainable_weights)
                return grads[0] + grads[1]

            watches = flat_inputs + self.trainable_weights
            watches = [tf.convert_to_tensor(x) for x in watches]
            tape.record_operation(
                call.__name__, flat_outputs, watches, actual_grad_fn
            )
            return outputs
        else:  # keras + tf >= 1.14 均可用
            return graph_mode_decorator(call_and_grad, *flat_inputs)

    return inner
