# recompute

通过重计算来节省显存，参考论文[《Training Deep Nets with Sublinear Memory Cost》](https://arxiv.org/abs/1604.06174)。

本程序已经内置在[bert4keras](https://github.com/bojone/bert4keras)中

## 使用方法

首先，确保环境变量加上`RECOMPUTE=1`。

然后，在自定义层的时候，用`recompute_grad`装饰call函数即可：
```python
from recompute import recompute_grad

class MyLayer(Layer):
    @recompute_grad
    def call(self, inputs):
        return inputs * 2
```

如果是现成的层，可以通过继承的方式来装饰：
```python
from recompute import recompute_grad

class MyDense(Dense):
    @recompute_grad
    def call(self, inputs):
        super(MyDense, self).call(inputs)
```

## 支持环境

在下面的环境下测试通过：
```
tensorflow 1.14 + keras 2.3.1
tensorflow 1.15 + keras 2.3.1
tensorflow 2.0 + keras 2.3.1
tensorflow 2.1 + keras 2.3.1
tensorflow 2.0 + 自带tf.keras
tensorflow 2.1 + 自带tf.keras
```

## 参考内容
- https://kexue.fm
- https://github.com/bojone/bert4keras
- https://github.com/davisyoshida/tf2-gradient-checkpointing
- https://github.com/tensorflow/tensorflow/blob/v2.1.0/tensorflow/python/ops/custom_gradient.py#L454-L499
