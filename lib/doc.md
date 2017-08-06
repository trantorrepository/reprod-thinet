INDEX















### Tensor

Tensors和numpy中的ndarrays较为相似, 因此Tensor也能够使用GPU来加速运算。

```
import torch

x = torch.Tensor(5, 3) # 构造一个未初始化的5*3的矩阵
x = torch.rand(5, 3)   # 构造一个随机初始化的矩阵
print(x)
x.size()
```






