# # 预测6-11.9
# import random
# a = [6,7,8,9,10,11]
# b = [0,0.1,0.22,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
#
# random.seed(2070)
#
# for i in range(21):
#     print(random.choice(a))
#     print(random.choice(b))
import random

import numpy as np

# 设定随机数生成器的种子以保证结果可复现
np.random.seed(1)

# 生成23个在【0.4，0.45，0.5，0.55，0.6】范围内的随机数
data = np.random.choice([0.4, 0.45, 0.5, 0.55, 0.6], 40)

# 打印原始数据
print("原始数据:", list(data))

# 根据规律调整数据：如果数据大于0.55或者小于0.45，那么就向反方向偏移0.05，但不修改当前数据
adjusted_data = []
for i in range(len(data)):
    if i > 0 and data[i - 1] >0.55:
        pre = data[i] - 0.05 * random.choice([0,1])
        pre = 0.4 if pre<0.4 else pre
        adjusted_data.append(max(pre,data[i]))
    elif i > 0 and data[i - 1] < 0.45:
        pre = data[i] + 0.05 * random.choice([0,1])
        pre = 0.6 if pre>0.6 else pre
        adjusted_data.append(min(pre,data[i]))
    else:
        adjusted_data.append(data[i])

    # 打印调整后的数据
print("调整后的数据:", adjusted_data)


