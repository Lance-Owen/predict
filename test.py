import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import random

# 输入数据
# data = [0.98, 0.94, 0.92, 0.96, 0.96, 0.92, 0.92, 0.96, 0.92, 0.92, 0.96, 0.96, 0.96, 0.92, 0.92, 0.96, 0.98, 0.98, 0.98, 0.94, 0.9, 0.92, 0.94]

data = [0.4, 0.6, 0.45, 0.45, 0.55, 0.5, 0.5, 0.4, 0.4, 0.5, 0.6, 0.45, 0.5, 0.4, 0.45, 0.45, 0.4, 0.4, 0.6, 0.6, 0.6, 0.45, 0.45, 0.45, 0.6, 0.4, 0.55, 0.6, 0.55, 0.45, 0.6, 0.4, 0.45, 0.4, 0.4, 0.55, 0.55, 0.55, 0.55, 0.6, 0.6, 0.6, 0.45, 0.45, 0.4, 0.5, 0.45, 0.5, 0.5, 0.45, 0.4, 0.4, 0.4, 0.6, 0.45, 0.6, 0.4, 0.5, 0.55, 0.55, 0.6, 0.55, 0.4, 0.4, 0.6]


for i in range(10):
        # 创建一个时间索引
        date_index = pd.date_range(start='2023-01-01 9:00', periods=len(data), freq='D')

        # 创建DataFrame
        df = pd.DataFrame({'data': data}, index=date_index)

        # 使用指数平滑进行预测
        model = ExponentialSmoothing(df['data'], trend='add', seasonal=None, initialization_method="estimated")
        model_fit = model.fit()

        # 预测下一个结果
        predicted_value = model_fit.forecast(steps=1)

        print("预测结果：", predicted_value[0])
        print(data)
        # data.append(float(predicted_value[0]))

        data.append(random.choice(data))

