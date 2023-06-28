from tools import *
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor as XGBR

from tools import *


def train_model_predict(df):
        
    d2 = df[['zbkzj','k1','k2','kbjj']]

    #提取数据
    d1 = d2.dropna().reset_index(drop=True)
    X = d1.drop(columns=['kbjj'])
    y = d1['kbjj']
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,shuffle=True,random_state=8)
    poly = PolynomialFeatures(degree=1)


    x_train = poly.fit_transform(X_train.values)
    x_test = poly.fit_transform(X_test)


    # 线性回归
    la = Lasso(alpha=0.1,max_iter=100000)
    la.fit(x_train,y_train)
    print(f'线性回归训练集得分：{round(la.score(x_train,y_train),2)}')
    print(f'线性回归测试集得分：{round(la.score(x_test,y_test),2)}')

    # 随机森林回归
    rf = RandomForestRegressor(n_jobs=-1)
    rf.fit(x_train,y_train)
    print(f'随机森林回归训练集得分：{round(rf.score(x_train,y_train),2)}')
    print(f'随机森林回归测试集得分：{round(rf.score(x_test,y_test),2)}')

    # 决策树回归
    dt = DecisionTreeRegressor(max_depth = 6)
    dt.fit(x_train,y_train)
    print(f'决策树回归训练集得分：{round(dt.score(x_train,y_train),2)}')
    print(f'决策树回归测试集得分：{round(dt.score(x_test,y_test),2)}')

    # K近邻回归
    kn = KNeighborsRegressor(n_neighbors=3,n_jobs=-1)
    kn.fit(x_train,y_train)
    print(f'k近邻回归测试集得分：{round(kn.score(x_test,y_test),2)}')

    # XGBbost
    xgb = XGBR(n_estimators=100)
    xgb.fit(x_train,y_train)
    # print(f"xgbbost测试集得分：{round(xgb.score(x_test,y_test),2)}") #你能想出这里应该返回什么模型评估指标么?

    return la,rf,dt,kn,xgb


def record_predict(df_train,KZJ):
    predict_k1 = zt(df.sample(n=8))['k1']
    predict_k2 = zt(df.sample(n=8))['k2']


    la,rf,dt,kn,xgb = train_model_predict(df_train)

    poly = PolynomialFeatures(degree=1)
    apply = np.array([KZJ,predict_k1,predict_k2]).reshape(1,-1)
    poly_apply = poly.fit_transform(apply)

    predict_list = [la.predict(poly_apply).tolist()[0],rf.predict(poly_apply).tolist()[0],dt.predict(poly_apply).tolist()[0],kn.predict(poly_apply).tolist()[0]]

    res=(sum(predict_list)-max(predict_list)-min(predict_list))/2

    xgb_mean = sum([rf.predict(poly_apply),kn.predict(poly_apply),xgb.predict(poly_apply)])/3
    return la.predict(poly_apply).tolist()[0],rf.predict(poly_apply).tolist()[0],dt.predict(poly_apply).tolist()[0],kn.predict(poly_apply).tolist()[0],res#,xgb.predict(poly_apply).tolist()[0],xgb_mean[0]







### 丽水市
df_total = lishui_data('lishui.csv')


df = lishui_data("丽水市第一页数据.csv")


# df_train = df[df['Date']<'2023-06']     #[['ZBKZJ','k1']]
# df_test = df[df['Date']>='2023-06']     #[['ZBKZJ','k1']]
# df_test = df

df['rule1'] = df['k1'].apply(lambda x: rule1(df_total))
df['rule1_result'] = abs(df['下浮率'] - df['rule1'])
df['rule2'] = df['k1'].apply(lambda x: rule2(df_total))
df['rule2_result'] = abs(df['下浮率'] - df['rule2'])
df['rule3'] = df['k1'].apply(lambda x: rule3(df_total))
df['rule3_result'] = abs(df['下浮率'] - df['rule3'])
df['rule4'] = df['rule1']*0.3 + df['rule2']*0.4 + df['rule3']*0.3
df['rule4_result'] = abs(df['下浮率'] - df['rule4'])
print(len(df[df['rule1_result'] < 1]), len(df[df['rule2_result'] < 1]), len(df[df['rule3_result'] < 1]), len(df[df['rule4_result']<1]))
print(len(df[df['rule1_result'] == 0]), len(df[df['rule2_result'] == 0]), len(df[df['rule3_result'] == 0]), len(df[df['rule4_result']==0]))

# plt.plot(range(len(df_test)),df_test[['下浮率','rule1','rule2']],marker = 'o',label = ['下浮率','rule1','rule2'])
# plt.legend()
# plt.show()
df.to_csv('lishui测试.csv', index=False)


plt.hist(df['下浮率'], bins=20)





### 六安市
df = luan_data()








### 淮南市 
df = huainan_data()
df_train,df_test = train_test_split(df,test_size=0.3,shuffle=True,random_state=8)
KZJ = 2473043.39
record_predict(df_train,KZJ)

df[df['zbkzj'] == KZJ].iloc[0]['kbjj']

# plt.plot(range(len(df)),df[['zbkzj','kbjj']],marker = 'o',label = ['zbkzj','kbjj'])
plt.scatter(df['zbkzj'],df['下浮率'],marker='o')
plt.legend()
plt.show()

df.to_csv('huainan预测值对比.csv',index=False)













# df_test[['线性回归', "随机森林",'决策树','K近邻','XGB','预测值']] = df_test.apply(lambda x:record_predict(df_train,x['ZBKZJ']),axis=1,result_type='expand')
# df_train[['线性回归', "随机森林",'决策树','K近邻','XGB','预测值']] = df_train.apply(lambda x:record_predict(df_train,x['ZBKZJ']),axis=1,result_type='expand')




# df_test.to_csv('预测值对比.csv',index=False)

# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.plot(range(len(df_test)),df_test[['k1','线性回归', "随机森林",'决策树','K近邻','XGB','预测值']],marker = 'o',label = ['k1','线性回归', "随机森林",'决策树','K近邻','XGB','预测值'])
# plt.legend()
# plt.show()

# chatgpt_k1 = [8.3, 7.6, 9.1, 8.8, 7.9, 8.5, 7.2, 9.0, 8.2, 8.6, 7.7, 8.4, 8.1, 7.8, 8.0, 7.5, 8.7, 7.4, 8.9, 7.3, 8.8, 7.6, 9.2, 8.3, 7.9, 9.1, 8.5, 7.7, 9.0, 8.6, 7.8, 8.7, 7.5, 8.9, 7.4, 8.8, 7.6, 9.3, 8.4, 7.9, 9.2, 8.5, 7.7, 9.1, 8.6, 7.8, 9.0, 8.7, 7.5, 9.2]
# df_test['chatgpt_k1'] = chatgpt_k1[:30]

# plt.plot(range(len(df_test)),df_test[['下浮率', "chatgpt_k1"]],marker = 'o')

# plt.sc


# # x.index(max(x))
# x.index(min(x))




# # 遍历所有数据，得到KZJ
# #     生成k1
# #     训练模型做预测，但是每调用一次，模型会变化一次，当生成两个k的时候，调用两次会模型训练数据效果不一致，期望同一个模型，不同k




import numpy as np
import matplotlib.pyplot as plt
random_data = df['下浮率'].values.tolist()
avg_data = list()
for i in range(0, 10000):
    one_sum = 0
    for j in range(0, 50):
        one_sum += random_data[np.random.randint(0, len(df))]
    avg_data.append(one_sum *1.0 / 50.0)
print(avg_data)
plt.hist(avg_data, bins=100)
plt.show()




import numpy as np

# 生成随机数据
np.random.seed(42)  # 设置随机种子，以保证结果可复现
# data = np.random.uniform(6, 15, size=1000)  # 生成1000个在6到15之间的随机数

# # 划分数据区间
# bins = np.linspace(6, 15, num=10)  # 将数据范围划分为10个区间


data = df['下浮率'].values.tolist()
bins = np.linspace(max(data), min(data), num=10)
num_bins = 10
bins = np.linspace(6, 15, num_bins + 1)  # Create bins with equal width

# Calculate histogram
hist, _ = np.histogram(data, bins=bins)

# Find the bin with the maximum count
max_index = np.argmax(hist)
max_bin_start = bins[max_index]
max_bin_end = bins[max_index + 1]

# Calculate the percentage of data in the max bin
total_data = len(data)
max_bin_count = hist[max_index]
max_bin_percentage = (max_bin_count / total_data) * 100

# Output the result
print(f"The data interval with the highest percentage is [{max_bin_start}, {max_bin_end})")
print(f"It contains {max_bin_percentage:.2f}% of the data.")


import numpy as np

def find_optimal_intervals(data, k):
    n = len(data)
    data_sorted = sorted(data)

    dp = np.zeros((n, k+1))
    prev = np.zeros((n, k+1), dtype=int)

    for i in range(n):
        dp[i][1] = (i+1) / n

    for j in range(2, k+1):
        for i in range(j-1, n):
            max_val = -np.inf
            max_prev = 0
            for p in range(i):
                val = dp[p][j-1] + (i-p) / n
                if val > max_val:
                    max_val = val
                    max_prev = p
            dp[i][j] = max_val
            prev[i][j] = max_prev

    max_val = np.max(dp[:, k])
    max_idx = np.argmax(dp[:, k])

    intervals = []
    for _ in range(k, 0, -1):
        start = prev[max_idx][k]
        interval = (data_sorted[start], data_sorted[max_idx])
        intervals.append(interval)
        max_idx = start
        k -= 1

    return max_val, intervals[::-1]

# Generate random data
np.random.seed(42)
data = np.random.uniform(6, 15, size=1000)

# Set the number of intervals
num_intervals = 10

data = df['下浮率'].values.tolist()

# Find the optimal intervals
max_percentage, intervals = find_optimal_intervals(data, num_intervals)

# Output the result
print(f"The optimal intervals with the maximum percentage are:")
for i, interval in enumerate(intervals):
    print(f"Interval {i+1}: {interval}")
print(f"They collectively contain {max_percentage*100:.2f}% of the data.")
