from tools import *

# df = mysql_select_df("SELECT *  from ggzyk_forecast WHERE city = '淮南市'")
# df.to_csv('淮南市数据.csv',index=False)
df = pd.read_csv('淮南市数据.csv')
df[['k1', 'k2']] = df[['k1', 'k2']].fillna(0)
# 计算方法一
print('计算方法一数据'.center(100, '*'))
df1 = df[(0.35 <= df['k1']) & (df['k1'] <= 0.6) & (df['k2'] > 0)]

df1 = df1[(df1['ZBKZJ'] > df1['kbjj']) & (df1['kbjj'] / df1['ZBKZJ'] > 0.8)]
print(df1['k1'].value_counts())
print(df1['k2'].value_counts())
print(len(df1))
print('可用数据', len(df1))

# 计算方法二
print('计算方法二数据'.center(100, '*'))
df2 = df[(0 < df['k1']) & (0.6 > df['k1']) & (df['k2'] == 0)]
df2 = df2[(df2['ZBKZJ'] > df2['kbjj']) & (df2['kbjj'] / df2['ZBKZJ'] > 0.8)]
print(df2['k1'].value_counts())
print('可用数据', len(df2))

# 计算方法三，四
print('计算方法三数据'.center(100, '*'))
df3 = df[(df['k1'] < 0) & (df['k2'] < 0)]
df3 = df3[(df3['ZBKZJ'] > df3['kbjj']) & (df3['kbjj'] / df3['ZBKZJ'] > 0.8)]
df3['k'] = (df3['k1'] + df3['k2']) / 2
print(df3['k'].value_counts())
print(len(df3))
print('可用数据', len(df3))

# 计算方法五
print('计算方法五数据'.center(100, '*'))
df5 = df[(0 < df['k1']) & (0.6 > df['k1']) & (df['k2'] < 0)]
df5 = df5[(df5['ZBKZJ'] > df5['kbjj']) & (df5['kbjj'] / df5['ZBKZJ'] > 0.8)]
print(df5['k1'].value_counts())
print(df5['k2'].value_counts())
print('可用数据', len(df5))
