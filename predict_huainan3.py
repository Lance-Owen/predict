from tools import *

# df = mysql_select_df("SELECT *  from tender_bid_opening WHERE city = '淮南市'")
# df[["id","county","classify","project_type","title","zbkzj","kbjj","k1","k2","numbers_bidders","publish_time","source_website_address","trade_method"]].to_csv('淮南市数据.csv',index=False)
df = pd.read_csv('淮南市数据.csv')
# df[['zbkzj', 'kbjj']] = df[['zbkzj', 'kbjj']].fillna(0)
# df[['zbkzj', 'kbjj']] = df[['zbkzj', 'kbjj']].astype(float)
# df = df[(df['zbkzj'] > df['kbjj']) & (df['kbjj'] / df['zbkzj'] > 0.8)]
# df.to_csv('淮南市数据.csv',index=False)
df[['k1', 'k2']] = df[['k1', 'k2']].fillna(0)
df[['k1', 'k2']] = df[['k1', 'k2']].astype(float)
df[['zbkzj', 'kbjj']] = df[['zbkzj', 'kbjj']].fillna(0)
df[['zbkzj', 'kbjj']] = df[['zbkzj', 'kbjj']].astype(float)
df['下浮率'] = 100-100*df['kbjj']/df['zbkzj']
def split_time(t):
    t = re.findall('(\d{4,4})-(\d{1,2})-(\d{1,2})', str(t))[0]
    return int(t[0]), int(t[1]), int(t[2])

df[['年', '月', '日']] = df.apply(lambda x: split_time(x['publish_time']), result_type='expand', axis=1)


degree = 4
pre_label = '下浮率'

# 计算方法一
print('计算方法一数据'.center(100, '*'))
df1 = df[(0.35 <= df['k1']) & (df['k1'] <= 0.6) & (df['k2'] > 0)]

df1 = df1[(df1['zbkzj'] > df1['kbjj']) & (df1['kbjj'] / df1['zbkzj'] > 0.8)]
print(df1['k1'].value_counts())
print(df1['k2'].value_counts())
print(len(df1))
print('可用数据', len(df1))
df1 = df1[['zbkzj','年', '月', '日', pre_label]]
df1 = df1.fillna(-999)
print(record_predict(df1,[41020100,2023,8,11],degree))

# 计算方法二
print('计算方法二数据'.center(100, '*'))
df2 = df[(0 < df['k1']) & (0.6 > df['k1']) & (df['k2'] == 0)]
df2 = df2[(df2['zbkzj'] > df2['kbjj']) & (df2['kbjj'] / df2['zbkzj'] > 0.8)]
print(df2['k1'].value_counts())
print('可用数据', len(df2))

# 计算方法三，四
print('计算方法三数据'.center(100, '*'))
df3 = df[(df['k1'] < 0) & (df['k2'] < 0)]
df3 = df3[(df3['zbkzj'] > df3['kbjj']) & (df3['kbjj'] / df3['zbkzj'] > 0.8)]
df3['k'] = (df3['k1'] + df3['k2']) / 2
print(df3['k'].value_counts())
print(len(df3))
print('可用数据', len(df3))

# 计算方法五
print('计算方法五数据'.center(100, '*'))
df5 = df[(0 < df['k1']) & (0.6 > df['k1']) & (df['k2'] < 0)]
df5 = df5[(df5['zbkzj'] > df5['kbjj']) & (df5['kbjj'] / df5['zbkzj'] > 0.8)]
print(df5['k1'].value_counts())
print(df5['k2'].value_counts())
print('可用数据', len(df5))
