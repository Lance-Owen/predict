from tools import *

df = pd.read_excel('池州市项目预测0801.xlsx')
df['开标时间'] = pd.to_datetime(df['开标时间'], format="%Y-%d-%m %H:%M:%S")


# print(df.columns)
def split_time(t):
    t = re.findall('(\d{4,4})-(\d{1,2})-(\d{1,2})\s(\d{1,2}):(\d{1,2}):\d{1,2}', str(t))[0]
    return int(t[0]), int(t[1]), int(t[2]), int(t[3]), int(t[4])


df[['年', '月', '日', '时', '分']] = df.apply(lambda x: split_time(x['开标时间']), result_type='expand', axis=1)

print(df.columns)
df = df[[
    'zbkzj', 'kbjj', 'k1', 'k2', '中标金额', '下浮率', '投标人数', 'project_type', '取值范围', '年', '月', '日', '时', '分']]


degree = 1
pre_label = 'k2'
df1 = df[df['取值范围'].str.contains('0.5-10')]
df1 = df1[['zbkzj','年', '月', '日', '时', '分', pre_label]]
df1 = df1.fillna(-999)
# print(record_predict(df1,[30626054.79,2022,10,8,11,30],degree))




df2 = df[df['取值范围'].str.contains('0.5-15')]
df2 = df2[['zbkzj','年', '月', '日', '时', '分', pre_label]]
df2 = df2.fillna(-999)
# print(record_predict(df2,[16216306.33,2023,8,18,8,30],degree))
print(record_predict(df2,[9673409.49,2022,8,25,9,0],degree))


df3 = df[df['取值范围'].str.contains('98-102')]
df3 = df3[['zbkzj','年', '月', '日', '时', '分', pre_label]]
df3 = df3.fillna(-999)
# print(record_predict(df3,[18361714.36,2023,8,2,9,0],degree))


df4 = df[df['取值范围'].str.contains('k:0.1，0.2，0.3')]
df5 = df[df['取值范围'].str.contains('k:0.2，0.3，0.4')]
df6 = df[df['取值范围'].str.contains('k:0.3，0.4，0.5')]

# df5 = df[df['取值范围'].str.contains('0.5-10')]
print(123)
