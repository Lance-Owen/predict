import pandas as pd
from tools import *


df_total = luan_data()


sql = "SELECT id,county,classify,project_type,zbkzj,k1,k2,kbjj,publish_time,source_website_address,numbers_bidders,trade_method FROM tender_bid_opening WHERE city = '六安市' ORDER BY publish_time desc LIMIT 100"
df = mysql_select_df(sql)
df = df[df['zbkzj'] > df["kbjj"]]
df['下浮率'] = 100 - round(df['kbjj'] / df['zbkzj'] * 100, 2)
df = df[df['下浮率'] > 6]
df = df[df['下浮率'] < 12]


df = df_total
print(len(df))

df['rule1'] = df['zbkzj'].apply(lambda x: rule1(df_total))
df['rule1_result'] = abs(df['下浮率'] - df['rule1'])
print(len(df[df['rule1_result'] < 1]),len(df[df['rule1_result'] == 0]))

df['rule2'] = df['zbkzj'].apply(lambda x: rule2(df_total,x))
df['rule2_result'] = abs(df['下浮率'] - df['rule2'])
print(len(df[df['rule2_result'] < 1]),len(df[df['rule2_result'] == 0]))

print(f"下浮率误差1%以内命中率：{round(len(df[df['rule2_result'] < 1])/len(df),2)*100}%，下浮率命中率：{round(len(df[df['rule2_result'] == 0])/len(df),2)*100}%")

df['rule3'] = df['zbkzj'].apply(lambda x: rule3(df_total, x))
df['rule3_result'] = abs(df['下浮率'] - df['rule3'])
print(len(df[df['rule3_result'] < 1]),len(df[df['rule3_result'] == 0]))
print(f"下浮率误差1%以内命中率：{round(len(df[df['rule3_result'] < 1])/len(df),2)*100}%，下浮率命中率：{round(len(df[df['rule3_result'] == 0])/len(df),2)*100}%")

df['rule4'] = round(df['rule1']*0.3 + df['rule2']*0.4 + df['rule3']*0.3,2)
df['rule4_result'] = abs(df['下浮率'] - df['rule4'])
print(len(df[df['rule4_result'] < 1]), len(df[df['rule4_result'] == 0]))

