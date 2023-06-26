import re
import datetime

def get_date(str_time:str):
    # str_time = 'http://lssggzy.lishui.gov.cn/art/2020/10/16/art_1229661956_138130.html'
    time = re.findall('(\d{4})\/(\d{1,2})\/(\d{1,2})',str_time)[0]
    s = "-".join(time)
    return s
    # datetime.datetime.strptime(s, '%Y-%m-%d')
