
import jieba.analyse

from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pandas as pd
import jieba.posseg as pseg


df = pd.read_excel(r"C:\Users\pc\Desktop\池州市项目预测.xlsx")
df['规模内容'] = df['规模内容'].fillna('')
def preprocess_text(text):
    # 使用正则表达式去除非中文字符
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)

    # 使用结巴分词进行分词
    # words = jieba.cut(text)
    words_with_pos = pseg.cut(text)
    #
    # # 去除停用词
    filtered_words = []
    for word, pos in words_with_pos:
        # 如果词性在过滤列表中，保留该词
        if pos in ['v','n']:
            filtered_words.append(word)
    # stop_words = set(["的", "是", "包括", "等"])  # 你可以根据具体文本内容添加其他停用词
    # filtered_words = [word for word in words if word not in stop_words]

    # 将分词结果拼接为一个字符串，用空格隔开
    # processed_text = " ".join(filtered_words)

    return filtered_words

# 假设你的文本数据保存在一个列表中
text_data = df['规模内容'].values.tolist()
# 预处理文本数据
processed_data = [t for text in text_data for t in preprocess_text(text)]
# processed_data = [preprocess_text(text) for text in text_data]
# 创建TfidfVectorizer对象
tfidf_vectorizer = TfidfVectorizer()

# 将预处理后的文本数据转换为TF-IDF特征向量
tfidf_features = tfidf_vectorizer.fit_transform(processed_data)

# 可以获取词汇表
vocabulary = tfidf_vectorizer.get_feature_names_out()
print(vocabulary)
print(sum(pd.DataFrame(tfidf_features.toarray(),columns=vocabulary).iloc[2].values.tolist()))

print(pd.DataFrame(tfidf_features.toarray(),columns=vocabulary))

