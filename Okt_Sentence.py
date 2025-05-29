from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from konlpy.tag import Okt
import pandas as pd

# 형태소 분석기
okt = Okt()

def tokenize(text):
    return okt.nouns(text)

df = pd.read_csv('./Scripts/classification_model_4.csv', names=['sentence','model'])
corpus = df['sentence']

# 형태소 분석 후 띄어쓰기 기준 재조합
tokenized_corpus = [" ".join(tokenize(doc)) for doc in corpus]

# TF-IDF 벡터라이저 학습
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(tokenized_corpus)

def find_top_n_similar_sentences(input_sentence, top_n=10):
    tokenized_input = " ".join(tokenize(input_sentence))
    input_vec = vectorizer.transform([tokenized_input])
    similarities = cosine_similarity(input_vec, tfidf_matrix).flatten()
    
    # 상위 top_n 인덱스와 점수
    top_n_idx = similarities.argsort()[-top_n:][::-1]
    top_n_scores = similarities[top_n_idx]
    
    return list(zip(top_n_idx, top_n_scores))

# 테스트
new_sentence = "야구 승률ㅇㄹ에 따른 순위 예측"
top_similar = find_top_n_similar_sentences(new_sentence)

sentences = []

for idx, score in top_similar:
    sentences.append(df['model'][idx])


    from collections import Counter

def most_common_sentence(sentences):
    """
    sentences: 문장 리스트
    return: 가장 많이 나온 문장과 그 빈도
    """
    count = Counter(sentences)
    most_common = count.most_common(1)
    if most_common:
        sentence, freq = most_common[0]
        return sentence, freq
    else:
        return None, 0

# 예시


result, freq = most_common_sentence(sentences)
print('필요 모델: '+ result)
