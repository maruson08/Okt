from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

# 형태소 분석기 초기화
okt = Okt()

# 형태소 분석 + 명사 추출 함수
def tokenize(text):
    return okt.nouns(text)

# 기존 문장 코퍼스
df = pd.read_csv('./Scripts/classification_model_3.csv', names=['sentence', 'model'])
corpus = df['sentence'].tolist()

# 형태소 분석 후 토큰을 다시 띄어쓰기로 연결
tokenized_corpus = [" ".join(tokenize(doc)) for doc in corpus]

# TF-IDF 벡터라이저 학습
vectorizer = TfidfVectorizer()
vectorizer.fit(tokenized_corpus)

# 새로운 문장 입력 받아 키워드 추출 함수
def extract_keywords_from_sentence(sentence, top_k=3):
    # 형태소 분석으로 명사만 추출
    tokens = tokenize(sentence)
    # 띄어쓰기 기준으로 다시 연결
    tokenized_sentence = " ".join(tokens)
    
    # 벡터라이저 변환 (TF-IDF 점수)
    tfidf_vector = vectorizer.transform([tokenized_sentence])
    
    # 단어 리스트
    feature_names = vectorizer.get_feature_names_out()
    
    # TF-IDF 점수 배열로 변환
    scores = tfidf_vector.toarray()[0]
    
    # 단어별 점수 딕셔너리 생성
    word_score = {word: scores[idx] for idx, word in enumerate(feature_names) if scores[idx] > 0}
    
    # 점수 높은 순으로 정렬 후 top_k개 단어 추출
    keywords = sorted(word_score.items(), key=lambda x: x[1], reverse=True)[2:top_k+2]
    
    return [word for word, score in keywords]

# 테스트: 새로운 문장 입력
new_sentence = "간수치 데이터를 보고 규칙을 예측하는 모델을 만들고 싶어"
keywords = extract_keywords_from_sentence(new_sentence)
print("추출된 키워드:", keywords)

keywords = []

for i in corpus:
    keywords.append(extract_keywords_from_sentence(i))

df['keywords'] = keywords

print(df.head())