from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

okt = Okt()

def tokenize(text):
    return okt.nouns(text)

df = pd.read_csv('./Scripts/classification_model_3.csv', names=['sentence', 'model'])
corpus = df['sentence'].tolist()

tokenized_corpus = [" ".join(tokenize(doc)) for doc in corpus]

vectorizer = TfidfVectorizer()
vectorizer.fit(tokenized_corpus)

def extract_keywords_from_sentence(sentence, top_k=3):
    tokens = tokenize(sentence)
    tokenized_sentence = " ".join(tokens)
    
    tfidf_vector = vectorizer.transform([tokenized_sentence])
    
    feature_names = vectorizer.get_feature_names_out()
    
    scores = tfidf_vector.toarray()[0]
    
    word_score = {word: scores[idx] for idx, word in enumerate(feature_names) if scores[idx] > 0}
    
    keywords = sorted(word_score.items(), key=lambda x: x[1], reverse=True)[2:top_k+2]
    
    return [word for word, score in keywords]

new_sentence = "간수치 데이터를 보고 규칙을 예측하는 모델을 만들고 싶어"
keywords = extract_keywords_from_sentence(new_sentence)
print("추출된 키워드:", keywords)

keywords = []

for i in corpus:
    keywords.append(extract_keywords_from_sentence(i))

df['keywords'] = keywords

print(df.head())
