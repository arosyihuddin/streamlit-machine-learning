from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import streamlit as st
import networkx as nx
import pandas as pd
import nltk
import re

nltk.download('stopwords')
nltk.download('punkt')

def cleaning(text):
  text = re.sub(r'[^\w\s.,()/\'\"]', '', text).strip()
  return text

def tokenizer(text):
  text = text.lower()
  return sent_tokenize(text)

def tfidf_transform(x):
  vectorizer = TfidfVectorizer()
  tfidf_matrix = vectorizer.fit_transform(x)
  return tfidf_matrix

def cosine_sim(tfidf_matrix):
  return cosine_similarity(tfidf_matrix)

def bulid_graph(x, cos_sim, threshold=0.11):
  G = nx.Graph()

  for i in range(len(x)):
    for j in range(i+1, len(x)):
      sim = cos_sim[i][j]
      if sim > threshold:
        G.add_edge(i, j, weight=sim)
  return G


def plot_graph(G, figsize=(35, 30), node_size=700, node_color='skyblue'):
  # Menggambar graf dengan canvas yang diperbesar
  pos = nx.spring_layout(G)  # Menentukan posisi simpul
  labels = nx.get_edge_attributes(G, 'weight')

  # Menentukan ukuran canvas
  fig = plt.figure(figsize=figsize)

  # Menggambar graf dengan ukuran canvas yang diperbesar
  nx.draw(G, pos, with_labels=True, node_size=node_size, node_color=node_color)
  nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='red')
  st.pyplot(fig)
  
def summarization(x, k = 4, index=1, threshold=0.11, show_score=False, detail_score=False):
    # Mengirimkan ke Session token sentence
    st.session_state.tokenSentence = x
    
    # Representasi kata dengan TFIDF
    tfidf_matrics = tfidf_transform(x)

    # Menghitung Kemiripan Kalimat dengan Cosine Similirity
    cos_sim = cosine_sim(tfidf_matrics)

    # Membuat Graph
    G = bulid_graph(x, cos_sim, threshold)

    # Perhitungan Score Text Rank
    score = nx.pagerank(G)

    # Sorted Score berdasarkan nilai terteinggi
    score = dict(sorted(score.items(), key=lambda item : item[1], reverse=True))
    
    # Mengirimkan Score Text Rank ke session
    st.session_state.score_text_rank = score

    summary_sentences = []
    for i, centr in enumerate(score.items()):
        if i < k:
            summary_sentences.append(x[centr[0]])

    if show_score:
        print(f"Nilai Text Rank Dokumen Ke - {index} : {score}")
        if detail_score:
            for i, nilai in score.items():
                print(f"Score Kalimat Index ke-{i} : {nilai}")

    return (' '.join(summary_sentences), G)


# ================ Preprocessing Hasil Summary =============== 

def cleaning_text(text):
  text = re.sub(r'[^a-zA-Z\s]', '', text).strip()
  return text

def tokenizer_text(text):
  text = text.lower()
  return word_tokenize(text)


corpus = stopwords.words('indonesian')

def stopwordText(words):
 return [word for word in words if word not in corpus]