from function import *
from streamlit_option_menu import option_menu
import joblib

st.header("Klasifikasi Artikel Berita Berdasarkan Hasil Summarization", divider='rainbow')

text = st.text_area("Masukkan Artikel Berita")
col1, col2 = st.columns(2)
with col1:
    k = st.number_input("Jumlah Kalimat Yang Di Ambil", 1)

with col2:
    threshold = st.number_input("Threshold")

button = st.button("Submit")

if "summary" not in st.session_state:
    st.session_state.summary = []
    st.session_state.graph_klasifikasi = []
    st.session_state.svmNonSUM = []
    st.session_state.svmSUM = []

if button:
    text_clean = cleaning(text)
    tokenizing = tokenizer(text_clean)
    summary, G = summarization(x=tokenizing, k=k, threshold=threshold)
    
    st.session_state.graph_klasifikasi = G
    st.session_state.summary = summary
    
    # Predict Model Tanpa Ringkasan
    vectorizer_nonSUM = joblib.load("resources/vectorizer_nonSummary.pkl")
    model_nonSum = joblib.load("resources/modelSVM_NonSummary.pkl")
    new_text_matrics_nonSum = vectorizer_nonSUM.transform([summary]).toarray()
    prediction_nonSum = model_nonSum.predict(new_text_matrics_nonSum)
    st.session_state.svmNonSUM = prediction_nonSum[0]
    
    # Predict Model Dengan Ringkasan
    vectorizer_sum = joblib.load("resources/vectorizer_summary.pkl")
    model_sum = joblib.load("resources/modelSVM_WithSummary.pkl")
    new_text_matrics_sum = vectorizer_sum.transform([summary]).toarray()
    prediction_sum = model_sum.predict(new_text_matrics_sum)
    st.session_state.svmSUM = prediction_sum[0]


selected = option_menu(
  menu_title="",
  options=["Summary", "Klasifikasi", "Graph Kalimat"],
  icons=["data", "Process", "model", "implemen", "Test", "sa"],
  orientation="horizontal"
  )

if selected == "Summary":
  if st.session_state.summary:
    sentence = st.session_state.tokenSentence
    score_text_rank = st.session_state.score_text_rank
    st.write("**Hasil Summarization:**")
    st.write("_"+ st.session_state.summary + "_")
    st.write("Nilai score Text Rank :")
    for i, cls in enumerate(score_text_rank):
      st.write(f"index {i} score Text Rank : {score_text_rank[cls]} -> Kalimat : {sentence[i]}")
  
  
elif selected == "Klasifikasi":
  if st.session_state.summary:
    #   st.caption("Klasifikasi Berdasarkan Hasil Summarization (Naive Bayes)")
      new_text = st.session_state.summary
      svm_nonSummary, svm_Summary = st.tabs(["Model SVM (Ringkasan)", "Model SVM (Tanpa Ringkasan)"])
      
      with svm_nonSummary:
        #   if st.session_state.svmNonSUM:
        # vectorizer = joblib.load("resources/vectorizer_nonSummary.pkl")
        # model = joblib.load("resources/modelSVM_NonSummary.pkl")
        # new_text_matrics = vectorizer.transform([new_text]).toarray()
        # prediction = model.predict(new_text_matrics)
        st.write("Prediction Category : ", st.session_state.svmNonSUM)
        
      with svm_Summary:
        # vectorizer = joblib.load("resources/vectorizer_summary.pkl")
        # model = joblib.load("resources/modelSVM_WithSummary.pkl")
        # new_text_matrics = vectorizer.transform([new_text]).toarray()
        # prediction = model.predict(new_text_matrics)
        st.write("Prediction Category : ", st.session_state.svmSUM)
     

elif selected == "Graph Kalimat":
  col1, col2 = st.columns(2)
  with col1:
    x_canvas = st.number_input('Lebar Canvas', 10)
  
  with col2:
    y_canvas = st.number_input('Panjang Canvas', 10)
    
  node_size = st.number_input('Node Size', 400)
  
  if st.session_state.graph_klasifikasi != []:
    plot_graph(st.session_state.graph_klasifikasi, (x_canvas, y_canvas), node_size)