import streamlit as st
import pickle
import nltk
nltk.data.path.append('./nltk_data')
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# -------------------- NLTK Setup --------------------
nltk.download('punkt')
nltk.download('stopwords')

# -------------------- Load Model & Vectorizer --------------------
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

# -------------------- Preprocessing Function --------------------
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    filtered = [ps.stem(word) for word in tokens if word.isalnum() and word not in stop_words and word not in string.punctuation]
    return " ".join(filtered)

# -------------------- Streamlit Page Setup --------------------
st.set_page_config(
    page_title="📩 SMS Spam Classifier",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------- UI Layout --------------------
left_col, right_col = st.columns([1, 2])

# ---------- Left Column: Image and Info ----------
with left_col:
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSCJpa_nWAxGVMDAtDGg1fmCxMwDPyIimyp9w&s", width=250)
    st.markdown("### 🔐 Your messages stay private.")
    st.markdown("#### 🤖 Powered by Machine Learning")
    st.markdown("✅ Real-time spam detection")

# ---------- Right Column: Spam Classifier ----------
with right_col:
    st.title("📩 Email/SMS Spam Classifier")
    st.markdown("🚀 Paste a message below to check if it's spam or not.")

    input_sms = st.text_area("✉️ Enter your message", height=150, placeholder="e.g. Congratulations! You've won a ₹10,000 Amazon voucher...")

    if st.button("🔍 Predict"):
        if input_sms.strip() == "":
            st.warning("⚠️ Please enter a message before predicting.")
        else:
            transformed_sms = transform_text(input_sms)
            vector_input = tfidf.transform([transformed_sms])
            result = model.predict(vector_input)[0]

            if result == 1:
                st.error("❌ This is a **SPAM** message.")
            else:
                st.success("✅ This is a **HAM** (not spam) message.")

# -------------------- Footer --------------------
st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>Made with ❤️ using <a href='https://streamlit.io' target='_blank'>Streamlit</a></p>",
    unsafe_allow_html=True
)
