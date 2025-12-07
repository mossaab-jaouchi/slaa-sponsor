import streamlit as st
import os

# إعداد الصفحة
st.set_page_config(page_title="SLAA AI Sponsor", page_icon="🛡️")
st.title("🛡️ رفيق التعافي")

# التحقق من المفتاح
if "GROQ_API_KEY" in st.secrets:
    groq_api_key = st.secrets["GROQ_API_KEY"]
else:
    st.error("المفتاح غير موجود! تأكد من إضافته في Secrets.")
    st.stop()

# محاولة استيراد المكتبات مع كاشف أخطاء
try:
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import FastEmbedEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    # في النسخة 0.1.20 هذا الاستيراد يعمل 100%
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError as e:
    st.error(f"خطأ في تثبيت المكتبات (تأكد من requirements.txt): {e}")
    st.stop()

@st.cache_resource
def load_library():
    folder_path = "library" 
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return None
    if not os.listdir(folder_path):
        return "EMPTY"

    try:
        with st.spinner("جاري قراءة الكتب..."):
            loader = PyPDFDirectoryLoader(folder_path)
            docs = loader.load()
            if not docs:
                return "EMPTY"
                
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = splitter.split_documents(docs)
            
            # استخدام FastEmbed لتفادي مشاكل الذاكرة والتوافق
            embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
            
            vectorstore = FAISS.from_documents(splits, embeddings)
            return vectorstore
    except Exception as e:
        st.error(f"حدث خطأ أثناء المعالجة: {e}")
        return None

vectorstore = load_library()

if vectorstore is None:
    st.info("👋 مرحبًا! قم برفع ملفات PDF في مجلد 'library' على GitHub.")
    st.stop()
elif vectorstore == "EMPTY":
    st.warning("⚠️ المكتبة فارغة. الرجاء وضع ملفات PDF.")
    st.stop()

# إعداد الذكاء الاصطناعي
system_prompt = (
    "Answer in Arabic only. You are a strict SLAA sponsor. "
    "Use the context below to guide the user.\n\n"
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")
retriever = vectorstore.as_retriever()
chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("تحدث مع موجهك..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("جاري الاتصال..."):
            response = chain.invoke({"input": user_input})
            st.markdown(response["answer"])
    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
