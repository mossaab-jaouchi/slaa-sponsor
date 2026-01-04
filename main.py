import streamlit as st
import os

# 1. تقليل استهلاك الذاكرة بإجبار النظام على خيط واحد
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(page_title="SLAA AI Sponsor", page_icon="🛡️")
st.title("🛡️ رفيق التعافي")

# استيراد آمن
try:
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import FastEmbedEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    # هذا السطر سيعمل 100% مع النسخة 0.1.20
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError as e:
    st.error(f"خطأ في المكتبات: {e}")
    st.stop()

# المفتاح
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    st.warning("⚠️ المفتاح غير موجود.")
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
        with st.spinner("جاري قراءة الكتب (وضع توفير الذاكرة)..."):
            loader = PyPDFDirectoryLoader(folder_path)
            docs = loader.load()
            
            if not docs:
                return "EMPTY"

            # تقليل حجم القطع لتخفيف العبء على الرام
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            splits = splitter.split_documents(docs)
            
            # إعدادات FastEmbed الخفيفة
            embeddings = FastEmbedEmbeddings(
                model_name="BAAI/bge-base-en-v1.5",
                threads=1 # مهم جداً لمنع الانهيار
            )
            
            vectorstore = FAISS.from_documents(splits, embeddings)
            return vectorstore
            
    except Exception as e:
        st.error(f"حدث خطأ أثناء المعالجة: {e}")
        return None

vectorstore = load_library()

if vectorstore is None:
    st.info("الرجاء رفع ملفات PDF.")
    st.stop()
elif vectorstore == "EMPTY":
    st.warning("⚠️ المكتبة فارغة.")
    st.stop()

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