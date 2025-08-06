#WELCOME TO DOCUSLEUTH MINI!
import streamlit as st
import pdfplumber
from PIL import Image
import pytesseract
import os
import re

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from transformers import pipeline
# --- Page config ---
st.set_page_config(page_title="DocuSleuth Mini", layout="centered")
st.title("📄 DocuSleuth Mini")
st.markdown("""
Welcome to **DocuSleuth Mini**!  
Upload your documents (PDF, images, or plain text), then search or ask questions.  
""")

# --- File uploader ---
uploaded_files = st.file_uploader(
    "Upload documents (PDF, PNG, JPG, TXT):",
    type=['pdf', 'png', 'jpg', 'jpeg', 'txt'],
    accept_multiple_files=True
)

query = st.text_input("🔎 Enter your search or question:")

process_button = st.button("Process Documents")

#-----------------------------------------------------------------------------#
# Helper function: Extract text per page for PDFs, or wrap text for images/txt #
#-----------------------------------------------------------------------------#

def extract_text_from_pdf_pages(pdf_file):
    pages_text = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text or not text.strip():
                # If no text, do OCR on the image instead
                page_img = page.to_image(resolution=300).original
                text = pytesseract.image_to_string(page_img)
            pages_text.append(text or "")
    return pages_text


def extract_text_from_image(image_file):
    image = Image.open(image_file)
    return pytesseract.image_to_string(image)

def extract_text_from_txt(txt_file):
    content = txt_file.read()
    try:
        return content.decode('utf-8')
    except:
        return content.decode('latin-1', errors='ignore')

def get_text_from_file(uploaded_file):
    file_type = uploaded_file.type
    if file_type == "application/pdf":
        return extract_text_from_pdf_pages(uploaded_file)
    elif file_type in ["image/png", "image/jpeg"]:
        return [extract_text_from_image(uploaded_file)]
    elif file_type == "text/plain":
        return [extract_text_from_txt(uploaded_file)]
    else:
        return [""]

#-------------------------
# Initialize embedding model
#-------------------------
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

#-------------------------
# Initialize or load ChromaDB persistent store
db_dir = os.path.join(os.getcwd(), "chromadb_storage")
chroma_client = chromadb.PersistentClient(path=db_dir, settings=Settings(anonymized_telemetry=False))
collection = chroma_client.get_or_create_collection('documents')

#-------------------------
# Text chunking (here: one full page = one chunk, so no splitting)
#-------------------------

def embed_and_store_docs(all_docs_pages):
    # Clear existing collection to avoid duplicates on re-process:
    existing_ids = collection.get()['ids']

    if existing_ids:
        collection.delete(ids=existing_ids)

    ids, docs, metas, embeddings = [], [], [], []
    for doc in all_docs_pages:
        name, pages = doc['name'], doc['pages']
        for i, page_text in enumerate(pages):
            if not page_text.strip():
                continue
            emb = embed_model.encode(page_text)
            chunk_id = f"{name}_page_{i}"
            ids.append(chunk_id)
            docs.append(page_text)
            metas.append({'source_file': name, 'page': i})
            embeddings.append(emb.tolist())
    collection.add(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)
    return len(ids)

#-------------------------
# Semantic search function
#-------------------------
def semantic_search(query, top_k=5):
    query_emb = embed_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        include=['documents', 'metadatas']
    )
    return results['documents'][0], results['metadatas'][0]

#-------------------------
# Load QA and Summarization pipelines (small, efficient models)
#-------------------------
qa_pipe = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
summarizer = pipeline("summarization", model="t5-small")

def answer_question(query, contexts):
    for context in contexts:
        try:
            answer = qa_pipe(question=query, context=context)['answer']
            if answer:
                return answer
        except Exception:
            pass
    return "No clear answer found."

def summarize_text(contexts):
    summaries = []
    for context in contexts:
        try:
            summary = summarizer(context, max_length=50, min_length=15, do_sample=False)[0]['summary_text']
            summaries.append(summary)
        except Exception:
            pass
    return summaries

#-------------------------
# Keyword highlighting for display
#-------------------------
def highlight_keywords(text, keywords):
    if not keywords:
        return text
    pattern = re.compile("|".join(re.escape(k) for k in keywords), re.IGNORECASE)
    return pattern.sub(lambda m: f"**:orange[{m.group(0)}]**", text)

#==========================
# Main App Logic
#==========================

if process_button:
    if not uploaded_files:
        st.error("Please upload at least one document.")
    else:
        all_docs_pages = []
        for uploaded_file in uploaded_files:
            st.info(f"Processing: {uploaded_file.name}")
            pages = get_text_from_file(uploaded_file)
            if pages:
                total_chars = sum(len(p) for p in pages)
                st.success(f"✅ Extracted {total_chars} characters from {len(pages)} pages of `{uploaded_file.name}`")
                preview = (pages[0][:600] + "...") if len(pages[0]) > 600 else pages[0]
                with st.expander(f"See extract first page from {uploaded_file.name}"):
                    st.write(preview)
                all_docs_pages.append({'name': uploaded_file.name, 'pages': pages})
            else:
                st.warning(f"⚠️ No text extracted from `{uploaded_file.name}`. Is it scanned or empty?")

        st.session_state['all_docs_pages'] = all_docs_pages

        # Embed and index pages
        if 'all_docs_pages' in st.session_state and st.session_state['all_docs_pages']:
            st.info("Embedding and indexing documents...")
            count_indexed = embed_and_store_docs(st.session_state['all_docs_pages'])
            st.success(f"Indexed {count_indexed} pages for semantic search.")

#-------------------------
# Query answering/search UI
#-------------------------
if query and 'all_docs_pages' in st.session_state:
    with st.spinner("Searching..."):
        matching_pages, metadatas = semantic_search(query)
        st.subheader("Pages Containing Your Query:")

        keywords = query.split()

        max_results = 5
        for i, (page_text, meta) in enumerate(zip(matching_pages[:max_results], metadatas[:max_results])):
            highlighted_page = highlight_keywords(page_text, keywords)
            st.markdown(f"### 📄 Document: `{meta['source_file']}` - Page {int(meta['page']) + 1}")
            st.markdown(highlighted_page, unsafe_allow_html=False)
            st.download_button(
                "Extract This Page",
                data=page_text,
                file_name=f"{meta['source_file']}_page_{int(meta['page']) + 1}.txt"
            )
            st.markdown("---")

        # Show QA and Summarize buttons below results
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Get Direct Text Answer"):
                answer = answer_question(query, matching_pages[:max_results])
                st.info(f"**Answer:** {answer}")

        with col2:
            if st.button("Summarize Top Results"):
                summaries = summarize_text(matching_pages[:max_results])
                st.info("**Summary of top results:**")
                for idx, s in enumerate(summaries):
                    st.write(f"Summary [{idx + 1}]: {s}")
