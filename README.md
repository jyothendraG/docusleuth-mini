DocuSleuth Mini
📄 DocuSleuth Mini is a lightweight, privacy-focused AI-powered document intelligence web app that allows you to upload PDFs, images, and text files, extract text via OCR or native PDF processing, perform semantic search, question answering, and summarization — all running locally without external paid APIs. Perfect for personal document management and AI experimentation.

Features
Upload and process multiple PDFs (including scanned/image PDFs), images (PNG/JPG), and text files.

Extract text per page with OCR fallback using Tesseract.

Semantic search over uploaded documents leveraging SentenceTransformers and ChromaDB vector search.

Keyword highlighting in search results for easy scanning.

View full pages containing queried keywords with an option to download the extracted page text.

Question Answering and Summarization over your own documents using Hugging Face transformers.

Fully open-source and runnable on modest hardware without subscription fees.

Technologies Used
Streamlit — frontend UI framework

pdfplumber — extract text from PDFs

Tesseract OCR — OCR for scanned image pages

Pytesseract — Python wrapper for Tesseract

SentenceTransformers — text embedding models

ChromaDB — vector similarity search database

Hugging Face Transformers — NLP pipelines for QA and summarization

Installation & Setup
Prerequisites
Python 3.9+

Tesseract OCR installed on your system:

Windows: Download and install from UB Mannheim builds.

macOS/Linux: Install via package manager (brew install tesseract or apt-get install tesseract-ocr).

Ensure the tesseract command is accessible via your system PATH.

Clone the repository
bash
git clone https://github.com/yourusername/docusleuth-mini.git
cd docusleuth-mini
Create & activate a Python virtual environment
bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
Install required Python packages
bash
pip install -r requirements.txt
If requirements.txt is not present, install these directly:

bash
pip install streamlit pdfplumber pillow pytesseract sentence-transformers chromadb transformers
Usage
Run the app locally:

bash
streamlit run app.py
Upload your PDFs, images, or text files.

Click Process Documents to extract and index.

Enter your search query or question in the input box.

Browse highlighted full-page results.

Use "Get Direct Text Answer" for question answering.

Use "Summarize Top Results" for summaries.

Download extracted text pages with the button in search results.

Notes
OCR quality depends on your PDFs’ scan quality.

The app runs entirely on your machine, so your documents remain private.

Embedding and search might take some time for large documents depending on your CPU.

For better performance, consider running on a machine with at least 8GB RAM and a modern CPU.

Contact
Created by Jyothendra. Feel free to raise issues or contribute!
