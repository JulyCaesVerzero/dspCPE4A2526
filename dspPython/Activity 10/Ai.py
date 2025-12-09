import google.generativeai as genai
import fitz
import os
import tkinter as tk
from tkinter import filedialog
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import warnings
warnings.filterwarnings('ignore')

# Configure Gemini API
genai.configure(api_key="AIzaSyBLcKpMXm-86zVCzzwsTwMzlsJTt-TP_HQ")

def choose_pdf_file():
    """Open file dialog to select PDF file"""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select a PDF file",
        filetypes=[("PDF files", "*.pdf")],
    )
    return file_path

def extract_pdf_text(file_path):
    """Extract text from PDF using PyMuPDF (fitz)"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with fitz.open(file_path) as pdf:
            text = ""
            for page_num, page in enumerate(pdf):
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.get_text("text")
        
        if not text.strip():
            raise ValueError("The PDF file has no readable text (it might be scanned or image-only).")
        
        print(f"✓ Successfully extracted {len(pdf)} pages from '{os.path.basename(file_path)}'")
        return text.strip()

    except Exception as e:
        print(f"✗ Error reading PDF file: {e}")
        return ""

def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    """Split text into manageable chunks for processing"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    print(f"✓ Split text into {len(chunks)} chunks")
    return chunks

def create_vector_store(text_chunks, pdf_name):
    """Create FAISS vector store from text chunks"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key="AIzaSyBLcKpMXm-86zVCzzwsTwMzlsJTt-TP_HQ"
        )
        
        # Create vector store
        vector_store = FAISS.from_texts(
            texts=text_chunks,
            embedding=embeddings
        )
        
        # Save vector store locally
        vector_store.save_local(f"faiss_index_{pdf_name}")
        print(f"✓ Created and saved vector store for '{pdf_name}'")
        return vector_store
        
    except Exception as e:
        print(f"✗ Error creating vector store: {e}")
        return None

def setup_qa_chain(vector_store):
    """Setup Retrieval QA chain with Gemini"""
    try:
        # Initialize Gemini LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3,
            google_api_key="AIzaSyBLcKpMXm-86zVCzzwsTwMzlsJTt-TP_HQ"
        )
        
        # Custom prompt template
        prompt_template = """You are a helpful assistant that answers questions based on the provided context.
        
        Context: {context}
        
        Question: {question}
        
        Instructions:
        1. Answer based ONLY on the context provided above
        2. If the context doesn't contain relevant information, say "I cannot find this information in the document"
        3. Be concise and accurate
        4. If relevant, include page numbers from the context
        
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(
                search_kwargs={"k": 4}  # Retrieve top 4 relevant chunks
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        return qa_chain
        
    except Exception as e:
        print(f"✗ Error setting up QA chain: {e}")
        return None

def interactive_qa_session(qa_chain, pdf_name):
    """Interactive question answering session"""
    print("\n" + "="*60)
    print(f"📚 Q&A Session for: {pdf_name}")
    print("="*60)
    print("\nType 'exit', 'quit', or 'q' to end the session")
    print("Type 'summary' for a document summary")
    print("Type 'sources' to see source information for the last answer\n")
    
    while True:
        question = input("\n❓ Ask a question: ").strip()
        
        if question.lower() in ["exit", "quit", "q"]:
            print("\n👋 Goodbye!")
            break
            
        if question.lower() == "summary":
            get_document_summary(qa_chain)
            continue
            
        if question.lower() == "sources":
            display_source_info()
            continue
            
        if not question:
            continue
            
        try:
            # Get answer from QA chain
            result = qa_chain({"query": question})
            
            print("\n" + "="*60)
            print("💡 Answer:")
            print("="*60)
            print(result["result"])
            
            # Store source documents for reference
            global last_source_docs
            last_source_docs = result.get("source_documents", [])
            
            if last_source_docs:
                print(f"\n📚 Retrieved {len(last_source_docs)} relevant text chunks")
            
        except Exception as e:
            print(f"\n⚠️ Error getting answer: {e}")

def get_document_summary(qa_chain):
    """Get a summary of the document"""
    try:
        summary_prompt = """Based on the document, provide a comprehensive summary including:
        1. Main topics covered
        2. Key points
        3. Overall purpose or conclusion
        4. Any important dates, names, or figures mentioned
        
        Keep the summary concise but informative."""
        
        result = qa_chain({"query": summary_prompt})
        print("\n" + "="*60)
        print("📋 Document Summary:")
        print("="*60)
        print(result["result"])
        
    except Exception as e:
        print(f"✗ Error generating summary: {e}")

def display_source_info():
    """Display source information for the last answer"""
    global last_source_docs
    
    if not 'last_source_docs' in globals() or not last_source_docs:
        print("\n⚠️ No source information available. Ask a question first.")
        return
    
    print("\n" + "="*60)
    print("📄 Source Information:")
    print("="*60)
    
    for i, doc in enumerate(last_source_docs, 1):
        print(f"\n📑 Source {i}:")
        content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        print(f"Content preview: {content_preview}")
        
        # Extract metadata if available
        if hasattr(doc, 'metadata'):
            metadata = doc.metadata
            print(f"Metadata: {metadata}")
        print("-" * 40)

def main():
    """Main function"""
    print("="*60)
    print("📄 PDF Document Q&A Assistant with LangChain")
    print("="*60)
    
    # Step 1: Select PDF file
    print("\n📂 Please select a PDF file to analyze...")
    pdf_path = choose_pdf_file()
    
    if not pdf_path:
        print("❌ No file selected. Exiting.")
        return
    
    pdf_name = os.path.basename(pdf_path).replace('.pdf', '')
    print(f"\n📖 Selected: {pdf_name}")
    
    # Step 2: Extract text from PDF
    print("\n📄 Extracting text from PDF...")
    text = extract_pdf_text(pdf_path)
    
    if not text:
        print("❌ Failed to extract text. Exiting.")
        return
    
    # Step 3: Split text into chunks
    print("\n✂️ Splitting text into chunks...")
    text_chunks = split_text_into_chunks(text)
    
    # Step 4: Create vector store
    print("\n🔧 Creating vector store (this may take a moment)...")
    vector_store = create_vector_store(text_chunks, pdf_name)
    
    if not vector_store:
        print("❌ Failed to create vector store. Exiting.")
        return
    
    # Step 5: Setup QA chain
    print("\n⚙️ Setting up question answering chain...")
    qa_chain = setup_qa_chain(vector_store)
    
    if not qa_chain:
        print("❌ Failed to setup QA chain. Exiting.")
        return
    
    # Step 6: Start interactive session
    interactive_qa_session(qa_chain, pdf_name)

if __name__ == "__main__":
    # Global variable to store last source documents
    last_source_docs = []
    
    # Install required packages if not already installed
    required_packages = [
        "google-generativeai",
        "PyMuPDF",
        "langchain",
        "langchain-google-genai",
        "langchain-community",
        "faiss-cpu"
    ]
    
    print("Checking dependencies...")
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            print(f"⚠️ Please install {package}: pip install {package}")
    
    main()