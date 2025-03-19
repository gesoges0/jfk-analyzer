import os
import logging
import time
import json
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Any

# LangChain imports
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("pdf_analysis_langchain.log"),
        logging.StreamHandler(),
    ],
)

# Constants
PDF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdf")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")

# Configuration for analysis
ANALYSIS_TEMPLATE = """
You are an expert historian and intelligence analyst reviewing declassified JFK assassination documents.
Based on the following document excerpt, please:
1. Identify key information related to potential motives for Kennedy's assassination
2. Note any suspicious activities, connections, or entities mentioned
3. Look for information about Lee Harvey Oswald and his potential connections
4. Extract details about any conspiracy theories supported by the documents
5. Identify any cover-up attempts or inconsistencies in the official narrative

Document excerpt:
{text}

Provide a detailed, objective analysis focusing only on information present in this excerpt that might help answer 
"Why was President Kennedy assassinated?" Do not speculate beyond what's in the text.
"""

SUMMARY_TEMPLATE = """
You are a historical researcher compiling a comprehensive report on "Why was President Kennedy assassinated?" 
based on the analysis of declassified documents. Using the following analyses from various documents, synthesize 
a detailed report that:

1. Presents the most credible theories on Kennedy's assassination based on evidence
2. Explores potential motives from various angles (political, geopolitical, personal)
3. Examines key figures involved and their relationships
4. Identifies gaps, contradictions, or suspicious elements in the official narrative
5. Provides a chronological timeline of events leading to the assassination
6. Concludes with the most likely explanation based on the available evidence

{text}

Create a detailed, well-structured report with sections, citations to specific documents when possible, and a 
conclusion that offers your assessment on the most probable explanation for Kennedy's assassination based on 
this documentary evidence.
"""


def create_output_directory() -> None:
    """Create the output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logging.info(f"Created directory: {OUTPUT_DIR}")


def load_pdf_documents(pdf_path: str) -> List[Document]:
    """Load a PDF file and convert it to LangChain documents."""
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        return documents
    except Exception as e:
        logging.error(f"Error loading PDF {pdf_path}: {e}")
        return []


def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into smaller chunks for processing."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
            length_function=len,
        )
        return text_splitter.split_documents(documents)
    except Exception as e:
        logging.error(f"Error splitting documents: {e}")
        return documents


def analyze_document_chunks(
    chunks: List[Document], model_name: str = "gpt-4"
) -> List[str]:
    """Analyze document chunks using LangChain and LLM."""
    try:
        # Create prompt template
        prompt = ChatPromptTemplate.from_template(ANALYSIS_TEMPLATE)

        # Initialize the LLM
        llm = ChatOpenAI(model_name=model_name, temperature=0.2)

        # Create the chain
        chain = LLMChain(llm=llm, prompt=prompt)

        # Process each chunk
        results = []
        for i, chunk in enumerate(tqdm(chunks, desc="Analyzing chunks")):
            logging.info(f"Analyzing chunk {i+1}/{len(chunks)}")
            response = chain.run(text=chunk.page_content)
            results.append(response)
            time.sleep(1)  # Rate limiting

        return results
    except Exception as e:
        logging.error(f"Error analyzing document chunks: {e}")
        return []


def create_final_report(
    all_analyses: Dict[str, List[str]], model_name: str = "gpt-4"
) -> str:
    """Create a final comprehensive report from all document analyses."""
    try:
        # Combine all analyses into a single text
        combined_analyses = ""
        for pdf_file, analyses in all_analyses.items():
            combined_analyses += f"=== ANALYSIS OF DOCUMENT: {pdf_file} ===\n\n"
            combined_analyses += "\n\n".join(analyses)
            combined_analyses += "\n\n"

        # Create documents for the summarization chain
        docs = [Document(page_content=combined_analyses)]

        # Create the prompt template
        prompt = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)

        # Initialize the LLM
        llm = ChatOpenAI(model_name=model_name, temperature=0.3)

        # Create the summarization chain
        stuff_chain = StuffDocumentsChain(
            llm_chain=LLMChain(llm=llm, prompt=prompt), document_variable_name="text"
        )

        # Run the chain
        final_report = stuff_chain.run(docs)
        return final_report
    except Exception as e:
        logging.error(f"Error creating final report: {e}")
        return "Failed to generate final report due to an error."


def save_analyses_to_file(analyses: Dict[str, List[str]]) -> str:
    """Save individual document analyses to a JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(OUTPUT_DIR, f"jfk_analyses_langchain_{timestamp}.json")

    # Convert analyses to a format suitable for JSON serialization
    serializable_analyses = {}
    for pdf_file, analysis_list in analyses.items():
        serializable_analyses[pdf_file] = analysis_list

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable_analyses, f, ensure_ascii=False, indent=4)

    logging.info(f"Saved analyses to {json_path}")
    return json_path


def save_report_to_file(report: str) -> str:
    """Save the final report to a markdown file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(
        OUTPUT_DIR, f"jfk_assassination_report_langchain_{timestamp}.md"
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    logging.info(f"Saved report to {report_path}")
    return report_path


def main() -> None:
    # Ensure OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        api_key = input("Please enter your OpenAI API key: ").strip()
        os.environ["OPENAI_API_KEY"] = api_key

    create_output_directory()

    # Get all PDF files
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        logging.warning(f"No PDF files found in {PDF_DIR}")
        return

    logging.info(f"Found {len(pdf_files)} PDF files")

    # Analyze each PDF file
    all_analyses = {}

    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        logging.info(f"Processing {pdf_file}")

        # Load PDF as LangChain documents
        documents = load_pdf_documents(pdf_path)
        if not documents:
            logging.warning(f"No content extracted from {pdf_file}")
            continue

        # Split documents into chunks
        chunks = split_documents(documents)
        logging.info(f"Split {pdf_file} into {len(chunks)} chunks")

        # Analyze document chunks
        analyses = analyze_document_chunks(chunks)
        if analyses:
            all_analyses[pdf_file] = analyses

    # Save individual analyses
    analyses_path = save_analyses_to_file(all_analyses)

    # Generate comprehensive report
    logging.info("Generating comprehensive report...")
    final_report = create_final_report(all_analyses)

    # Save final report
    report_path = save_report_to_file(final_report)

    logging.info("Analysis complete!")
    logging.info(f"Individual analyses saved to: {analyses_path}")
    logging.info(f"Final report saved to: {report_path}")


if __name__ == "__main__":
    main()
