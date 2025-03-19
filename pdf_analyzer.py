import os
import logging
import time
import PyPDF2
import openai
import json
from datetime import datetime
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("pdf_analysis.log"), logging.StreamHandler()],
)

# Constants
PDF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdf")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
MAX_TOKENS = 4000  # Maximum number of tokens for each chunk to send to OpenAI
OVERLAP_TOKENS = 200  # Overlap between chunks to maintain context

# Configuration for analysis
ANALYSIS_PROMPT = """
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

SUMMARY_PROMPT = """
You are a historical researcher compiling a comprehensive report on "Why was President Kennedy assassinated?" 
based on the analysis of declassified documents. Using the following analyses from various documents, synthesize 
a detailed report that:

1. Presents the most credible theories on Kennedy's assassination based on evidence
2. Explores potential motives from various angles (political, geopolitical, personal)
3. Examines key figures involved and their relationships
4. Identifies gaps, contradictions, or suspicious elements in the official narrative
5. Provides a chronological timeline of events leading to the assassination
6. Concludes with the most likely explanation based on the available evidence

Analyses from various documents:
{analyses}

Create a detailed, well-structured report with sections, citations to specific documents when possible, and a 
conclusion that offers your assessment on the most probable explanation for Kennedy's assassination based on 
this documentary evidence.
"""


def create_output_directory():
    """Create the output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logging.info(f"Created directory: {OUTPUT_DIR}")


def extract_text_from_pdf(pdf_path):
    """Extract text content from a PDF file."""
    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
            return text
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {e}")
        return ""


def split_text_into_chunks(text, max_tokens=MAX_TOKENS, overlap=OVERLAP_TOKENS):
    """Split the text into overlapping chunks to maintain context."""
    words = text.split()
    chunks = []

    if len(words) <= max_tokens:
        return [text]

    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + max_tokens])
        chunks.append(chunk)
        i += max_tokens - overlap

    return chunks


def analyze_text_with_openai(text, model="gpt-4"):
    """Send text to OpenAI API for analysis."""
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a historical analyst specializing in the JFK assassination.",
                },
                {"role": "user", "content": ANALYSIS_PROMPT.format(text=text)},
            ],
            max_tokens=1500,
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error analyzing text with OpenAI: {e}")
        return ""


def generate_summary_report(analyses, model="gpt-4"):
    """Generate a comprehensive summary report from all analyses."""
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a historical researcher specializing in the JFK assassination.",
                },
                {"role": "user", "content": SUMMARY_PROMPT.format(analyses=analyses)},
            ],
            max_tokens=4000,
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error generating summary with OpenAI: {e}")
        return "Failed to generate summary report due to an error."


def save_analyses_to_file(analyses):
    """Save individual document analyses to a JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(OUTPUT_DIR, f"jfk_analyses_{timestamp}.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(analyses, f, ensure_ascii=False, indent=4)

    logging.info(f"Saved analyses to {json_path}")
    return json_path


def save_report_to_file(report):
    """Save the final report to a markdown file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(OUTPUT_DIR, f"jfk_assassination_report_{timestamp}.md")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    logging.info(f"Saved report to {report_path}")
    return report_path


def main():
    # Ensure OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        api_key = input("Please enter your OpenAI API key: ").strip()
        os.environ["OPENAI_API_KEY"] = api_key
        openai.api_key = api_key
    else:
        openai.api_key = os.environ.get("OPENAI_API_KEY")

    create_output_directory()

    # Get all PDF files
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        logging.warning(f"No PDF files found in {PDF_DIR}")
        return

    logging.info(f"Found {len(pdf_files)} PDF files")

    # Analyze each PDF file
    all_analyses = {}

    for pdf_file in tqdm(pdf_files, desc="Analyzing PDFs"):
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        logging.info(f"Processing {pdf_file}")

        # Extract text from PDF
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            logging.warning(f"No text content extracted from {pdf_file}")
            continue

        # Split text into chunks
        chunks = split_text_into_chunks(text)
        logging.info(f"Split {pdf_file} into {len(chunks)} chunks")

        # Analyze each chunk
        chunk_analyses = []
        for i, chunk in enumerate(chunks):
            logging.info(f"Analyzing chunk {i+1}/{len(chunks)} of {pdf_file}")
            analysis = analyze_text_with_openai(chunk)
            if analysis:
                chunk_analyses.append(analysis)
            time.sleep(1)  # Rate limiting

        # Combine analyses for this PDF
        all_analyses[pdf_file] = "\n\n".join(chunk_analyses)

    # Save individual analyses
    analyses_path = save_analyses_to_file(all_analyses)

    # Prepare combined analyses for summary
    combined_analyses = ""
    for pdf_file, analysis in all_analyses.items():
        combined_analyses += (
            f"=== ANALYSIS OF DOCUMENT: {pdf_file} ===\n\n{analysis}\n\n"
        )

    # Generate comprehensive report
    logging.info("Generating comprehensive report...")
    final_report = generate_summary_report(combined_analyses)

    # Save final report
    report_path = save_report_to_file(final_report)

    logging.info("Analysis complete!")
    logging.info(f"Individual analyses saved to: {analyses_path}")
    logging.info(f"Final report saved to: {report_path}")


if __name__ == "__main__":
    main()
