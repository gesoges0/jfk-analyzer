# JFK Assassination Analysis Project

This project downloads and analyzes declassified documents related to the JFK assassination from the National Archives, using AI to generate insights and a comprehensive report on "Why was President Kennedy assassinated?"

## Components

- **PDF Downloader**: Fetches all PDF documents from the National Archives JFK Release 2025 collection
- **PDF Analyzer**: Processes the downloaded documents with the OpenAI API to extract insights and generate a report
- **PDF Analyzer (LangChain)**: An alternative analyzer using the LangChain framework for more modular and extensible document processing

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## Usage

1. Download the PDFs:
   ```bash
   python pdf_downloader.py
   ```

2. Analyze the PDFs and generate a report:
   
   Using direct OpenAI API:
   ```bash
   python pdf_analyzer.py
   ```
   
   Or using LangChain (recommended):
   ```bash
   python pdf_analyzer_langchain.py
   ```

The analysis will create:
- A JSON file with analyses of individual documents
- A comprehensive markdown report on Kennedy's assassination

## Output

The generated reports will be saved in the `reports` directory:
- Individual document analyses: `jfk_analyses_YYYYMMDD_HHMMSS.json`
- Final report: `jfk_assassination_report_YYYYMMDD_HHMMSS.md`

## Benefits of the LangChain Version

The LangChain-based analyzer offers several advantages:
- More modular architecture for easier maintenance and extension
- Built-in document loading and splitting capabilities
- Better handling of document metadata
- Simplified prompt management
- Easier to extend with additional processing steps in the future

## Note

This project will consume OpenAI API credits as it processes documents. The amount depends on the number and size of PDFs analyzed.
