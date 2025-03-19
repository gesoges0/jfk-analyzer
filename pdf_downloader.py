import os
import requests
from bs4 import BeautifulSoup
import time
import re
from urllib.parse import urljoin, urlparse
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("pdf_download.log"), logging.StreamHandler()],
)

# Constants
BASE_URL = "https://www.archives.gov/research/jfk/release-2025"
PDF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdf")
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"


def create_pdf_directory():
    """Create the PDF directory if it doesn't exist."""
    if not os.path.exists(PDF_DIR):
        os.makedirs(PDF_DIR)
        logging.info(f"Created directory: {PDF_DIR}")


def get_page_content(url):
    """Get the HTML content of a page."""
    headers = {"User-Agent": USER_AGENT}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching {url}: {e}")
        return None


def extract_pdf_links(html_content):
    """Extract PDF links from the HTML content."""
    if not html_content:
        return []

    soup = BeautifulSoup(html_content, "html.parser")
    pdf_links = []

    # Find all anchor tags
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        # Check if the link points to a PDF file
        if href.lower().endswith(".pdf"):
            pdf_links.append(href)

    return pdf_links


def extract_pagination_links(html_content, base_url):
    """Extract pagination links from the HTML content."""
    if not html_content:
        return []

    soup = BeautifulSoup(html_content, "html.parser")
    pagination_div = soup.find("div", class_="pagination")

    if not pagination_div:
        return []

    pagination_links = []

    for a_tag in pagination_div.find_all("a", href=True):
        href = a_tag["href"]
        full_url = urljoin(base_url, href)
        if full_url != base_url and full_url not in pagination_links:
            pagination_links.append(full_url)

    return pagination_links


def download_pdf(pdf_url, save_path):
    """Download a PDF file and save it to the specified path."""
    headers = {"User-Agent": USER_AGENT}
    try:
        response = requests.get(pdf_url, headers=headers, stream=True)
        response.raise_for_status()

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading {pdf_url}: {e}")
        return False


def get_filename_from_url(url):
    """Extract a valid filename from a URL."""
    parsed_url = urlparse(url)
    path = parsed_url.path
    filename = os.path.basename(path)

    # Clean the filename to ensure it's valid
    filename = re.sub(r"[^\w\-\.]", "_", filename)

    return filename


def main():
    create_pdf_directory()

    # Start with the base URL
    pages_to_visit = [BASE_URL]
    visited_pages = set()
    downloaded_pdfs = set()

    while pages_to_visit:
        current_page = pages_to_visit.pop(0)
        if current_page in visited_pages:
            continue

        logging.info(f"Visiting page: {current_page}")
        visited_pages.add(current_page)

        html_content = get_page_content(current_page)
        if not html_content:
            continue

        # Extract PDF links
        pdf_links = extract_pdf_links(html_content)
        for pdf_link in pdf_links:
            full_pdf_url = urljoin(current_page, pdf_link)

            if full_pdf_url in downloaded_pdfs:
                continue

            filename = get_filename_from_url(full_pdf_url)
            save_path = os.path.join(PDF_DIR, filename)

            # Skip if already downloaded
            if os.path.exists(save_path):
                logging.info(f"Skipping already downloaded: {filename}")
                downloaded_pdfs.add(full_pdf_url)
                continue

            logging.info(f"Downloading: {filename}")
            if download_pdf(full_pdf_url, save_path):
                logging.info(f"Successfully downloaded: {filename}")
                downloaded_pdfs.add(full_pdf_url)

            # Be nice to the server
            time.sleep(1)

        # Find pagination links
        pagination_links = extract_pagination_links(html_content, current_page)

        # Add new pagination links to pages_to_visit if they haven't been visited
        for link in pagination_links:
            if link not in visited_pages and link not in pages_to_visit:
                pages_to_visit.append(link)

    logging.info("Download process completed.")


if __name__ == "__main__":
    main()
