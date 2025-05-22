import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urldefrag
import time
import random
import os
from collections import deque

# How deep the crawler can go from the starting page
maxDepth = 1

# How many characters to show per page
charsPrinted = 100000

# List of starting URLs
start_urls = [ "https://www.retsd.mb.ca/rec/"]

# Name of the folder to save the extracted text
output_folder = "extractedText"

# Adds a random delay between 0.2–0.5 seconds to avoid overwhelming servers
def crawl_with_delay():
    time.sleep(random.uniform(0.2, 0.5))

# Fetches the webpage content and parses it into a BeautifulSoup object
def getSoup(url):
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return BeautifulSoup(response.text, 'html.parser')
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

# Extracts only visible, readable text from a page
def textGetter(soup):
    for tag in soup(['script', 'style', 'noscript']):
        tag.decompose()
    return soup.get_text(separator=' ', strip=True)

# Cleans a URL by removing any fragment like "#section"
def clean_url(url):
    return urldefrag(url)[0]

# Function to save the indexed data to a text file (appending instead of overwriting)
def save_index_to_file(data, output_dir="extractedText", filename="website_index.txt"):
    try:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "a", encoding="utf-8") as f:  # open in append mode
            for item in data:
                f.write(f"URL: {item[0]}\n")
                f.write(f"Text:\n{item[1]}\n")
                f.write("-" * 80 + "\n")
        print(f"Indexed data appended to {filepath}")
    except Exception as e:
        print(f"Error saving indexed data: {e}")

# The main BFS web crawler function for one URL
def bfsCrawler(start_url):
    visited = set()
    indexed_data = []

    queue = deque()
    queue.append((start_url, 0))
    visited.add(start_url)

    base_domain = urlparse(start_url).netloc

    while queue:
        url, depth = queue.popleft()
        print(f"\nVisiting (depth {depth}): {url}")
        crawl_with_delay()

        soup = getSoup(url)
        if soup:
            text = textGetter(soup)
            if text:
                print(f"Text from {url}:\n{text[:charsPrinted]}...\n")
                indexed_data.append([url, text])

            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = clean_url(urljoin(url, href))
                parsed = urlparse(full_url)

                if parsed.scheme in ["http", "https"] and parsed.netloc == base_domain and depth + 1 <= maxDepth:
                    if full_url not in visited:
                        queue.append((full_url, depth + 1))
                        visited.add(full_url)

    save_index_to_file(indexed_data)

# Loop through all URLs in the list and run the crawler
for url in start_urls:
    bfsCrawler(url)
