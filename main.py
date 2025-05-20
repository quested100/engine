import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urldefrag
import time
import random
import os
from collections import deque

# Set to track all visited URLs to avoid duplicates
visited = set()

# How deep the crawler can go from the starting page
maxDepth = 1

# How many characters to show per page
charsPrinted = 1000

# The first URL to start crawling from
start_url = "https://www.apple.com/"

# List to store URL and extracted text
indexed_data = []

# Name of the folder to save the extracted text
output_folder = "extractedText"

# Adds a random delay between 1–2 seconds to avoid overwhelming servers
def crawl_with_delay():
    time.sleep(random.randint(1, 2))

# Fetches the webpage content and parses it into a BeautifulSoup object
def getSoup(url):
    try:
        response = requests.get(url, timeout=30)  # tries to contact the server for up to 30 seconds
        response.raise_for_status()  # raises an error for bad responses (like 404 or 500)
        return BeautifulSoup(response.text, 'html.parser')  # parses HTML content
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

# Extracts only visible, readable text from a page (removes scripts and styles)
def textGetter(soup):
    for tag in soup(['script', 'style', 'noscript']):  # remove code and formatting tags
        tag.decompose()
    return soup.get_text(separator=' ', strip=True)  # clean up whitespace and return text

# Cleans a URL by removing any fragment like "#section"
def clean_url(url):
    return urldefrag(url)[0]

# Function to save the indexed data to a text file
def save_index_to_file(data, output_dir="extractedText", filename="website_index.txt"):
    try:
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            for item in data:
                f.write(f"URL: {item[0]}\n")
                f.write(f"Text:\n{item[1]}\n")
                f.write("-" * 80 + "\n")  # Separator for readability
        print(f"Indexed data saved to {filepath}")
    except Exception as e:
        print(f"Error saving indexed data: {e}")

# The main BFS web crawler function
def bfsCrawler(start_url):
    queue = deque()  # deque used for FIFO behavior (first-in-first-out)
    queue.append((start_url, 0))  # store tuple of (url, depth)
    visited.add(start_url)  # Mark the starting URL as visited

    base_domain = urlparse(start_url).netloc  # get domain of start URL (e.g., 'en.wikipedia.org')

    while queue:
        url, depth = queue.popleft()  # get next URL and its depth from the queue

        print(f"\nVisiting (depth {depth}): {url}")  # log the visit
        crawl_with_delay()  # Add delay *before* processing the URL

        soup = getSoup(url)  # download and parse the page
        if soup:  # Only proceed if soup is not None
            text = textGetter(soup)  # extract visible text from the HTML
            if text:
                print(f"Text from {url}:\n{text[:charsPrinted]}...\n")  # print preview of the text
                indexed_data.append([url, text])  # Store URL and text

            # Find and follow all links on this page
            for link in soup.find_all('a', href=True):  # loop through anchor tags with href attributes
                href = link['href']
                full_url = clean_url(urljoin(url, href))  # turn relative URLs into full ones and clean them

                parsed = urlparse(full_url)

                # Only queue links from the same domain and within depth limit
                if parsed.scheme in ["http", "https"] and parsed.netloc == base_domain and depth + 1 <= maxDepth:
                    if full_url not in visited:
                        queue.append((full_url, depth + 1))  # add new URL with increased depth
                        visited.add(full_url)  # Mark the URL as visited when it's queued

    # Call the function to save the indexed data AFTER the loop finishes
    save_index_to_file(indexed_data)

# Run the crawler starting from the given URL
bfsCrawler(start_url)
