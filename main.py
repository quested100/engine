import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import signal
import sys
import random
import time
from collections import deque

# starting URL for the crawler
start_url = "https://www.apple.com/"

# creates a set to keep track of visited URLs
visited = set()

# maximum depth for crawling
maxDepth = 2

# number of characters printed
charsPrinted = 250

# handles Ctrl+C to stop the program
def signal_handler(sig, frame):
    print("\nCrawling interrupted by user.")
    sys.exit(0)

# register the signal handler
signal.signal(signal.SIGINT, signal_handler)

# adds a delay before making a new request to avoid overloading the server
def crawl_with_delay():
    # Random sleep time between 1 and 2 seconds
    time.sleep(random.randint(1, 2))

# gets and parses the HTML from the URL
def getSoup(url):
    try:
        response = requests.get(url, timeout=30)  # tries to get information from the URL, for 30 seconds
        response.raise_for_status()  # checks for HTTP errors (4xx, 5xx)
        return BeautifulSoup(response.text, 'html.parser')  # parses the HTML content
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

# extracts and cleans the text from the HTML content
def textGetter(soup):
    # Remove script, style, and noscript elements
    for tag in soup(['script', 'style', 'noscript']):
        tag.decompose()  # removes the tag and everything inside it

    # Get and clean the text
    text = soup.get_text(separator=' ', strip=True)  # adds a space between elements and removes leading/trailing whitespace
    return text  # returns the cleaned text

# the main crawler function using BFS (queue-based approach)
def bfsCrawler(start_url):
    queue = deque()
    queue.append((start_url, 0))  # each queue item is a tuple (url, depth)

    while queue:
        url, depth = queue.popleft()  # gets the next URL and its depth

        if depth > maxDepth or url in visited:  # if we’ve reached max depth or already visited, skip
            continue

        visited.add(url)  # mark the URL as visited
        print(f"\nVisiting (depth {depth}): {url}")  # show which page is being visited

        soup = getSoup(url)  # get the parsed HTML
        if not soup:
            continue  # skip if there's an error loading the page

        text = textGetter(soup)  # extract text from the page
        if text:  # check if text is not empty
            print(f"Text from {url}:\n{text[:charsPrinted]}...\n")  # print the first N characters

        # Find and process all links in the current page
        for link in soup.find_all('a', href=True):  # find all links with href attribute
            href = link['href']
            full_url = urljoin(url, href)  # builds a full absolute URL from a relative one

            # check if the domain of the link matches the starting domain
            if urlparse(full_url).netloc == urlparse(start_url).netloc:
                if full_url not in visited:  # make sure we haven’t already visited it
                    queue.append((full_url, depth + 1))  # add the new URL with increased depth
                    crawl_with_delay()  # adds a delay before crawling the next URL (to be nice to the server)

# starts the crawler from the starting URL
print(f"Starting crawl from: {start_url}")
bfsCrawler(start_url)
