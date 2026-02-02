import cloudscraper
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re
import os

# Initialize the scraper (mimics a real Chrome browser)
scraper = cloudscraper.create_scraper(
    browser={
        'browser': 'chrome',
        'platform': 'windows',
        'desktop': True
    }
)

URLS_FILE = 'found_urls.txt'
DATA_FILE = 'mbfc_data.csv'

def save_urls_to_file(urls):
    """Appends new URLs to a text file to ensure progress isn't lost."""
    # Read existing to avoid duplicates
    existing = set()
    if os.path.exists(URLS_FILE):
        with open(URLS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                existing.add(line.strip())
    
    with open(URLS_FILE, 'a', encoding='utf-8') as f:
        count = 0
        for url in urls:
            if url not in existing:
                f.write(url + '\n')
                count += 1
    return count

def get_existing_urls():
    """Loads all URLs we have already found."""
    if not os.path.exists(URLS_FILE):
        return []
    with open(URLS_FILE, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def get_soup(url):
    """
    Fetches URL using Cloudscraper to bypass 403/429/Disconnects.
    """
    retries = 3
    for i in range(retries):
        try:
            # Random sleep to look human
            time.sleep(random.uniform(2.0, 5.0))
            
            # cloudscraper handles the request automatically
            response = scraper.get(url, timeout=20)
            
            if response.status_code == 200:
                return BeautifulSoup(response.content, 'html.parser')
            elif response.status_code == 429:
                print(f"  [!] Rate limited (429). Sleeping 60s...")
                time.sleep(60)
            else:
                print(f"  [!] Error {response.status_code} for {url}")
        
        except Exception as e:
            print(f"  [!] Connection issue: {e}. Retrying...")
            time.sleep(10)
            
    return None

def collect_urls():
    """Scrapes category pages and saves URLs incrementally."""
    categories = {
        'left': 'https://mediabiasfactcheck.com/left/',
        'left-center': 'https://mediabiasfactcheck.com/leftcenter/',
        'center': 'https://mediabiasfactcheck.com/center/',
        'right-center': 'https://mediabiasfactcheck.com/right-center/',
        'right': 'https://mediabiasfactcheck.com/right/',
        'conspiracy': 'https://mediabiasfactcheck.com/conspiracy/',
        'pro-science': 'https://mediabiasfactcheck.com/pro-science/',
        'satire': 'https://mediabiasfactcheck.com/satire/'
    }

    print("--- Step 1: Collecting Source URLs ---")
    
    # Check what we already have
    existing_urls = get_existing_urls()
    print(f"Resuming... we already have {len(existing_urls)} URLs saved.")

    for cat_name, cat_url in categories.items():
        print(f"\nScanning Category: {cat_name}...")
        
        soup = get_soup(cat_url)
        if not soup:
            print(f"  [X] Failed to load {cat_name}. Skipping.")
            continue
        
        new_links = set()
        
        # Method 1: Table
        table = soup.find('table', {'id': 'mbfc-table'})
        if table:
            rows = table.find_all('tr')
            for row in rows:
                link = row.find('a')
                if link and 'href' in link.attrs:
                    url = link['href']
                    if "mediabiasfactcheck.com" in url:
                        new_links.add(url)
        
        # Method 2: Content Div (Fallback)
        else:
            entry_content = soup.find('div', class_='entry-content')
            if entry_content:
                links = entry_content.find_all('a')
                for link in links:
                    if 'href' in link.attrs:
                        url = link['href']
                        if "mediabiasfactcheck.com" in url and len(url) > 30:
                            new_links.add(url)

        # Save immediately!
        added_count = save_urls_to_file(new_links)
        print(f"  > Found {len(new_links)} links. ({added_count} new ones saved to file)")

    print("\nURL Collection Complete.")
    return get_existing_urls()

def parse_source_page(url):
    soup = get_soup(url)
    if not soup:
        return None

    data = {
        'url': url,
        'name': '',
        'bias_rating': None,
        'factual_reporting': None,
        'credibility': None,
        'country': None
    }

    h1 = soup.find('h1')
    if h1:
        data['name'] = h1.get_text(strip=True).replace(' – Bias and Credibility', '')

    text = soup.get_text(" ", strip=True)
    
    def extract(pattern):
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip()[:50] if match else None

    data['bias_rating'] = extract(r"Bias Rating:([^\n\r]+?)(?=(Factual|Country|MBFC|Media|Traffic|$))")
    data['factual_reporting'] = extract(r"Factual Reporting:([^\n\r]+?)(?=(Country|MBFC|Media|Traffic|$))")
    data['country'] = extract(r"Country:([^\n\r]+?)(?=(MBFC|Media|Traffic|$))")
    data['credibility'] = extract(r"Credibility Rating:([^\n\r]+?)(?=(History|Source|$))")

    return data

def main():
    # 1. Collect URLs (or load existing ones if we crashed previously)
    all_urls = collect_urls()
    
    if not all_urls:
        print("No URLs found. Exiting.")
        return

    # 2. Parse Sources
    print(f"\n--- Step 2: Parsing {len(all_urls)} Sources ---")
    
    # Load already parsed results to avoid re-doing work
    parsed_urls = set()
    results = []
    
    if os.path.exists(DATA_FILE):
        try:
            existing_df = pd.read_csv(DATA_FILE)
            parsed_urls = set(existing_df['url'].unique())
            results = existing_df.to_dict('records')
            print(f"Loaded {len(results)} already parsed sources from {DATA_FILE}")
        except:
            print("Could not read existing CSV. Starting fresh.")

    # Only process URLs we haven't parsed yet
    urls_to_process = [u for u in all_urls if u not in parsed_urls]
    
    # TEST MODE: Process only first 20 new ones
    # Remove [:20] when you are ready for the full run
    urls_to_process = urls_to_process[:20]

    for i, url in enumerate(urls_to_process):
        print(f"[{i+1}/{len(urls_to_process)}] Parsing: {url}")
        
        info = parse_source_page(url)
        if info:
            results.append(info)
        
        # Save every 5 items
        if i % 5 == 0:
            pd.DataFrame(results).to_csv(DATA_FILE, index=False)

    # Final Save
    pd.DataFrame(results).to_csv(DATA_FILE, index=False)
    print(f"Done! Saved to {DATA_FILE}")

if __name__ == "__main__":
    main()
