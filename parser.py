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

def is_valid_source_url(url):
    """
    Validates if a URL is a valid MBFC source page (not a category/meta page).
    """
    if not url or "mediabiasfactcheck.com" not in url:
        return False

    # Exclude category pages, meta pages, and non-source URLs
    exclude_patterns = [
        '/left/', '/leftcenter/', '/center/', '/right-center/', '/right/',
        '/extremeleft/', '/extremeright/',
        '/conspiracy/', '/pro-science/', '/satire/', '/fake-news/',
        '/methodology/', '/about/', '/contact/', '/privacy-policy/',
        '/filtered-search/', '/re-evaluated-sources/',
        '/category/', '/tag/', '/page/', '/author/',
        '/wp-content/', '/wp-admin/', '/feed/',
        '/#', '/search/', '/login/', '/register/',
        '/mbfc-ratings-by-the-numbers/',
        'mediabiasfactcheck.com/?', # Query strings on homepage
    ]

    for pattern in exclude_patterns:
        if pattern in url.lower():
            return False

    # Must be a direct source page (e.g., /source-name/)
    # Valid URLs look like: https://mediabiasfactcheck.com/cnn/
    if url.endswith('/') or url.endswith('.html') or url.endswith('.htm'):
        # Check it has a slug (not just the domain)
        parts = url.rstrip('/').split('/')
        if len(parts) >= 4:  # https://mediabiasfactcheck.com/slug
            slug = parts[-1]
            # Slug should be alphanumeric with hyphens, not empty
            if slug and len(slug) > 1 and re.match(r'^[a-zA-Z0-9\-]+$', slug):
                return True

    return False


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
    # COMPLETE list of all MBFC category pages
    # Note: "extreme left" and "extreme right" are NOT separate pages -
    # they are included within the /left/ and /right/ categories
    #
    # MBFC Statistics (for reference):
    # - Total Sources Reviewed: 10,151 (includes inactive)
    # - Total Live Sources: 9,251
    # - Categories overlap (a source can be in both "right" AND "questionable")

    categories = {
        # Bias Rating Categories
        'left': 'https://mediabiasfactcheck.com/left/',                        # Left + Extreme Left
        'left-center': 'https://mediabiasfactcheck.com/leftcenter/',          # Left-Center
        'center': 'https://mediabiasfactcheck.com/center/',                    # Least Biased
        'right-center': 'https://mediabiasfactcheck.com/right-center/',       # Right-Center
        'right': 'https://mediabiasfactcheck.com/right/',                      # Right + Extreme Right

        # Special Categories (may overlap with bias categories)
        'questionable': 'https://mediabiasfactcheck.com/fake-news/',           # Questionable/Low Credibility
        'conspiracy': 'https://mediabiasfactcheck.com/conspiracy/',            # Conspiracy/Pseudoscience
        'pro-science': 'https://mediabiasfactcheck.com/pro-science/',          # Pro-Science
        'satire': 'https://mediabiasfactcheck.com/satire/',                    # Satire

        # Re-evaluated sources (sources that changed ratings)
        're-evaluated': 'https://mediabiasfactcheck.com/re-evaluated-sources/',
    }

    print("--- Step 1: Collecting Source URLs ---")
    
    # Check what we already have
    existing_urls = get_existing_urls()
    print(f"Resuming... we already have {len(existing_urls)} URLs saved.")

    for cat_name, cat_url in categories.items():
        print(f"\nScanning Category: {cat_name}...")

        # Handle pagination - scrape all pages for this category
        page_num = 1
        total_links_in_category = set()

        while True:
            # Construct URL for current page
            if page_num == 1:
                current_url = cat_url
            else:
                # MBFC uses /page/N/ for pagination
                current_url = cat_url.rstrip('/') + f'/page/{page_num}/'

            soup = get_soup(current_url)
            if not soup:
                if page_num == 1:
                    print(f"  [X] Failed to load {cat_name}. Skipping.")
                break

            new_links = set()

            # Method 1: Table with ID 'mbfc-table'
            table = soup.find('table', {'id': 'mbfc-table'})
            if table:
                rows = table.find_all('tr')
                for row in rows:
                    link = row.find('a')
                    if link and 'href' in link.attrs:
                        url = link['href']
                        if is_valid_source_url(url):
                            new_links.add(url)

            # Method 2: Any table on the page (some pages use different table structures)
            all_tables = soup.find_all('table')
            for tbl in all_tables:
                rows = tbl.find_all('tr')
                for row in rows:
                    links = row.find_all('a')
                    for link in links:
                        if 'href' in link.attrs:
                            url = link['href']
                            if is_valid_source_url(url):
                                new_links.add(url)

            # Method 3: Content Div (catches sources listed as plain links)
            entry_content = soup.find('div', class_='entry-content')
            if entry_content:
                links = entry_content.find_all('a')
                for link in links:
                    if 'href' in link.attrs:
                        url = link['href']
                        if is_valid_source_url(url):
                            new_links.add(url)

            # Method 4: Scan entire page for any MBFC source links
            all_links = soup.find_all('a')
            for link in all_links:
                if 'href' in link.attrs:
                    url = link['href']
                    if is_valid_source_url(url):
                        new_links.add(url)

            # If no new links found on this page, stop pagination
            new_unique = new_links - total_links_in_category
            if not new_unique and page_num > 1:
                break

            total_links_in_category.update(new_links)

            # Check if there's a next page link
            next_page = soup.find('a', class_='next') or soup.find('a', string=re.compile(r'Next|›|»'))
            if not next_page and page_num > 1:
                break

            if page_num > 1:
                print(f"    Page {page_num}: +{len(new_unique)} sources")

            page_num += 1

            # Safety limit to prevent infinite loops
            if page_num > 100:
                print(f"  [!] Hit page limit (100) for {cat_name}")
                break

        # Save immediately!
        added_count = save_urls_to_file(total_links_in_category)
        print(f"  > Found {len(total_links_in_category)} links across {page_num-1} page(s). ({added_count} new ones saved to file)")

    print("\nURL Collection Complete.")
    return get_existing_urls()

def print_statistics():
    """Prints statistics about collected URLs and parsed data."""
    print("\n" + "="*60)
    print("MBFC DATA COLLECTION STATISTICS")
    print("="*60)

    # URL Statistics
    urls = get_existing_urls()
    print(f"\nTotal URLs Collected: {len(urls)}")

    # If we have parsed data, show bias category breakdown
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE)
            print(f"Total Sources Parsed: {len(df)}")

            print("\n--- BIAS RATING BREAKDOWN ---")
            if 'bias_rating' in df.columns:
                bias_counts = df['bias_rating'].value_counts(dropna=False)
                for bias, count in bias_counts.items():
                    bias_label = bias if pd.notna(bias) else "Unknown/Not Parsed"
                    print(f"  {bias_label}: {count}")

            print("\n--- FACTUAL REPORTING BREAKDOWN ---")
            if 'factual_reporting' in df.columns:
                factual_counts = df['factual_reporting'].value_counts(dropna=False)
                for factual, count in factual_counts.items():
                    factual_label = factual if pd.notna(factual) else "Unknown/Not Parsed"
                    print(f"  {factual_label}: {count}")

            print("\n--- CREDIBILITY BREAKDOWN ---")
            if 'credibility' in df.columns:
                cred_counts = df['credibility'].value_counts(dropna=False)
                for cred, count in cred_counts.items():
                    cred_label = cred if pd.notna(cred) else "Unknown/Not Parsed"
                    print(f"  {cred_label}: {count}")

            print("\n--- COUNTRY BREAKDOWN (Top 10) ---")
            if 'country' in df.columns:
                country_counts = df['country'].value_counts(dropna=False).head(10)
                for country, count in country_counts.items():
                    country_label = country if pd.notna(country) else "Unknown"
                    print(f"  {country_label}: {count}")

        except Exception as e:
            print(f"Could not load parsed data: {e}")
    else:
        print("No parsed data file found yet.")

    print("\n" + "="*60)
    print("EXPECTED vs ACTUAL (from MBFC website stats)")
    print("="*60)
    print(f"Expected Total Sources (all-time):  10,151")
    print(f"Expected Live Sources:              9,251")
    print(f"Collected URLs:                     {len(urls)}")
    if len(urls) >= 9251:
        print(f"Status: COMPLETE ({len(urls) - 9251} extra from overlapping categories)")
    else:
        print(f"Missing from live sources:          {9251 - len(urls)}")
    print("="*60 + "\n")


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

def main(mode='full', test_limit=20):
    """
    Main function to collect and parse MBFC data.

    Args:
        mode: 'full' - collect URLs and parse sources
              'urls_only' - only collect URLs (no parsing)
              'parse_only' - only parse (use existing URLs)
              'stats_only' - only show statistics
        test_limit: Number of sources to parse in test mode (default: 20)
                   Set to None or 0 for full parsing
    """
    if mode == 'stats_only':
        print_statistics()
        return

    # 1. Collect URLs (or load existing ones if we crashed previously)
    if mode in ['full', 'urls_only']:
        all_urls = collect_urls()
    else:
        all_urls = get_existing_urls()

    if not all_urls:
        print("No URLs found. Exiting.")
        return

    # Show statistics after URL collection
    print_statistics()

    if mode == 'urls_only':
        print("URL collection complete. Skipping parsing.")
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

    # Apply test limit if specified
    if test_limit and test_limit > 0:
        print(f"\n*** TEST MODE: Processing only {test_limit} sources ***")
        print(f"*** Set test_limit=0 in main() for full parsing ***\n")
        urls_to_process = urls_to_process[:test_limit]

    if not urls_to_process:
        print("All URLs have already been parsed.")
        print_statistics()
        return

    for i, url in enumerate(urls_to_process):
        print(f"[{i+1}/{len(urls_to_process)}] Parsing: {url}")

        info = parse_source_page(url)
        if info:
            results.append(info)

        # Save every 5 items
        if (i + 1) % 5 == 0:
            pd.DataFrame(results).to_csv(DATA_FILE, index=False)
            print(f"  [Checkpoint] Saved {len(results)} records")

    # Final Save
    pd.DataFrame(results).to_csv(DATA_FILE, index=False)
    print(f"\nDone! Saved {len(results)} records to {DATA_FILE}")

    # Show final statistics
    print_statistics()


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == 'stats':
            main(mode='stats_only')
        elif arg == 'urls':
            main(mode='urls_only')
        elif arg == 'parse':
            main(mode='parse_only')
        elif arg == 'full':
            # Full mode with no limit
            main(mode='full', test_limit=0)
        else:
            print("Usage: python parser.py [stats|urls|parse|full]")
            print("  stats  - Show statistics only")
            print("  urls   - Collect URLs only (no parsing)")
            print("  parse  - Parse existing URLs only")
            print("  full   - Full run (collect + parse ALL sources)")
            print("  (no args) - Test mode: collect URLs + parse 20 sources")
    else:
        # Default: test mode with 20 sources
        main(mode='full', test_limit=20)
