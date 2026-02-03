import cloudscraper
from bs4 import BeautifulSoup
import json
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
DATA_FILE = 'mbfc_data.json'

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

def load_json_data():
    """Load existing parsed data from JSON file."""
    if not os.path.exists(DATA_FILE):
        return []
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return []


def save_json_data(data):
    """Save parsed data to JSON file."""
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def count_values(data_list, key):
    """Count occurrences of values for a given key."""
    counts = {}
    for item in data_list:
        value = item.get(key)
        if value is None:
            value = "Unknown/Not Parsed"
        if value not in counts:
            counts[value] = 0
        counts[value] += 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def print_statistics():
    """Prints statistics about collected URLs and parsed data."""
    print("\n" + "="*70)
    print("MBFC DATA COLLECTION STATISTICS")
    print("="*70)

    # URL Statistics
    urls = get_existing_urls()
    print(f"\nTotal URLs Collected: {len(urls)}")

    # If we have parsed data, show breakdowns
    data = load_json_data()
    if data:
        print(f"Total Sources Parsed: {len(data)}")

        print("\n--- BIAS RATING BREAKDOWN ---")
        bias_counts = count_values(data, 'bias_rating')
        for bias, count in bias_counts.items():
            print(f"  {bias}: {count}")

        print("\n--- FACTUAL REPORTING BREAKDOWN ---")
        factual_counts = count_values(data, 'factual_reporting')
        for factual, count in factual_counts.items():
            print(f"  {factual}: {count}")

        print("\n--- CREDIBILITY BREAKDOWN ---")
        cred_counts = count_values(data, 'credibility_rating')
        for cred, count in cred_counts.items():
            print(f"  {cred}: {count}")

        print("\n--- COUNTRY BREAKDOWN (Top 15) ---")
        country_counts = count_values(data, 'country')
        for i, (country, count) in enumerate(country_counts.items()):
            if i >= 15:
                break
            print(f"  {country}: {count}")

        print("\n--- MEDIA TYPE BREAKDOWN ---")
        media_counts = count_values(data, 'media_type')
        for media, count in media_counts.items():
            print(f"  {media}: {count}")

        # Show average bias score
        bias_scores = [item['bias_score'] for item in data if item.get('bias_score') is not None]
        if bias_scores:
            avg_bias = sum(bias_scores) / len(bias_scores)
            print(f"\n--- BIAS SCORE STATISTICS ---")
            print(f"  Average Bias Score: {avg_bias:.2f}")
            print(f"  Min: {min(bias_scores):.1f}, Max: {max(bias_scores):.1f}")

        # Count sources with failed fact checks
        sources_with_fc = sum(1 for item in data if item.get('failed_fact_checks'))
        total_fc = sum(len(item.get('failed_fact_checks', [])) for item in data)
        print(f"\n--- FAILED FACT CHECKS ---")
        print(f"  Sources with failed fact checks: {sources_with_fc}")
        print(f"  Total failed fact checks: {total_fc}")

    else:
        print("No parsed data file found yet.")

    print("\n" + "="*70)
    print("EXPECTED vs ACTUAL (from MBFC website stats)")
    print("="*70)
    print(f"Expected Total Sources (all-time):  10,151")
    print(f"Expected Live Sources:              9,251")
    print(f"Collected URLs:                     {len(urls)}")
    if len(urls) >= 9251:
        print(f"Status: COMPLETE ({len(urls) - 9251} extra from overlapping categories)")
    else:
        print(f"Missing from live sources:          {9251 - len(urls)}")
    print("="*70 + "\n")


def parse_source_page(url):
    """
    Parses a single MBFC source page and extracts all available information.

    Returns a dictionary with:
    - Basic info: name, mbfc_url, source_url
    - Ratings: bias_rating, bias_score, factual_reporting, factual_score, credibility_rating
    - Metadata: country, country_freedom_rating, media_type, traffic_popularity
    - Content: history, ownership, analysis, overall_summary
    - Fact checks: failed_fact_checks (list)
    - Timestamps: last_updated
    """
    soup = get_soup(url)
    if not soup:
        return None

    data = {
        'mbfc_url': url,
        'name': None,
        'source_url': None,

        # Ratings
        'bias_rating': None,
        'bias_score': None,
        'factual_reporting': None,
        'factual_score': None,
        'credibility_rating': None,

        # Metadata
        'country': None,
        'country_freedom_rating': None,
        'media_type': None,
        'traffic_popularity': None,

        # Content sections
        'bias_category_description': None,
        'overall_summary': None,
        'history': None,
        'ownership': None,
        'analysis': None,

        # Fact checks
        'failed_fact_checks': [],

        # Timestamp
        'last_updated': None
    }

    # Extract name from h1
    h1 = soup.find('h1', class_='entry-title')
    if h1:
        name = h1.get_text(strip=True)
        # Clean up common suffixes
        for suffix in [' – Bias and Credibility', ' - Bias and Credibility',
                       ' – Media Bias/Fact Check', ' - Media Bias/Fact Check']:
            name = name.replace(suffix, '')
        data['name'] = name.strip()

    # Get the main content area
    entry_content = soup.find('div', class_='entry-content')
    if not entry_content:
        return data

    # Get full text for regex extraction
    full_text = entry_content.get_text(" ", strip=True)

    # === EXTRACT DETAILED REPORT FIELDS ===

    # Bias Rating with score: "LEFT (-6.8)" or just "LEFT"
    bias_match = re.search(r'Bias Rating:\s*([A-Z\-\s]+?)\s*\(([+-]?\d+\.?\d*)\)', full_text)
    if bias_match:
        data['bias_rating'] = bias_match.group(1).strip()
        try:
            data['bias_score'] = float(bias_match.group(2))
        except:
            pass
    else:
        # Fallback: just get the rating without score
        bias_match = re.search(r'Bias Rating:\s*([A-Z][A-Z\-\s]*?)(?:\s*Factual|\s*Country|\s*Press|\s*MBFC|\s*Media|\s*Traffic|$)', full_text)
        if bias_match:
            data['bias_rating'] = bias_match.group(1).strip()

    # Factual Reporting with score: "MIXED (5.0)" or just "HIGH"
    factual_match = re.search(r'Factual Reporting:\s*([A-Z][A-Z\s]*?)\s*\(([+-]?\d+\.?\d*)\)', full_text)
    if factual_match:
        data['factual_reporting'] = factual_match.group(1).strip()
        try:
            data['factual_score'] = float(factual_match.group(2))
        except:
            pass
    else:
        factual_match = re.search(r'Factual Reporting:\s*([A-Z][A-Z\s]*?)(?:\s*Country|\s*Press|\s*MBFC|\s*Media|\s*Traffic|$)', full_text)
        if factual_match:
            data['factual_reporting'] = factual_match.group(1).strip()

    # Country - be more specific with the pattern
    country_match = re.search(r'Country:\s*([A-Za-z][A-Za-z\s,]*?)(?=(?:\s*(?:Press|MBF|Media|Traffic|World|Factual)|$))', full_text, re.IGNORECASE)
    if country_match:
        data['country'] = country_match.group(1).strip()
        
    # Press Freedom Rating (MBFC uses both "Press Freedom Rating" and "MBFC's Country Freedom Rating/Rank")
    freedom_match = re.search(r'(?:Press Freedom Rating|MBF[Cc][’\']s Country Freedom (?:Rating|Rank)):\s*([A-Z][A-Z\s]*?)(?:\s*Media|\s*Traffic|\s*MBFC|$)', full_text)
    if freedom_match:
        data['country_freedom_rating'] = freedom_match.group(1).strip()
        
    # Media Type
    media_match = re.search(r'Media Type:\s*([A-Za-z][A-Za-z\s/,]*?)(?:\s*Traffic|\s*MBFC Credibility|$)', full_text)
    if media_match:
        data['media_type'] = media_match.group(1).strip()

    # Traffic/Popularity
    traffic_match = re.search(r'Traffic/Popularity:\s*([A-Za-z][A-Za-z\s]*?)(?:\s*MBFC Credibility|\s*History|\s*Funded|$)', full_text)
    if traffic_match:
        data['traffic_popularity'] = traffic_match.group(1).strip()

    # Credibility Rating
    cred_match = re.search(r'MBFC Credibility Rating:\s*([A-Z][A-Z\s]*?)(?:\s*History|\s*Funded|\s*Analysis|\s*Source|$)', full_text)
    if cred_match:
        data['credibility_rating'] = cred_match.group(1).strip()

    # === EXTRACT CONTENT SECTIONS ===

    # Find all paragraphs and headers
    all_elements = entry_content.find_all(['p', 'h2', 'h3', 'h4'])

    current_section = None
    section_content = {
        'history': [],
        'ownership': [],
        'analysis': [],
        'fact_checks': []
    }

    for elem in all_elements:
        text = elem.get_text(strip=True)

        # Detect section headers
        if elem.name in ['h2', 'h3', 'h4']:
            text_lower = text.lower()
            if 'history' in text_lower:
                current_section = 'history'
            elif 'funded' in text_lower or 'ownership' in text_lower:
                current_section = 'ownership'
            elif 'analysis' in text_lower or 'bias' in text_lower:
                current_section = 'analysis'
            elif 'fact check' in text_lower or 'failed' in text_lower:
                current_section = 'fact_checks'
            else:
                current_section = None
        elif current_section and elem.name == 'p' and len(text) > 20:
            section_content[current_section].append(text)

    # Combine section content
    if section_content['history']:
        data['history'] = ' '.join(section_content['history'])
    if section_content['ownership']:
        data['ownership'] = ' '.join(section_content['ownership'])
    if section_content['analysis']:
        data['analysis'] = ' '.join(section_content['analysis'])

    # === EXTRACT FAILED FACT CHECKS ===
    # Look for list items or paragraphs containing fact check info
    fact_check_section = entry_content.find(string=re.compile(r'Failed Fact Check', re.I))
    if fact_check_section:
        parent = fact_check_section.find_parent()
        if parent:
            # Find the next ul/ol list or subsequent paragraphs
            next_list = parent.find_next(['ul', 'ol'])
            if next_list:
                for li in next_list.find_all('li'):
                    fc_text = li.get_text(strip=True)
                    if fc_text and len(fc_text) > 10:
                        data['failed_fact_checks'].append(fc_text)
            else:
                # Look for inline fact checks in paragraphs
                for sibling in parent.find_next_siblings('p'):
                    sibling_text = sibling.get_text(strip=True)
                    if sibling_text and ('–' in sibling_text or '-' in sibling_text) and len(sibling_text) > 20:
                        # Likely a fact check entry like "Claim - Rating"
                        if 'Overall' in sibling_text:
                            break
                        data['failed_fact_checks'].append(sibling_text)

    # === EXTRACT OVERALL SUMMARY ===
    # Usually starts with "Overall, we rate..."
    overall_match = re.search(r'(Overall,?\s+.*?(?:\.[^.]+){0,3}\.)', full_text)
    if overall_match:
        data['overall_summary'] = overall_match.group(1).strip()
    
    # Strategy 2: Fallback for bullet-point summaries (looking for "is rated" inside <li>)
    if not data['overall_summary']:
        list_items = entry_content.find_all('li')
        for li in list_items:
            li_text = li.get_text(strip=True)
            # Check for signature summary phrases
            if ('is rated' in li_text or 'we rate' in li_text) and len(li_text) > 50:
                 # Simple filter to avoid grabbing menu items or footers
                if any(x in li_text for x in ['Left', 'Right', 'Center', 'Satire', 'Bias', 'Factual']):
                    data['overall_summary'] = li_text
                    break

    # === EXTRACT BIAS CATEGORY DESCRIPTION ===
    # The intro paragraph that describes the bias category
    first_paras = entry_content.find_all('p', limit=3)
    for p in first_paras:
        p_text = p.get_text(strip=True)
        if 'BIAS' in p_text.upper() and ('media sources' in p_text.lower() or 'these sources' in p_text.lower()):
            data['bias_category_description'] = p_text
            break

    # === EXTRACT SOURCE URL ===
    source_match = re.search(r'Source:\s*(https?://[^\s<>"]+)', full_text)
    if source_match:
        data['source_url'] = source_match.group(1).strip()
    else:
        # Look for link in the page
        source_link = entry_content.find('a', href=re.compile(r'^https?://(?!mediabiasfactcheck)'))
        if source_link and 'href' in source_link.attrs:
            potential_url = source_link['href']
            # Filter out social media and common non-source URLs
            if not any(x in potential_url for x in ['facebook.com', 'twitter.com', 'linkedin.com', 'patreon.com']):
                data['source_url'] = potential_url

    # === EXTRACT LAST UPDATED ===
    updated_match = re.search(r'Last Updated on ([A-Za-z]+ \d+, \d+)', full_text)
    if updated_match:
        data['last_updated'] = updated_match.group(1)

    return data

def main(mode='full', test_limit=0):
    """
    Main function to collect and parse MBFC data.

    Args:
        mode: 'full' - collect URLs and parse sources
              'urls_only' - only collect URLs (no parsing)
              'parse_only' - only parse (use existing URLs)
              'stats_only' - only show statistics
        test_limit: Number of sources to parse (default: 0 = no limit)
                   Set to a positive number to limit parsing for testing
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
    results = load_json_data()
    parsed_urls = {item['mbfc_url'] for item in results}

    if results:
        print(f"Loaded {len(results)} already parsed sources from {DATA_FILE}")

    # Only process URLs we haven't parsed yet
    urls_to_process = [u for u in all_urls if u not in parsed_urls]

    # Apply test limit if specified
    if test_limit and test_limit > 0:
        print(f"\n*** TEST MODE: Processing only {test_limit} sources ***")
        print(f"*** Run 'python parser.py parse' for full parsing ***\n")
        urls_to_process = urls_to_process[:test_limit]
    else:
        print(f"\n*** FULL MODE: Processing all {len(urls_to_process)} remaining sources ***")
        print(f"*** This will take a while. Progress is saved every 10 sources. ***\n")

    if not urls_to_process:
        print("All URLs have already been parsed.")
        print_statistics()
        return

    failed_urls = []

    for i, url in enumerate(urls_to_process):
        print(f"[{i+1}/{len(urls_to_process)}] Parsing: {url}")

        try:
            info = parse_source_page(url)
            if info:
                results.append(info)
            else:
                failed_urls.append(url)
                print(f"  [!] Failed to parse: {url}")
        except Exception as e:
            failed_urls.append(url)
            print(f"  [!] Error parsing {url}: {e}")

        # Save checkpoint every 10 items
        if (i + 1) % 10 == 0:
            save_json_data(results)
            print(f"  [Checkpoint] Saved {len(results)} records to {DATA_FILE}")

    # Final Save
    save_json_data(results)
    print(f"\nDone! Saved {len(results)} records to {DATA_FILE}")

    if failed_urls:
        print(f"\n[!] Failed to parse {len(failed_urls)} URLs:")
        for url in failed_urls[:10]:
            print(f"    - {url}")
        if len(failed_urls) > 10:
            print(f"    ... and {len(failed_urls) - 10} more")

        # Save failed URLs for retry
        with open('failed_urls.txt', 'w', encoding='utf-8') as f:
            for url in failed_urls:
                f.write(url + '\n')
        print(f"  Failed URLs saved to failed_urls.txt")

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
            # Parse all remaining URLs
            main(mode='parse_only', test_limit=0)
        elif arg == 'test':
            # Test mode: parse only 20 sources
            main(mode='parse_only', test_limit=40)
        elif arg == 'full':
            # Full mode: collect URLs + parse ALL
            main(mode='full', test_limit=0)
        else:
            print("Usage: python parser.py [stats|urls|parse|test|full]")
            print("  stats  - Show statistics only")
            print("  urls   - Collect URLs only (no parsing)")
            print("  parse  - Parse ALL remaining URLs (resumes from checkpoint)")
            print("  test   - Parse only 20 sources for testing")
            print("  full   - Full run (collect URLs + parse ALL sources)")
            print("  (no args) - Same as 'parse' (parse all remaining URLs)")
    else:
        # Default: parse all remaining URLs
        main(mode='parse_only', test_limit=0)
