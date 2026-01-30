"""
Web Scraper Module - "Brute Force" Edition
Designed to aggressively crawl homepage links when sitemaps fail.
"""

import re
import time
import logging
import random
from typing import List, Optional, Set, Dict
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# --- Data Models ---

@dataclass
class Article:
    url: str
    title: str
    text: str
    author: Optional[str] = None
    date: Optional[str] = None
    category: Optional[str] = None
    has_sources: bool = False
    source_links: List[str] = field(default_factory=list)
    is_opinion: bool = False

@dataclass
class SiteMetadata:
    domain: str
    has_about_page: bool = False
    about_text: str = ""
    ownership_disclosed: bool = False
    ownership_info: str = ""
    funding_disclosed: bool = False
    funding_info: str = ""
    location_disclosed: bool = False
    location_info: str = ""
    contact_info: str = ""
    has_author_pages: bool = False

# --- The Scraper Class ---

class MediaScraper:
    def __init__(self, base_url: str, max_articles: int = 30):
        self.base_url = base_url.rstrip('/')
        self.domain = urlparse(self.base_url).netloc.replace('www.', '')
        self.max_articles = max_articles
        self.visited_urls: Set[str] = set()
        self.session = requests.Session()
        
        # Robust Headers to look like a real browser (Chrome on Windows)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.google.com/',
        }

    def fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """Downloads and parses a page safely."""
        try:
            time.sleep(random.uniform(0.5, 1.5)) # Delay to avoid 429 Rate Limits
            resp = self.session.get(url, headers=self.headers, timeout=15)
            resp.raise_for_status()
            
            # Fix encoding issues
            if resp.encoding == 'ISO-8859-1':
                resp.encoding = resp.apparent_encoding
                
            return BeautifulSoup(resp.text, 'html.parser')
        except Exception as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return None

    def scrape_feed(self) -> List[Article]:
        """
        The Main Method called by profiler.py.
        Strategy: Get homepage, find internal links, scrape them.
        """
        logger.info(f"Scraping homepage: {self.base_url}")
        soup = self.fetch_page(self.base_url)
        if not soup:
            logger.error("Could not load homepage. Site might be blocking requests.")
            return []

        # 1. Collect all potential links
        candidates = set()
        for a in soup.find_all('a', href=True):
            href = a['href']
            full_url = urljoin(self.base_url, href)
            
            # Filter logic
            if self.domain not in full_url: continue # Internal only
            if len(full_url) < len(self.base_url) + 10: continue # Too short
            # Skip obvious non-article pages
            if any(x in full_url for x in ['/tag/', '/search/', '/category/', '/login', '.pdf', '.jpg']): continue
            
            candidates.add(full_url)

        logger.info(f"Found {len(candidates)} links on homepage. Scraping {self.max_articles} of them...")

        # 2. Scrape them in parallel
        articles = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Take top candidates (shuffle slightly to avoid getting only header links)
            target_links = list(candidates)
            random.shuffle(target_links)
            target_links = target_links[:self.max_articles * 2]
            
            future_to_url = {executor.submit(self._parse_article, url): url for url in target_links}
            
            for future in as_completed(future_to_url):
                if len(articles) >= self.max_articles: break
                
                res = future.result()
                if res and len(res.text) > 500: # Ensure valid article text
                    articles.append(res)
                    print(f"✅ Scraped: {res.title[:50]}...")
        
        return articles

    def _parse_article(self, url: str) -> Optional[Article]:
        """Parses a single article URL."""
        if url in self.visited_urls: return None
        self.visited_urls.add(url)

        soup = self.fetch_page(url)
        if not soup: return None

        # Extract Title
        title = soup.title.get_text(strip=True) if soup.title else ""
        h1 = soup.find('h1')
        if h1: title = h1.get_text(strip=True)

        # Extract Text (Heuristic: Find the container with the most paragraphs)
        best_div = None
        max_p = 0
        
        # Search common content containers
        candidates = soup.find_all(['div', 'article', 'section', 'main'])
        for div in candidates:
            p_count = len(div.find_all('p', recursive=False))
            if p_count > max_p:
                max_p = p_count
                best_div = div
        
        if best_div and max_p > 3:
            paragraphs = best_div.find_all('p')
        else:
            paragraphs = soup.find_all('p') # Fallback

        text = "\n\n".join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 30])

        if len(text) < 200: return None # Trash result

        # Check for Sources (External links)
        sources = []
        for a in soup.find_all('a', href=True):
            if 'http' in a['href'] and self.domain not in a['href']:
                sources.append(a['href'])

        return Article(
            url=url,
            title=title,
            text=text,
            author="Unknown",
            date="Unknown",
            has_sources=len(sources) > 0,
            source_links=sources
        )

    def get_metadata(self) -> SiteMetadata:
        """
        Scans homepage for 'About', 'Contact', 'Terms' to estimate transparency.
        """
        meta = SiteMetadata(domain=self.domain)
        
        soup = self.fetch_page(self.base_url)
        if not soup:
            return meta

        # Convert all link text to lowercase for searching
        links_text = " ".join([a.get_text().lower() for a in soup.find_all('a', href=True)])
        footer_text = " ".join([f.get_text().lower() for f in soup.find_all('footer')])

        # 1. Check for About Page
        if any(x in links_text for x in ['about us', 'about the bbc', 'who we are', 'our story']):
            meta.has_about_page = True
            
        # 2. Check for Contact/Location
        if any(x in links_text for x in ['contact', 'contact us', 'help', 'locations']):
            meta.location_disclosed = True
            meta.contact_info = "Found contact link"

        # 3. Check for Authors/Masthead
        if any(x in links_text for x in ['meet the team', 'editorial staff', 'authors', 'journalists']):
            meta.has_author_pages = True

        # 4. Check for Funding/Ownership (Keywords in footer often indicate this)
        if any(x in footer_text for x in ['copyright', 'all rights reserved', 'published by', 'funded by']):
            meta.ownership_disclosed = True # Basic assumption for standard footers
            meta.funding_disclosed = True

        return meta