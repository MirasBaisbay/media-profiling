"""
main_pipeline.py
The primary entry point for analyzing a site.
"""

import argparse
import logging
import sys
from urllib.parse import urlparse

from scraper import MediaScraper
from research import MediaProfiler
from storage import StorageManager
from report_generator import ReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def analyze_site(url: str, force_refresh: bool = False):
    """
    Main logic flow:
    1. Extract domain.
    2. Check storage for existing fresh report.
    3. If cached, load and print.
    4. If not, scrape -> profile -> generate -> save -> print.
    """
    
    # 1. Setup
    domain = urlparse(url).netloc.replace("www.", "")
    storage = StorageManager()
    
    # 2. Check Cache
    if not force_refresh and storage.exists(domain):
        logger.info(f"✅ Found cached report for {domain}")
        
        # Load the pretty report
        report_text = storage.load_report_text(domain)
        
        # (Optional) Load data if you need to do something programmatic with it
        # data = storage.load_data(domain)
        
        print("\n" + "="*80)
        print(f"CACHED REPORT: {domain.upper()}")
        print("="*80 + "\n")
        print(report_text)
        return

    # 3. If no cache, perform analysis
    logger.info(f"🚀 Starting fresh analysis for {domain}...")
    
    # A. Scrape
    scraper = MediaScraper(url, max_articles=15)
    articles_obj = scraper.scrape_feed() # Returns Article objects
    
    if not articles_obj:
        logger.error("No articles found. Aborting.")
        return

    # Convert Article objects to dicts for the profiler
    articles_data = [
        {"title": a.title, "text": a.text, "url": a.url}
        for a in articles_obj
    ]

    # B. Profile (Run all analyzers)
    profiler = MediaProfiler()
    # Note: profile() returns a ComprehensiveReportData object
    report_data = profiler.profile(url, articles_data)

    # C. Generate Prose Report
    logger.info("✍️  Generating narrative report...")
    generator = ReportGenerator()
    report_text = generator.generate(report_data)

    # D. Save Results
    storage.save(domain, report_data, report_text)

    # E. Output
    print("\n" + "="*80)
    print(f"NEW REPORT: {domain.upper()}")
    print("="*80 + "\n")
    print(report_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Media Bias Analysis Pipeline")
    parser.add_argument("url", help="The URL of the news site to analyze")
    parser.add_argument("--refresh", action="store_true", help="Ignore cache and force re-analysis")
    
    args = parser.parse_args()
    
    analyze_site(args.url, force_refresh=args.refresh)