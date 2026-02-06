"""
app.py — Streamlit Web Interface for Media Profiler
Deploy to HuggingFace Spaces with: sdk: streamlit

Usage:
  streamlit run app.py
"""

import json
import os
import streamlit as st
from urllib.parse import urlparse

from scraper import MediaScraper
from research import MediaProfiler
from storage import StorageManager
from report_generator import ReportGenerator

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Media Profiler",
    page_icon="📰",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_domain(url: str) -> str:
    parsed = urlparse(url if url.startswith("http") else f"https://{url}")
    domain = parsed.netloc or parsed.path
    return domain.replace("www.", "").split("/")[0].lower()


def load_cached_reports() -> list[dict]:
    """Load all previously analyzed reports from disk."""
    reports_dir = "reports"
    cached = []
    if not os.path.isdir(reports_dir):
        return cached
    for domain_dir in sorted(os.listdir(reports_dir)):
        data_path = os.path.join(reports_dir, domain_dir, "data.json")
        if os.path.isfile(data_path):
            try:
                with open(data_path) as f:
                    d = json.load(f)
                cached.append(d)
            except Exception:
                pass
    return cached


def run_analysis(url: str, force_refresh: bool = False):
    """Run the full pipeline with progress updates."""
    domain = extract_domain(url)
    storage = StorageManager()

    # Check cache
    if not force_refresh and storage.exists(domain):
        st.toast("Loaded from cache", icon="✅")
        report_text = storage.load_report_text(domain)
        data = storage.load_data(domain)
        return report_text, data.model_dump() if data else {}

    progress = st.progress(0, text="Starting analysis...")

    # 1. Scrape
    progress.progress(5, text="Scraping articles from homepage...")
    scraper = MediaScraper(url, max_articles=15)
    articles_obj = scraper.scrape_feed()

    if not articles_obj:
        st.error("No articles found. The site may be blocking requests.")
        return None, None

    articles_data = [{"title": a.title, "text": a.text, "url": a.url} for a in articles_obj]
    progress.progress(20, text=f"Scraped {len(articles_data)} articles")

    # 2. Profile
    progress.progress(25, text="Resolving outlet name...")
    profiler = MediaProfiler()

    # We run profile step by step so we can update progress
    researcher = profiler.researcher
    prof_domain = profiler._extract_domain(url)
    outlet_name = researcher.resolve_outlet_name(url, domain=prof_domain)
    progress.progress(30, text=f"Analyzing: {outlet_name}")

    progress.progress(35, text="Analyzing editorial bias...")
    report_data = profiler.profile(url, articles_data, outlet_name=outlet_name)
    progress.progress(75, text="Analysis complete")

    # 3. Generate report
    progress.progress(80, text="Generating narrative report...")
    generator = ReportGenerator()
    report_text = generator.generate(report_data)
    progress.progress(90, text="Saving results...")

    # 4. Save
    storage.save(domain, report_data, report_text)
    progress.progress(100, text="Done!")

    return report_text, report_data.model_dump()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("📰 Media Profiler")
st.caption("Analyze news outlets for political bias, factual reliability, and credibility — MBFC methodology")

# --- Sidebar: input + cached reports ---
with st.sidebar:
    st.header("Analyze a Site")
    url_input = st.text_input("News site URL", placeholder="https://www.bbc.com")
    force_refresh = st.checkbox("Force re-analysis (ignore cache)")
    analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

    st.divider()
    st.header("Previous Reports")
    cached = load_cached_reports()
    if cached:
        for r in cached:
            domain = r.get("target_domain", "?")
            name = r.get("outlet_name", domain)
            bias = r.get("bias_label", "—")
            if st.button(f"{name} — {bias}", key=f"cached_{domain}", use_container_width=True):
                st.session_state["view_domain"] = domain
    else:
        st.caption("No reports yet. Analyze a site to get started.")

    st.divider()
    st.caption("**How it works:**")
    st.caption("1. Scrapes articles (prioritizes hard news)")
    st.caption("2. 7 LLM analyzers evaluate bias, sourcing, fact-checks")
    st.caption("3. Researches history & ownership via about page + web")
    st.caption("4. Generates MBFC-style credibility report")

# --- Main area ---

# Handle analyze button
if analyze_btn and url_input:
    if not url_input.startswith("http"):
        url_input = "https://" + url_input
    with st.spinner("Running analysis..."):
        report_text, report_data = run_analysis(url_input, force_refresh)
    if report_text:
        st.session_state["report_text"] = report_text
        st.session_state["report_data"] = report_data
        st.session_state.pop("view_domain", None)

# Handle cached report click
if "view_domain" in st.session_state:
    domain = st.session_state["view_domain"]
    storage = StorageManager()
    report_text = storage.load_report_text(domain)
    data = storage.load_data(domain)
    if report_text:
        st.session_state["report_text"] = report_text
        st.session_state["report_data"] = data.model_dump() if data else {}
    st.session_state.pop("view_domain", None)

# Display report if available
if "report_data" in st.session_state and st.session_state.get("report_data"):
    rd = st.session_state["report_data"]
    rt = st.session_state.get("report_text", "")

    # --- Quick stats ---
    st.subheader(rd.get("outlet_name", "Unknown"))
    st.caption(f'{rd.get("target_url", "")} — Analyzed: {rd.get("analysis_date", "—")} — Articles: {rd.get("articles_analyzed", "—")}')

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Bias Rating", rd.get("bias_label", "—"), f'Score: {rd.get("bias_score", "—")}')
    with col2:
        st.metric("Factual Reporting", rd.get("factuality_label", "—"), f'Score: {rd.get("factuality_score", "—")}')
    with col3:
        cred_score = rd.get("credibility_score", 0)
        st.metric("Credibility", rd.get("credibility_label", "—"), f"Score: {cred_score:.1f}/10" if isinstance(cred_score, (int, float)) else "—")
    with col4:
        st.metric("Media Type", rd.get("media_type", "—"), f'Traffic: {rd.get("traffic_tier", "—")}')

    # --- Tabs ---
    tab_report, tab_bias, tab_facts, tab_data = st.tabs(["📄 Report", "⚖️ Bias Detail", "✅ Fact Checks", "📊 Raw Data"])

    with tab_report:
        st.markdown(rt)

    with tab_bias:
        eb = rd.get("editorial_bias_result")
        if eb:
            st.markdown(f"**Overall Bias:** {eb.get('overall_bias', '—')} (score: {eb.get('bias_score', '—')})")
            st.markdown(f"**MBFC Label:** {eb.get('mbfc_label', '—')}")

            lang = eb.get("uses_loaded_language", False)
            st.markdown(f"**Loaded Language:** {'Yes' if lang else 'No'}")
            if lang and eb.get("loaded_language_examples"):
                for ex in eb["loaded_language_examples"]:
                    st.markdown(f'- *"{ex}"*')

            positions = eb.get("policy_positions", [])
            if positions:
                st.markdown("### Policy Positions")
                for pp in positions:
                    domain_name = pp.get("domain", "—")
                    leaning = pp.get("leaning", "—")
                    indicators = pp.get("indicators", [])
                    source_articles = pp.get("source_articles", [])
                    with st.expander(f"**{domain_name}** — {leaning}"):
                        for ind in indicators:
                            st.markdown(f"- {ind}")
                        if source_articles:
                            st.caption("**Sources:** " + " | ".join(source_articles))

            ideology = eb.get("ideology_summary", "")
            economy = eb.get("economy_summary", "")
            if ideology or economy:
                st.markdown("### Ideology & Economy")
                if ideology:
                    st.markdown(f"**Ideology:** {ideology}")
                if economy:
                    st.markdown(f"**Economy:** {economy}")

            st.markdown("### Reasoning")
            st.info(eb.get("reasoning", "—"))
        else:
            st.warning("No editorial bias data available.")

    with tab_facts:
        fc = rd.get("fact_check_result")
        if fc:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total Checks", fc.get("total_checks_count", 0))
            with c2:
                st.metric("Failed Checks", fc.get("failed_checks_count", 0))
            with c3:
                st.metric("Score", f'{fc.get("score", "—")}/10')

            findings = fc.get("findings", [])
            if findings:
                st.markdown("### Findings")
                for f in findings:
                    verdict = f.get("verdict", "—")
                    claim = f.get("claim_summary", f.get("claim", "—"))
                    source = f.get("source_site", "—")
                    url = f.get("url", "")
                    st.markdown(f"- **[{verdict}]** {claim} — *{source}*" + (f" [link]({url})" if url else ""))
            else:
                st.success("No failed fact checks found in IFCN-approved fact-checkers.")
        else:
            st.warning("No fact check data available.")

    with tab_data:
        st.json(rd)

else:
    # Landing page
    st.markdown("---")
    st.markdown("### Enter a news site URL in the sidebar to get started")

    st.markdown("""
    **What gets analyzed:**
    - **Editorial Bias** — Policy positions across economic, social, environmental, healthcare, immigration domains
    - **Factual Reporting** — Fact-check search across PolitiFact, Snopes, FactCheck.org, FullFact
    - **Sourcing Quality** — Link extraction and source credibility assessment
    - **Pseudoscience** — Detection of anti-vax, climate denial, alternative medicine claims
    - **History & Ownership** — Founding year, owner, funding model, headquarters
    - **External Analyses** — MBFC, Ad Fontes, NewsGuard reviews

    **Methodology:** Follows [Media Bias/Fact Check](https://mediabiasfactcheck.com/methodology/) scoring.
    Bias scale: -10 (far left) to +10 (far right). Credibility = FactChecks(40%) + Sourcing(30%) + Pseudoscience(30%).
    """)
