"""
Research Module - Web Research for MBFC Reports

This module gathers external context about media outlets through
targeted web searches for:
1. History (founding, key events, evolution)
2. Ownership and funding information
3. External analysis and criticism
4. Existing MBFC entry (for validation)

Uses DuckDuckGo search and LLM extraction for structured data.
"""

import logging
import json
import re
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse
from dataclasses import asdict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun

from evidence import (
    ExternalEvidence, HistoryInfo, OwnershipInfo,
    ExternalAnalysis, ResearchResults
)

logger = logging.getLogger(__name__)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
search_tool = DuckDuckGoSearchRun()


class MediaResearcher:
    """
    Gathers external context for comprehensive MBFC-style reports.

    Performs 2-3 targeted web searches per outlet to gather:
    - History and founding information
    - Ownership and funding details
    - External criticism and analysis
    """

    def __init__(self):
        self.search_tool = search_tool
        self.llm = llm

    def research_outlet(
        self,
        url: str,
        outlet_name: Optional[str] = None
    ) -> ResearchResults:
        """
        Perform comprehensive research on a media outlet.

        Args:
            url: The outlet's URL
            outlet_name: Optional name override (otherwise extracted from URL)

        Returns:
            ResearchResults with all gathered information
        """
        # Extract outlet name from URL if not provided
        if not outlet_name:
            parsed = urlparse(url)
            domain = parsed.netloc.replace('www.', '')
            outlet_name = domain.split('.')[0].upper()
            # Handle special cases
            if 'bbc' in domain.lower():
                outlet_name = 'BBC'
            elif 'cnn' in domain.lower():
                outlet_name = 'CNN'
            elif 'nytimes' in domain.lower():
                outlet_name = 'New York Times'
            elif 'washingtonpost' in domain.lower():
                outlet_name = 'Washington Post'
            elif 'foxnews' in domain.lower():
                outlet_name = 'Fox News'

        logger.info(f"Researching outlet: {outlet_name} ({url})")

        results = ResearchResults(
            outlet_name=outlet_name,
            outlet_url=url
        )

        # Perform searches in sequence (to avoid rate limits)
        try:
            # 1. Search for history
            history_info = self._search_history(outlet_name)
            results.history = history_info
            results.all_evidence.extend(history_info.sources)
            results.search_queries_used.append(f"{outlet_name} founded history media")

            # 2. Search for ownership
            ownership_info = self._search_ownership(outlet_name)
            results.ownership = ownership_info
            results.all_evidence.extend(ownership_info.sources)
            results.search_queries_used.append(f"{outlet_name} ownership funded by parent company")

            # 3. Search for external analysis
            external = self._search_external_analysis(outlet_name)
            results.external_analyses = external
            for analysis in external:
                results.all_evidence.append(ExternalEvidence(
                    source_name=analysis.source_name,
                    source_url=analysis.source_url,
                    finding=analysis.summary,
                    finding_type="analysis"
                ))
            results.search_queries_used.append(f"{outlet_name} bias analysis criticism")

        except Exception as e:
            logger.error(f"Research failed for {outlet_name}: {e}")

        return results

    def _search_history(self, outlet_name: str) -> HistoryInfo:
        """Search for outlet history and founding information."""
        query = f'"{outlet_name}" founded history media news organization'

        try:
            search_results = self.search_tool.run(query)

            # Use LLM to extract structured history info
            prompt = f"""
Extract history information about the media outlet "{outlet_name}" from these search results.

SEARCH RESULTS:
{search_results[:3000]}

Return a JSON object with:
{{
    "founding_year": <number or null>,
    "founder": "<name or null>",
    "original_name": "<original name if different, or null>",
    "key_events": ["<event 1>", "<event 2>", ...],
    "summary": "<2-3 sentence history summary>"
}}

Return ONLY the JSON object, nothing else.
"""
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()
            content = content.replace('```json', '').replace('```', '').strip()
            data = json.loads(content)

            history = HistoryInfo(
                founding_year=data.get("founding_year"),
                founder=data.get("founder"),
                original_name=data.get("original_name"),
                key_events=data.get("key_events", [])
            )

            # Add source evidence
            history.sources.append(ExternalEvidence(
                source_name="Web Search",
                source_url="",
                finding=data.get("summary", ""),
                finding_type="history",
                reliability="medium"
            ))

            return history

        except Exception as e:
            logger.warning(f"History search failed: {e}")
            return HistoryInfo()

    def _search_ownership(self, outlet_name: str) -> OwnershipInfo:
        """Search for ownership and funding information."""
        query = f'"{outlet_name}" ownership funded by parent company headquarters'

        try:
            search_results = self.search_tool.run(query)

            # Use LLM to extract structured ownership info
            prompt = f"""
Extract ownership and funding information about "{outlet_name}" from these search results.

SEARCH RESULTS:
{search_results[:3000]}

Return a JSON object with:
{{
    "owner": "<owner name or null>",
    "parent_company": "<parent company or null>",
    "funding_model": "<advertising/subscription/public/nonprofit/mixed or null>",
    "headquarters": "<city, country or null>",
    "notes": "<any relevant notes about ownership/funding>"
}}

Return ONLY the JSON object, nothing else.
"""
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()
            content = content.replace('```json', '').replace('```', '').strip()
            data = json.loads(content)

            ownership = OwnershipInfo(
                owner=data.get("owner"),
                parent_company=data.get("parent_company"),
                funding_model=data.get("funding_model"),
                headquarters=data.get("headquarters"),
                transparency_notes=data.get("notes", "")
            )

            # Add source evidence
            ownership.sources.append(ExternalEvidence(
                source_name="Web Search",
                source_url="",
                finding=f"Owner: {ownership.owner}, Funding: {ownership.funding_model}",
                finding_type="ownership",
                reliability="medium"
            ))

            return ownership

        except Exception as e:
            logger.warning(f"Ownership search failed: {e}")
            return OwnershipInfo()

    def _search_external_analysis(self, outlet_name: str) -> List[ExternalAnalysis]:
        """Search for external analysis and criticism of the outlet."""
        query = f'"{outlet_name}" media bias analysis criticism review'

        analyses = []

        try:
            search_results = self.search_tool.run(query)

            # Use LLM to extract structured analysis
            prompt = f"""
Extract external analyses and criticism about "{outlet_name}" from these search results.

SEARCH RESULTS:
{search_results[:3000]}

Return a JSON object with:
{{
    "analyses": [
        {{
            "source_name": "<source name>",
            "source_url": "<url if available>",
            "summary": "<brief summary of analysis/criticism>",
            "sentiment": "positive" | "negative" | "neutral" | "mixed"
        }},
        ...
    ]
}}

Include up to 3 most relevant analyses. Focus on media watchdogs, journalism reviews, and academic sources.
Return ONLY the JSON object, nothing else.
"""
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()
            content = content.replace('```json', '').replace('```', '').strip()
            data = json.loads(content)

            for item in data.get("analyses", []):
                analyses.append(ExternalAnalysis(
                    source_name=item.get("source_name", "Unknown"),
                    source_url=item.get("source_url", ""),
                    summary=item.get("summary", ""),
                    sentiment=item.get("sentiment", "neutral")
                ))

        except Exception as e:
            logger.warning(f"External analysis search failed: {e}")

        return analyses

    def _check_existing_mbfc(self, outlet_name: str) -> Optional[Dict]:
        """Check if MBFC already has an entry for this outlet."""
        query = f'site:mediabiasfactcheck.com "{outlet_name}"'

        try:
            search_results = self.search_tool.run(query)

            if 'mediabiasfactcheck.com' in search_results.lower():
                # Extract URL if present
                urls = re.findall(r'https://mediabiasfactcheck\.com/[^\s]+', search_results)
                if urls:
                    return {
                        "exists": True,
                        "url": urls[0],
                        "note": "MBFC entry found - can be used for validation"
                    }

            return {"exists": False}

        except Exception as e:
            logger.warning(f"MBFC check failed: {e}")
            return {"exists": False}


def research_node(state: Dict) -> Dict:
    """
    LangGraph node for research phase.

    Gathers external context about the media outlet.
    """
    researcher = MediaResearcher()

    # Extract outlet name from scraped metadata if available
    outlet_name = None
    if state.get("site_metadata"):
        # Try to get name from site metadata
        pass

    results = researcher.research_outlet(
        url=state["target_url"],
        outlet_name=outlet_name
    )

    return {
        "research_results": asdict(results)
    }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    researcher = MediaResearcher()

    # Test with BBC
    results = researcher.research_outlet("https://www.bbc.com", "BBC")

    print("\n" + "=" * 60)
    print("RESEARCH RESULTS")
    print("=" * 60)
    print(f"\nOutlet: {results.outlet_name}")
    print(f"URL: {results.outlet_url}")

    print("\n--- HISTORY ---")
    if results.history.founding_year:
        print(f"Founded: {results.history.founding_year}")
    if results.history.founder:
        print(f"Founder: {results.history.founder}")
    if results.history.key_events:
        print("Key Events:")
        for event in results.history.key_events[:3]:
            print(f"  - {event}")

    print("\n--- OWNERSHIP ---")
    if results.ownership.owner:
        print(f"Owner: {results.ownership.owner}")
    if results.ownership.funding_model:
        print(f"Funding: {results.ownership.funding_model}")
    if results.ownership.headquarters:
        print(f"HQ: {results.ownership.headquarters}")

    print("\n--- EXTERNAL ANALYSIS ---")
    for analysis in results.external_analyses:
        print(f"  [{analysis.sentiment}] {analysis.source_name}: {analysis.summary[:100]}...")

    print(f"\nSearches performed: {len(results.search_queries_used)}")
    print(f"Evidence items collected: {len(results.all_evidence)}")
