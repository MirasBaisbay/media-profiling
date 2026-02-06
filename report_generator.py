"""
report_generator.py
Generates the specific MBFC-style prose report using LLM synthesis.
"""

from langchain_openai import ChatOpenAI
from schemas import ComprehensiveReportData

class ReportGenerator:
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.4):
        self.llm = ChatOpenAI(model=model, temperature=temperature)

    def generate(self, data: ComprehensiveReportData) -> str:
        """
        Synthesizes the ComprehensiveReportData into the specific MBFC prose format.
        """
        
        # Prepare context for the prompt
        context_str = data.model_dump_json()
        
        # Build explicit history/ownership context so the LLM doesn't miss it
        history_context = f"""
HISTORY DATA:
- History Summary: {data.history_summary or 'Not available'}
- Founded: {data.founding_year or 'Unknown'}
- Founder(s): {getattr(data, 'founder', None) or 'Unknown'}
- Original Name: {getattr(data, 'original_name', None) or 'Same as current'}
- Key Events: {', '.join(getattr(data, 'key_events', []) or []) or 'None listed'}

OWNERSHIP DATA:
- Owner: {data.owner or 'Unknown'}
- Parent Company: {getattr(data, 'parent_company', None) or 'N/A'}
- Funding Model: {data.funding_model or 'Unknown'}
- Headquarters: {data.headquarters or 'Unknown'}
- Additional Notes: {getattr(data, 'ownership_notes', None) or 'None'}

IDEOLOGY & ECONOMY SUMMARIES (from bias analysis):
- Ideology: {data.editorial_bias_result.ideology_summary if data.editorial_bias_result and data.editorial_bias_result.ideology_summary else 'Not available'}
- Economy: {data.editorial_bias_result.economy_summary if data.editorial_bias_result and data.editorial_bias_result.economy_summary else 'Not available'}
"""

        prompt = f"""
You are a senior editor for Media Bias/Fact Check (MBFC). Write a comprehensive report for the news outlet "{data.outlet_name}" based on the provided analysis data.

Use the exact structure and tone of MBFC reports.

{history_context}

FULL ANALYSIS DATA (JSON):
{context_str}

STRUCTURE & REQUIREMENTS:

1. **Header**: The Bias Rating description block.
   - If bias is Left/Left-Center: "These sources are moderately to strongly biased toward liberal causes..."
   - If bias is Right/Right-Center: "These sources are moderately to strongly biased toward conservative causes..."
   - If Center/Least Biased: "These sources have minimal bias..."

2. **Overall Summary**: A bolded paragraph summarizing the rating.
   - Format: "Overall, we rate {data.outlet_name} [Bias Label]... We also rate them [Factuality Label]..."
   - Include the reasoning (e.g., "due to strong editorial positions on climate change" or "based on a clean fact check record").

3. **Detailed Report**: A list of metrics.
   - Bias Rating: {data.bias_label.upper()} ({data.bias_score})
   - Factual Reporting: {data.factuality_label.upper()} ({data.factuality_score})
   - Country: [Use the Headquarters field from OWNERSHIP DATA above]
   - Media Type: {data.media_type}
   - Traffic/Popularity: {data.traffic_tier}
   - Credibility Rating: {data.credibility_label.upper()}

4. **History**:
   - Write a narrative paragraph using the HISTORY DATA above.
   - IMPORTANT: Include founding year, founder, original name, and key milestones when available.
   - Use the history_summary as a basis and expand with the structured fields.
   - If founding_year or founder are available, they MUST appear in this section.

5. **Funded by / Ownership**:
   - IMPORTANT: Use the OWNERSHIP DATA above to describe who owns it, parent company, where it is based, and how it is funded.
   - If owner, funding_model, or headquarters are available, they MUST appear in this section.
   - Mention transparency if relevant.

6. **Analysis / Bias**:
   - This is the main body. Discuss the editorial stance.
   - **MANDATORY**: You MUST list specific policy positions found in the 'editorial_bias_result.policy_positions' data.
     For each position, write a DIRECT, EXPLICIT statement about what the outlet thinks or advocates.
     DO NOT be vague. State the position clearly.
     Format: "On [domain], the outlet [specific position]. For example, in '[article title]', they [specific evidence]."
     Example: "On environmental policy, the outlet treats climate change as an urgent human-caused crisis and advocates for renewable energy transition. In 'Climate summit fails to deliver binding commitments', the outlet frames government inaction as negligent."
     Example: "On economic policy, the outlet supports free-market principles and lower corporate taxes. In 'Tax reform bill advances in Senate', the reporting emphasizes economic growth benefits over income inequality concerns."
     Example (Center): "On healthcare, the outlet presents both universal healthcare and private insurance arguments without editorial preference. In 'Healthcare debate heats up ahead of midterms', both Democratic and Republican proposals receive equal coverage."
   - **MANDATORY**: Include the source article title for EACH piece of evidence. If 'source_articles' is provided in the policy position data, reference those articles.
   - If loaded language examples exist in 'loaded_language_examples', list them explicitly with the article context.
   - Mention external critiques found in the 'external_analyses' data.

   **IDEOLOGY & ECONOMY SECTION** (write this as a sub-section within Analysis/Bias):
   - Summarize the outlet's overall ideological position in 2-3 direct sentences.
     Example: "Ideologically, BBC positions itself as a centrist public broadcaster. It does not advocate for specific political parties but tends to reflect mainstream UK consensus on issues like climate change and public healthcare."
     Example: "Ideologically, Fox News consistently promotes conservative positions. It advocates for lower taxes, stricter immigration enforcement, and traditional social values."
   - Summarize the outlet's economic stance in 1-2 direct sentences.
     Example: "Economically, the outlet favors free-market capitalism with minimal government intervention."
     Example: "Economically, the outlet supports a mixed economy with strong social safety nets and progressive taxation."

7. **Failed Fact Checks**:
   - List them as bullet points if they exist in the data.
   - If none, state: "A search of IFCN fact checkers revealed no failed fact checks in the last 5 years."

TONE: Objective, professional, journalistic, but decisive about the bias rating.
DO NOT say "information not provided" or "not available" for fields that have actual values in the data above.
        """

        messages = [
            {"role": "system", "content": "You are an expert media analyst writing a report for Media Bias/Fact Check."},
            {"role": "user", "content": prompt}
        ]

        response = self.llm.invoke(messages)
        return response.content