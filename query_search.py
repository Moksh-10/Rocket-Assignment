import os
import asyncio
import json
from typing import List, Dict, Literal
from dotenv import load_dotenv
from tavily import AsyncTavilyClient
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from database import get_connection

# ====================== ENV SETUP ======================
# init_db()
load_dotenv()
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

tavily_client = AsyncTavilyClient()
analyzer_llm = init_chat_model("groq:llama-3.1-8b-instant")
final_llm = init_chat_model("groq:llama-3.1-8b-instant")

SearchType = Literal["depth", "breadth"]

# ====================== MODELS ======================
class SubAnswer(BaseModel):
    summary: List[str]

sub_answer_parser = JsonOutputParser(pydantic_object=SubAnswer)

SUB_ANSWER_PROMPT = PromptTemplate(
    template="""
You are a research analyst.

SUB-QUESTION:
{question}

SEARCH SNIPPETS:
{snippets}

Task:
- Answer the sub-question based on the search snippets
- Produce EXACTLY 4–5 concise factual bullet points
- Return ONLY a valid JSON object with a "summary" key containing the bullet points
- Do NOT include any other text, explanations, or schema definitions

Example output:
{{
  "summary": [
    "First key point from the research",
    "Second key point from the research",
    "Third key point from the research",
    "Fourth key point from the research"
  ]
}}

Return JSON:
{format_instructions}
""",
    input_variables=["question", "snippets"],
    partial_variables={
        "format_instructions": sub_answer_parser.get_format_instructions()
    },
)

FINAL_PROMPT = PromptTemplate(
    template="""
ORIGINAL QUERY:
{original_query}

SUB-QUESTION SUMMARIES:
{summaries}

Write a clear, structured, comprehensive answer that synthesizes all the information above.
""",
    input_variables=["original_query", "summaries"],
)

# ====================== SEARCH ======================
async def tavily_search(
    queries: List[str],
    search_type: SearchType,
) -> Dict[str, Dict]:

    max_results = 5 if search_type == "depth" else 10

    payloads = [
        {
            "query": q,
            "search_depth": "basic",
            "max_results": max_results,
            "include_raw_content": False,
            "include_usage": True,
        }
        for q in queries
    ]

    responses = await asyncio.gather(
        *(tavily_client.search(**p) for p in payloads)
    )

    results_by_question = {}

    for q, r in zip(queries, responses):
        urls, snippets = [], []
        
        if isinstance(r, dict) and "results" in r:
            results = r["results"]
        else:
            results = r if isinstance(r, list) else []
        
        for item in results:
            if isinstance(item, dict):
                score = item.get("score", 0)
                if score > 0.5:
                    if item.get("url"):
                        urls.append(item["url"])
                    content = item.get('content', '')
                    title = item.get('title', '')
                    snippets.append(f"- {title} | {content}")

        results_by_question[q] = {
            "urls": list(set(urls)),
            "snippets": "\n".join(snippets),
        }

    return results_by_question

# ====================== SUB-QUESTION ANALYSIS ======================
def extract_summary_from_response(response_text: str) -> List[str]:
    """Extract summary from LLM response, handling various response formats."""
    
    # First, try to parse as JSON directly
    try:
        # Try to find JSON in the response (in case there's extra text)
        lines = response_text.strip().split('\n')
        json_str = None
        
        # Look for lines that look like JSON
        for line in lines:
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                json_str = line
                break
        
        # If no single-line JSON found, try the entire response
        if json_str is None and response_text.strip().startswith('{'):
            json_str = response_text.strip()
        
        if json_str:
            data = json.loads(json_str)
            if isinstance(data, dict) and "summary" in data:
                summary = data["summary"]
                if isinstance(summary, list):
                    return summary
    
    except json.JSONDecodeError:
        pass
    
    # If JSON parsing failed, try to extract from "properties" format (seen in error)
    try:
        if "properties" in response_text:
            data = json.loads(response_text.strip())
            if "properties" in data and "summary" in data["properties"]:
                summary = data["properties"]["summary"]
                if isinstance(summary, list):
                    return summary
    except:
        pass
    
    # If all else fails, return empty list
    return ["Failed to parse response"]

async def analyze_sub_questions(search_results: Dict[str, Dict]) -> Dict[str, Dict]:
    analyzed = {}
    conn = get_connection()
    cur = conn.cursor()

    for question, data in search_results.items():
        # Get sub_question_id
        cur.execute("""
            SELECT id
            FROM sub_questions
            WHERE sub_question = ?
        """, (question,))
        row = cur.fetchone()

        if not row:
            continue

        sub_question_id = row["id"]

        # ====================== SUMMARY CACHE ======================
        cur.execute("""
            SELECT summary_json
            FROM sub_question_summaries
            WHERE sub_question_id = ?
        """, (sub_question_id,))

        cached = cur.fetchone()

        if cached:
            summary = json.loads(cached["summary_json"])
        else:
            prompt = SUB_ANSWER_PROMPT.format(
                question=question,
                snippets=data["snippets"],
            )

            response = await analyzer_llm.ainvoke(prompt)
            summary = extract_summary_from_response(response.content)

            cur.execute("""
                INSERT INTO sub_question_summaries (
                    sub_question_id,
                    summary_json
                ) VALUES (?, ?)
            """, (sub_question_id, json.dumps(summary)))

            conn.commit()

        analyzed[question] = {
            "summary": summary,
            "urls": data["urls"],
        }

    conn.close()
    return analyzed

# ====================== FINAL SYNTHESIS ======================
async def generate_final_answer(
    original_query: str,
    analyzed: Dict[str, Dict],
) -> str:

    summaries_block = []
    for q, data in analyzed.items():
        bullets = "\n".join(f"  - {b}" for b in data["summary"])
        summaries_block.append(f"{q}\n{bullets}")

    prompt = FINAL_PROMPT.format(
        original_query=original_query,
        summaries="\n\n".join(summaries_block),
    )

    response = await final_llm.ainvoke(prompt)
    return response.content

# ====================== ORCHESTRATOR ======================
async def run_pipeline(
    original_query: str,
    base_sub_questions: List[str],
    follow_up_questions: List[str],
    run_id: str,
):

    conn = get_connection()
    cur = conn.cursor()

    # Map sub_question text → sub_question_id
    cur.execute("""
        SELECT id, sub_question
        FROM sub_questions
        WHERE research_run_id = ?
    """, (run_id,))

    sub_q_map = {
        row["sub_question"]: row["id"]
        for row in cur.fetchall()
    }

    conn.close()

    # ====================== SEARCH CACHE CHECK ======================
    cached_search_results = {}
    missing_questions = []

    conn = get_connection()
    cur = conn.cursor()

    # for question in sub_questions:
    all_questions = list(dict.fromkeys(base_sub_questions + follow_up_questions))
    for question in all_questions:
        sub_question_id = sub_q_map.get(question)
        if not sub_question_id:
            missing_questions.append(question)
            continue

        cur.execute("""
            SELECT url, snippet
            FROM search_results
            WHERE sub_question_id = ? AND search_type = ?
        """, (sub_question_id, "depth"))

        rows = cur.fetchall()

        if rows:
            urls = set()
            snippets = []
            for r in rows:
                if r["url"]:
                    urls.add(r["url"])
                if r["snippet"]:
                    snippets.append(r["snippet"])

            cached_search_results[question] = {
                "urls": list(urls),
                "snippets": "\n".join(snippets),
            }
        else:
            missing_questions.append(question)

    conn.close()


    print("\n================ SEARCHING ==================\n")
    #search_results = await tavily_search(sub_questions, "depth")
    fresh_results = {}
    if missing_questions:
        fresh_results = await tavily_search(missing_questions, "depth")

    search_results = {**cached_search_results, **fresh_results}


    conn = get_connection()
    cur = conn.cursor()

    for question, data in fresh_results.items():
        sub_question_id = sub_q_map.get(question)

        if not sub_question_id:
            continue

        # for snippet_block in data["snippets"].split("\n"):
        for snippet_block in filter(None, data["snippets"].split("\n")):
            cur.execute("""
                INSERT INTO search_results (
                    sub_question_id,
                    search_query,
                    search_type,
                    url,
                    title,
                    snippet,
                    score
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                sub_question_id,
                question,
                "depth",
                None,          # URL stored separately below
                None,
                snippet_block,
                None
            ))

        for url in data["urls"]:
            cur.execute("""
                INSERT INTO search_results (
                    sub_question_id,
                    search_query,
                    search_type,
                    url
                ) VALUES (?, ?, ?, ?)
            """, (
                sub_question_id,
                question,
                "depth",
                url
            ))

    conn.commit()
    conn.close()


    print(f"Search results keys: {list(search_results.keys())}")
    for q, data in search_results.items():
        print(f"\nQuestion: {q}")
        print(f"Snippets length: {len(data['snippets'])}")
        print(f"URLs found: {len(data['urls'])}")

    print("\n================ ANALYZING ==================\n")
    analyzed = await analyze_sub_questions(search_results)

    print("\n================ FINAL ANSWER ==================\n")
    final_answer = await generate_final_answer(original_query, analyzed)
    print(final_answer)

    print("\n================ SOURCES ==================\n")
    for q, data in analyzed.items():
        print(f"\n{q}")
        for url in data["urls"]:
            print(f"- {url}")

    return final_answer

# ====================== RUN ======================
# if __name__ == "__main__":
#     ORIGINAL_QUERY = "Explain how blockchain technology works and its potential applications in healthcare"

#     SUB_QUESTIONS = [
#         "What is blockchain technology?",
#         "How does blockchain technology work?",
#         "What are the key features of blockchain technology?",
#         "How is blockchain technology used in healthcare?",
#         "What are the potential applications of blockchain technology in healthcare?",
#         "What are the benefits of using blockchain technology in healthcare?",
#         "What are the challenges of implementing blockchain technology in healthcare?",
#     ]

    # asyncio.run(run_pipeline(ORIGINAL_QUERY, SUB_QUESTIONS, decomposition_result["run_id"]))


