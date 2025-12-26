# judge.py

from typing import Dict, List, Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.chat_models import init_chat_model
import json
from database import get_connection


# ====================== JUDGE OUTPUT SCHEMA ======================
class JudgeResult(BaseModel):
    overall_status: Literal["READY_TO_DELIVER", "NEEDS_MORE_RESEARCH"]
    confidence_score: float = Field(ge=0.0, le=1.0)
    coverage_assessment: Dict[str, Literal["covered", "partially_covered", "missing"]]
    detected_gaps: List[str]
    recommended_follow_up: List[str]
    termination_reasoning: str


judge_parser = JsonOutputParser(pydantic_object=JudgeResult)


# ====================== JUDGE PROMPT ======================
JUDGE_PROMPT = PromptTemplate(
    template="""
You are an expert research evaluator.

Evaluate the FINAL ANSWER based ONLY on:
- ORIGINAL QUERY
- DECOMPOSED SUB-QUESTIONS
- FINAL ANSWER

Do NOT use external knowledge.
Do NOT re-answer the query.

Tasks:
1. Check whether each sub-question is covered.
2. Judge depth and completeness.
3. Identify missing or shallow areas.
4. Recommend follow-up questions if gaps exist.

ORIGINAL QUERY:
{original_query}

SUB-QUESTIONS:
{sub_questions}

FINAL ANSWER:
{final_answer}

Return ONLY valid JSON.
{format_instructions}
""",
    input_variables=["original_query", "sub_questions", "final_answer"],
    partial_variables={
        "format_instructions": judge_parser.get_format_instructions()
    },
)

# ====================== INIT MODEL ======================
judge_llm = init_chat_model("groq:llama-3.1-8b-instant")
judge_chain = JUDGE_PROMPT | judge_llm | judge_parser


# ====================== JUDGE RUNNER ======================
def run_judge(
    original_query: str,
    sub_questions: List[str],
    final_answer: str,
    run_id: str,
) -> Dict:
    result = judge_chain.invoke(
        {
            "original_query": original_query,
            "sub_questions": "\n".join(f"- {q}" for q in sub_questions),
            "final_answer": final_answer,
        }
    )

    # ====================== DB WRITE ======================
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO judge_evaluations (
            research_run_id,
            overall_status,
            confidence_score,
            coverage_assessment,
            detected_gaps,
            recommended_follow_up,
            termination_reasoning
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        run_id,
        result["overall_status"],
        result["confidence_score"],
        json.dumps(result["coverage_assessment"]),
        json.dumps(result["detected_gaps"]),
        json.dumps(result["recommended_follow_up"]),
        result["termination_reasoning"]
    ))

    conn.commit()
    conn.close()

    return result

