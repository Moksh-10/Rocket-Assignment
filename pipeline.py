import asyncio
from typing import Dict, Any
from query_classification_decomposition import research_decomposition_pipeline
from query_search import run_pipeline
from llm_as_a_judge import run_judge
from database import get_connection
from report_generation import generate_report
from vector_db import store_run_vector


class ResearchPipeline:
    async def run(self, query: str) -> Dict[str, Any]:

        MAX_PASSES = 3
        CONFIDENCE_THRESHOLD = 0.85

        print("\n================ NEW RESEARCH RUN ================")
        print(f"Query: {query}")

        # ---------- Step 1: Decomposition ----------
        decomposition_result = research_decomposition_pipeline(query)
        run_id = decomposition_result["run_id"]
        base_sub_questions = decomposition_result["decomposition"]["sub_questions"]

        print(f"\nRun ID: {run_id}")
        print(f"Initial sub-questions ({len(base_sub_questions)}):")
        for q in base_sub_questions:
            print(" -", q)

        follow_up_questions: list[str] = []
        final_answer = None
        judge_result = None

        # ====================== MAIN LOOP ======================
        for pass_idx in range(MAX_PASSES):
            print(f"\n================ PASS {pass_idx + 1}/{MAX_PASSES} =================")

            # ---------- Decide questions ----------
            if pass_idx == 0:
                base_qs = base_sub_questions
                follow_qs = []
                print("[Pass mode] BASE questions only")
            else:
                base_qs = []
                follow_qs = follow_up_questions
                print("[Pass mode] FOLLOW-UP questions only")

            print(f"Base questions: {len(base_qs)}")
            print(f"Follow-ups: {len(follow_qs)}")

            # ---------- Research ----------
            final_answer = await run_pipeline(
                original_query=query,
                base_sub_questions=base_qs,
                follow_up_questions=follow_qs,
                run_id=run_id,
            )

            # ---------- Judge ----------
            print("\n[Judge] Evaluating answer...")

            judge_result = run_judge(
                original_query=query,
                sub_questions=base_sub_questions + follow_up_questions,
                final_answer=final_answer,
                run_id=run_id,
            )

            print(f"[Judge] Confidence score: {judge_result['confidence_score']}")
            # print(f"[Judge] Overall status: {judge_result['overall_status']}")

            # ---------- Termination ----------
            if judge_result["confidence_score"] >= CONFIDENCE_THRESHOLD:
                print("\n[Pipeline] Confidence threshold met. Stopping.")
                break

            # ---------- Add new follow-ups ----------
            new_followups = [
                q for q in judge_result["recommended_follow_up"]
                if q not in follow_up_questions
            ]

            print(f"[Pipeline] Adding {len(new_followups)} follow-up questions")
            follow_up_questions.extend(new_followups)

        # ====================== SAVE FINAL ANSWER ======================
        conn = get_connection()
        cur = conn.cursor()
        if final_answer:
            cur.execute(
                "UPDATE research_runs SET final_answer = ? WHERE id = ?",
                (final_answer, run_id),
            )
        conn.commit()
        conn.close()

        # ====================== FINAL OUTPUT ======================
        print("\n================ FINAL JUDGE EVALUATION ================\n")
        # print(f"Overall status: {judge_result['overall_status']}")
        print(f"Confidence score: {judge_result['confidence_score']}")
        # print("\nTermination reasoning:")
        # print(judge_result["termination_reasoning"])

        # ====================== REPORT ======================
        report_paths = generate_report(run_id)

        print("\n================ REPORT GENERATED ================\n")
        print(f"Markdown: {report_paths['markdown']}")
        print(f"HTML: {report_paths['html']}")

        with open(report_paths["markdown"], "r") as f:
            md_text = f.read()

        store_run_vector(
            run_id=run_id,
            text=f"QUERY:\n{query}\n\nFINAL REPORT:\n{md_text}"
        )

        return {
            "query": query,
            "run_id": run_id,
            "final_answer": final_answer,
            "judge_evaluation": judge_result,
        }


async def main():
    pipeline = ResearchPipeline()

    while True:
        query = input("\nEnter query (or type 'exit'): ").strip()
        if query.lower() == "exit":
            break

        await pipeline.run(query)


if __name__ == "__main__":
    asyncio.run(main())

