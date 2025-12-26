
import json
from pathlib import Path
from database import get_connection




def generate_report(run_id: str):
    conn = get_connection()
    cur = conn.cursor()

    base_output_dir = Path("outputs")
    run_output_dir = base_output_dir / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)


    # ------------------ FETCH RUN ------------------
    cur.execute("""
        SELECT *
        FROM research_runs
        WHERE id = ?
    """, (run_id,))
    run = cur.fetchone()
    if not run:
        raise ValueError("Invalid run_id")

    # ------------------ FETCH SUB-QUESTIONS ------------------
    cur.execute("""
        SELECT id, sub_question, position
        FROM sub_questions
        WHERE research_run_id = ?
        ORDER BY position
    """, (run_id,))
    sub_questions = cur.fetchall()

    sub_question_ids = [row["id"] for row in sub_questions]

    # ------------------ FETCH SOURCES ------------------
    sources = set()
    for sq_id in sub_question_ids:
        cur.execute("""
            SELECT url
            FROM search_results
            WHERE sub_question_id = ? AND url IS NOT NULL
        """, (sq_id,))
        for row in cur.fetchall():
            sources.add(row["url"])

    # ------------------ FETCH JUDGE ------------------
    cur.execute("""
        SELECT *
        FROM judge_evaluations
        WHERE research_run_id = ?
        ORDER BY evaluated_at DESC
        LIMIT 1
    """, (run_id,))
    judge = cur.fetchone()

    conn.close()

    # ------------------ MARKDOWN ------------------
    md = []
    md.append("# Research Report\n")

    md.append("## Original Query")
    md.append(run["original_query"] + "\n")

    md.append("## Query Classification")
    md.append(f"- Type: **{run['classification']}**")
    md.append(f"- Difficulty: **{run['difficulty_level']}**")
    md.append(f"- Relationship: **{run['relationship']}**\n")

    md.append("## Decomposed Sub-Questions")
    for row in sub_questions:
        md.append(f"{row['position']}. {row['sub_question']}")
    md.append("")

    md.append("## Final Answer")
    # md.append(run.get("final_answer", "Not available") + "\n")
    md.append((run["final_answer"] if run["final_answer"] else "Not available") + "\n")


    md.append("## Sources")
    for url in sorted(sources):
        md.append(f"- {url}")
    md.append("")

    if judge:
        # md.append("## Judge Evaluation")
        # md.append(f"- **Overall Status:** {judge['overall_status']}")
        # md.append(f"- **Confidence Score:** {judge['confidence_score']}\n")

        # md.append("### Coverage Assessment")
        # coverage = json.loads(judge["coverage_assessment"])
        # for q, status in coverage.items():
            # md.append(f"- {q}: {status}")
        # md.append("")

        # gaps = json.loads(judge["detected_gaps"])
        # if gaps:
        #     md.append("### Detected Gaps")
        #     for g in gaps:
        #         md.append(f"- {g}")
        #     md.append("")

        followups = json.loads(judge["recommended_follow_up"])
        if followups:
            md.append("### Recommended Follow-Up Questions")
            for fq in followups:
                md.append(f"- {fq}")
            md.append("")

        # md.append("### Termination Reasoning")
        # md.append(judge["termination_reasoning"])
        # md.append("")

    md_content = "\n".join(md)

    # md_path = OUTPUT_DIR / f"run_{run_id}.md"
    md_path = run_output_dir / f"{run_id}.md"

    md_path.write_text(md_content, encoding="utf-8")

    # ------------------ HTML (simple render) ------------------
    html = f"""
    <html>
    <head>
        <title>Research Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2, h3 {{ color: #333; }}
            ul {{ margin-left: 20px; }}
            li {{ margin-bottom: 6px; }}
            .box {{ border: 1px solid #ddd; padding: 16px; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <pre>{md_content}</pre>
    </body>
    </html>
    """

    # html_path = OUTPUT_DIR / f"run_{run_id}.html"
    html_path = run_output_dir / f"{run_id}.html"
    html_path.write_text(html, encoding="utf-8")

    return {
        "markdown": str(md_path),
        "html": str(html_path),
        "output_dir": str(run_output_dir),
    }


if __name__ == "__main__":
    rid = input("Enter run_id: ").strip()
    paths = generate_report(rid)
    print("Generated:", paths)

