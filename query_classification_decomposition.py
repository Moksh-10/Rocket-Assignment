from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import json
import os
import uuid
from database import get_connection, init_db
from vector_db import retrieve_context

init_db()
load_dotenv()
def normalize_sub_questions(sub_questions):
    normalized = []

    for q in sub_questions:
        if isinstance(q, dict):
            for k, v in q.items():
                normalized.append(f"[{k.upper()}] {v}")
        else:
            normalized.append(str(q))

    return normalized

# ====================== 1. EXISTING CLASSIFICATION ======================
class QueryType(BaseModel):
    category: Literal["factual", "speculative", "ambiguous"] = Field(
        description="Categorisation of the user query"
    )

classification_parser = JsonOutputParser(pydantic_object=QueryType)

classification_prompt = PromptTemplate(
    template="""Classify the following query as 'factual', 'speculative', or 'ambiguous'.

DEFINITIONS:
- **Factual**: Has a single, verifiable, objective answer. Can be proven right/wrong with evidence.
  Examples: "What is the capital of France?", "How many planets are in our solar system?"

- **Speculative**: Involves opinions, predictions, subjective judgments, future possibilities, or what-if scenarios.
  Examples: "Will AI replace all jobs?", "Is pineapple good on pizza?"

- **Ambiguous**: Contains multiple topics, mixes factual and speculative elements, or is unclear/vague.
  Examples: "Explain WWII and what if Germany won", "Quantum computing and its ethical implications"

DECISION RULES:
1. If query has BOTH verifiable facts AND opinions/predictions → 'ambiguous'
2. If query is vague or could be interpreted multiple ways → 'ambiguous'
3. If query is purely about preferences/future/predictions → 'speculative'
4. If query is purely about verifiable facts/current reality → 'factual'

QUERY: {query}

Respond ONLY with a JSON object containing the "category" key.
{format_instructions}
""",
    input_variables=["query"],
    partial_variables={"format_instructions": classification_parser.get_format_instructions()},
)

# ====================== 2. NEW DECOMPOSITION MODELS ======================
class DecomposedQuery(BaseModel):
    """Model for decomposed query components"""
    sub_questions: List[str] = Field(
        description="Specific, searchable sub-questions. Each should be answerable independently."
    )
    primary_intent: str = Field(
        description="Main goal or purpose of the original query"
    )
    relationship: Literal["sequential", "parallel", "hierarchical", "comparative"] = Field(
        description="How sub-questions relate: sequential (steps/timeline), parallel (different aspects), hierarchical (main + details), comparative (compare/contrast)"
    )
    difficulty_level: Literal["simple", "moderate", "complex"] = Field(
        description="Overall complexity of the query"
    )

decomposition_parser = JsonOutputParser(pydantic_object=DecomposedQuery)

decomposition_prompt = PromptTemplate(
    template="""Decompose the query into specific searchable sub-questions.

ORIGINAL QUERY: {query}
CLASSIFICATION: {query_type}

CRITICAL DECOMPOSITION RULES:
1. If query contains BOTH factual AND speculative elements, create SEPARATE sub-questions for each
2. For speculative parts (what-if, could-have, alternative scenarios), frame as hypothetical questions
3. For factual parts, focus on verifiable events, data, timelines
4. Ensure ALL parts of the original query are covered in sub-questions

SPECIAL HANDLING FOR MIXED QUERIES:
- "What happened... and what if..." → Create factual sub-questions for "what happened" AND speculative sub-questions for "what if"
- "Explain X and predict Y" → Technical sub-questions for X + future-oriented sub-questions for Y
- "Compare A vs B and which is better" → Factual comparison sub-questions + opinion/value sub-questions

EXAMPLES OF GOOD DECOMPOSITION:

1. FACTUAL QUERY EXAMPLE:
Query: "What is the greenhouse effect and how does it cause global warming?"
Good sub-questions:
- "What is the scientific definition of the greenhouse effect?"
- "Which gases contribute most to the greenhouse effect?"
- "How does increased greenhouse gas concentration raise global temperatures?"
- "What is the evidence linking human activity to enhanced greenhouse effect?"

2. SPECULATIVE QUERY EXAMPLE:
Query: "Will AI replace most human jobs by 2040?"
Good sub-questions:
- "What are current AI capabilities in automating different job types?"
- "Which industries are most vulnerable to AI job displacement?"
- "What new types of jobs might AI create by 2040?"
- "What do economists predict about AI's impact on employment rates?"

3. AMBIGUOUS/MIXED QUERY EXAMPLE:
Query: "How did WW2 end and what if atomic bombs weren't dropped?"
Good sub-questions:
- "What were the key events leading to Japan's surrender in 1945?" [FACTUAL]
- "What was the role of atomic bombs in ending WW2?" [FACTUAL]
- "What alternative strategies did Allies consider besides atomic bombs?" [FACTUAL]
- "How might the war have ended differently without atomic bombs?" [SPECULATIVE]
- "What would be the geopolitical consequences if Japan hadn't surrendered in 1945?" [SPECULATIVE]

4. AMBIGUOUS/COMPARATIVE QUERY EXAMPLE:
Query: "Compare solar vs wind energy: which is more cost-effective?"
Good sub-questions:
- "What are the installation costs of solar panels vs wind turbines?" [FACTUAL]
- "How do maintenance costs compare between solar and wind systems?" [FACTUAL]
- "What is the energy output efficiency of solar vs wind in different regions?" [FACTUAL]
- "Which energy source has better long-term cost projections?" [SPECULATIVE/ANALYTICAL]

Now decompose this query:

{format_instructions}
""",
    input_variables=["query", "query_type"],
    partial_variables={"format_instructions": decomposition_parser.get_format_instructions()},
)

# ====================== 3. INITIALIZE MODELS ======================
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
model = init_chat_model("groq:llama-3.1-8b-instant")

# Create chains
classification_chain = classification_prompt | model | classification_parser
decomposition_chain = decomposition_prompt | model | decomposition_parser

# ====================== 4. COMPLETE PIPELINE FUNCTION ======================
def research_decomposition_pipeline(query: str) -> dict:
    """Complete classification and decomposition pipeline."""

    run_id = str(uuid.uuid4())

    conn = get_connection()
    cur = conn.cursor()

    # ====================== CACHE CHECK ======================
    cur.execute("""
        SELECT id, classification, primary_intent, relationship, difficulty_level
        FROM research_runs
        WHERE original_query = ?
        ORDER BY created_at DESC
        LIMIT 1
    """, (query,))

    cached_run = cur.fetchone()

    if cached_run:
        cur.execute("""
            SELECT sub_question
            FROM sub_questions
            WHERE research_run_id = ?
            ORDER BY position
        """, (cached_run["id"],))

        cached_sub_questions = [row["sub_question"] for row in cur.fetchall()]
        conn.close()

        return {
            "run_id": cached_run["id"],
            "original_query": query,
            "classification": cached_run["classification"],
            "decomposition": {
                "sub_questions": cached_sub_questions,
                "primary_intent": cached_run["primary_intent"],
                "relationship": cached_run["relationship"],
                "difficulty_level": cached_run["difficulty_level"],
            }
        }

    conn.close()

    
    print(f"\n{'='*60}")
    print(f"PROCESSING QUERY: {query}")
    print('='*60)
    
    # Step 1: Classify the query
    print("\n1. CLASSIFICATION PHASE")
    print("-" * 40)
    try:
        #classification_result = classification_chain.invoke({"query": query})
        memory_context = retrieve_context(query)
        classification_result = classification_chain.invoke({
            "query": f"{query}\n\nPREVIOUS CONTEXT:\n{memory_context}"
        })

        query_type = classification_result["category"]
        print(f"✓ Query classified as: {query_type}")
    except Exception as e:
        print(f"✗ Classification failed: {e}")
        # Fallback: treat as ambiguous if classification fails
        query_type = "ambiguous"
        print(f"  Using fallback type: {query_type}")
    
    # Step 2: Decompose the query
    print(f"\n2. DECOMPOSITION PHASE (type: {query_type})")
    print("-" * 40)
    try:
        # decomposition_result = decomposition_chain.invoke({
        #     "query": query,
        #     "query_type": query_type
        # })

        decomposition_result = decomposition_chain.invoke({
            "query": f"{query}\n\nPREVIOUS CONTEXT:\n{memory_context}",
            "query_type": query_type
        })
        decomposition_result["sub_questions"] = normalize_sub_questions(
            decomposition_result["sub_questions"]
        )

        
        # Display results
        print(f"✓ Primary intent: {decomposition_result['primary_intent']}")
        print(f"✓ Relationship type: {decomposition_result['relationship']}")
        print(f"✓ Difficulty: {decomposition_result['difficulty_level']}")
        
        print(f"\n✓ Generated {len(decomposition_result['sub_questions'])} sub-questions:")
        for i, sub_q in enumerate(decomposition_result['sub_questions'], 1):
            print(f"  {i}. {sub_q}")
            
    except Exception as e:
        print(f"✗ Decomposition failed: {e}")
        # Fallback: create simple decomposition
        decomposition_result = {
            "sub_questions": [query],  # Use original query as fallback
            "primary_intent": "Answer the query directly",
            "relationship": "parallel",
            "difficulty_level": "moderate"
        }
        print(f"  Using fallback: single sub-question")

    conn = get_connection()
    cur = conn.cursor()

    # Insert research run
    cur.execute("""
    INSERT INTO research_runs (
        id,
        original_query,
        classification,
        primary_intent,
        relationship,
        difficulty_level
    ) VALUES (?, ?, ?, ?, ?, ?)
    """, (
        run_id,
        query,
        query_type,
        decomposition_result["primary_intent"],
        decomposition_result["relationship"],
        decomposition_result["difficulty_level"]
    ))

    # Insert sub-questions
    for idx, sub_q in enumerate(decomposition_result["sub_questions"], start=1):
        cur.execute("""
        INSERT INTO sub_questions (
            research_run_id,
            sub_question,
            position
        ) VALUES (?, ?, ?)
        """, (run_id, sub_q, idx))

    conn.commit()
    conn.close()
    
    # Return complete analysis
    return {
        "run_id": run_id,
        "original_query": query,
        "classification": query_type,
        "decomposition": decomposition_result
    }

# ====================== 5. TEST FUNCTION ======================
def test_decomposition_pipeline():
    """Test the decomposition pipeline with various queries."""
    
    test_queries = [
        # Factual
        "What are the main causes of climate change and their effects?",
        
        # Speculative  
        "Will electric vehicles completely replace gasoline cars by 2040?",
        
        # Ambiguous (mixed)
        "How did World War 2 end, and what would have happened if the atomic bombs weren't dropped on Japan?",
        
        # Comparative
        "Compare remote work vs office work in terms of productivity and employee satisfaction",
        
        # Complex technical
        "Explain how blockchain technology works and its potential applications in healthcare"
    ]
    
    results = []
    
    for query in test_queries:
        result = research_decomposition_pipeline(query)
        results.append(result)
        
        # Pretty print the full result
        print(f"\n📋 FULL RESULT FOR: {query}...")
        print(json.dumps(result, indent=2))
        print("\n" + "="*60 + "\n")
    
    return results

if __name__ == "__main__":
    results = test_decomposition_pipeline()

    with open("decomposition_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Results saved to decomposition_results.json")

