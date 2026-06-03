from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import os
import json
import psycopg2

import vertexai
from vertexai.generative_models import GenerativeModel

from schema_context import NV_PLAY_DICTIONARY
from derived_metrics import DERIVED_METRICS
from cricket_rules import CRICKET_RULES

# =========================================================
# LOAD ENV VARIABLES
# =========================================================

load_dotenv()

# =========================================================
# VERTEX AI SETUP
# =========================================================

vertexai.init(
    project=os.getenv("GCP_PROJECT"),
    location=os.getenv("GCP_LOCATION", "us-central1")
)

model = GenerativeModel("gemini-2.5-flash")

# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI()

# =========================================================
# ENABLE CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# DATABASE CONNECTION
# =========================================================

def get_db_connection():

    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        connect_timeout=10
    )

# =========================================================
# REQUEST MODEL
# =========================================================

class QueryRequest(BaseModel):

    question: str

# =========================================================
# VERTEX LLM CALL
# Single function — all prompts go through here
# =========================================================

def llm(prompt: str) -> str:

    response = model.generate_content(prompt)

    return response.text.strip()

# =========================================================
# BUILD SCHEMA CONTEXT
# =========================================================

def build_schema_context(question=""):

    schema_lines = []

    for column, metadata in NV_PLAY_DICTIONARY.items():

        line = f"""
Column: {column}
Description: {metadata['description']}
Datatype: {metadata['datatype']}
Category: {metadata['category']}
Aggregation: {metadata['aggregation']}
Synonyms: {", ".join(metadata['synonyms'])}
"""
        schema_lines.append(line)

    return "\n".join(schema_lines)

# =========================================================
# BUILD METRICS CONTEXT
# =========================================================

def build_metrics_context():

    metric_lines = []

    for metric, metadata in DERIVED_METRICS.items():

        line = f"""
Metric: {metric}
Description: {metadata['description']}
Formula: {metadata['formula']}
Category: {metadata['category']}
Synonyms: {", ".join(metadata['synonyms'])}
"""
        metric_lines.append(line)

    return "\n".join(metric_lines)

# =========================================================
# BUILD RULES CONTEXT
# =========================================================

def build_rules_context():

    rules = CRICKET_RULES
    lines = []

    # Minimum sample sizes
    lines.append("MINIMUM SAMPLE SIZES:")
    for category, values in rules["minimum_sample_size"].items():
        for k, v in values.items():
            lines.append(f"  {category} -> {k}: {v}")

    # Match phases T20
    lines.append("\nMATCH PHASES (T20):")
    for phase, data in rules["match_phases"]["T20"].items():
        lines.append(
            f"  {phase}: overs {data['start_over']}-{data['end_over']} -- {data['description']}"
        )

    # Batting benchmarks T20
    lines.append("\nBATTING BENCHMARKS (T20 Strike Rate):")
    for tier, data in rules["batting_benchmarks"]["T20"]["strike_rate"].items():
        lines.append(f"  {tier}: {data}")

    # Bowling benchmarks T20
    lines.append("\nBOWLING BENCHMARKS (T20 Economy):")
    for tier, data in rules["bowling_benchmarks"]["T20"]["economy_rate"].items():
        lines.append(f"  {tier}: {data}")

    # Dismissal types
    lines.append("\nDISMISSAL TYPES:")
    for dtype, meta in rules["dismissal_types"].items():
        lines.append(f"  {dtype}: {meta['tactical_note']}")

    # Legal ball
    lines.append("\nLEGAL BALL:")
    lines.append(f"  {rules['legal_ball_rules']['analysis_note']}")

    # Keeper position
    lines.append("\nKEEPER POSITION:")
    lines.append(f"  keeper_up=TRUE: {rules['keeper_position_rules']['keeper_up']['impact']}")
    lines.append(f"  keeper_up=FALSE: {rules['keeper_position_rules']['keeper_back']['impact']}")

    # Bowling angle
    lines.append("\nBOWLING ANGLE:")
    lines.append(f"  around_the_wicket: {rules['bowling_angle_rules']['around_the_wicket']['right_arm_to_right_batter']}")

    # Free hit
    lines.append("\nFREE HIT:")
    lines.append(f"  {rules['free_hit_rules']['analysis_note']}")

    # All rounder thresholds
    lines.append("\nALL ROUNDER THRESHOLDS (T20):")
    for k, v in rules["all_rounder_thresholds"]["T20"].items():
        lines.append(f"  {k}: {v}")

    # Comparison rules
    lines.append("\nCOMPARISON RULES:")
    for rule in rules["comparison_rules"]:
        lines.append(f"  - {rule}")

    # Head to head
    lines.append("\nHEAD TO HEAD:")
    for note in rules["head_to_head_rules"]["insight_notes"]:
        lines.append(f"  - {note}")

    return "\n".join(lines)

# =========================================================
# BUILD INSIGHT RULES CONTEXT
# =========================================================

def build_insight_rules_context():

    rules = CRICKET_RULES
    lines = []

    # Performance labels
    lines.append("PERFORMANCE TIER LABELS (use these in analysis):")
    lines.append("  BATTING:")
    for tier, label in rules["performance_labels"]["batting"].items():
        lines.append(f"    {tier}: {label}")
    lines.append("  BOWLING:")
    for tier, label in rules["performance_labels"]["bowling"].items():
        lines.append(f"    {tier}: {label}")

    # Batting benchmarks full
    lines.append("\nBATTING BENCHMARKS (T20):")
    for metric, tiers in rules["batting_benchmarks"]["T20"].items():
        lines.append(f"  {metric}:")
        for tier, data in tiers.items():
            lines.append(f"    {tier}: {data}")

    # Bowling benchmarks full
    lines.append("\nBOWLING BENCHMARKS (T20):")
    for metric, tiers in rules["bowling_benchmarks"]["T20"].items():
        lines.append(f"  {metric}:")
        for tier, data in tiers.items():
            lines.append(f"    {tier}: {data}")

    # Insight rules
    lines.append("\nINSIGHT RULES (follow all of these):")
    for rule in rules["insight_rules"]:
        lines.append(f"  - {rule}")

    # Match situation context
    lines.append("\nMATCH SITUATION CONTEXT:")
    for situation, data in rules["match_situations"].items():
        lines.append(f"  {situation}: {data['insight']}")

    # Dismissal tactical notes
    lines.append("\nDISMISSAL TACTICAL CONTEXT:")
    for dtype, meta in rules["dismissal_types"].items():
        lines.append(f"  {dtype}: {meta['tactical_note']}")

    # Scoring zones
    lines.append("\nSCORING ZONE CONTEXT:")
    for zone, data in rules["scoring_zone_context"].items():
        lines.append(f"  {zone}: {data['insight']}")

    # Head to head
    lines.append("\nHEAD TO HEAD INSIGHT RULES:")
    for note in rules["head_to_head_rules"]["insight_notes"]:
        lines.append(f"  - {note}")

    return "\n".join(lines)

# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health_check():

    try:
        conn = get_db_connection()
        conn.close()
        return {"status": "ok", "database": "connected"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# =========================================================
# GEMINI TEST (VERTEX)
# =========================================================

@app.get("/gemini-test")
def gemini_test():

    return {"response": llm("Say hello from Cricket_Scorer_AI")}

# =========================================================
# CLEAN SQL
# =========================================================

def clean_sql(sql_query):

    sql_query = sql_query.replace("```sql", "").replace("```", "")
    sql_query = sql_query.strip()
    sql_query = " ".join(sql_query.split())

    sql_query = sql_query.replace("wicket = TRUE",  "wicket IS NOT NULL")
    sql_query = sql_query.replace("wicket=TRUE",    "wicket IS NOT NULL")
    sql_query = sql_query.replace("wicket = FALSE", "wicket IS NULL")
    sql_query = sql_query.replace("wicket=FALSE",   "wicket IS NULL")

    return sql_query.strip()

# =========================================================
# VALIDATE SQL
# =========================================================

def validate_sql(sql_query):

    allowed_keywords = ["SELECT", "WITH"]

    first_word = sql_query.strip().split()[0].upper()

    if first_word not in allowed_keywords:
        raise Exception("Only SELECT queries are allowed.")

    blocked_keywords = [
        "DELETE", "DROP", "UPDATE", "INSERT",
        "ALTER", "TRUNCATE", "CREATE"
    ]

    upper_sql = sql_query.upper()

    for keyword in blocked_keywords:
        if keyword in upper_sql:
            raise Exception(f"{keyword} operation not allowed.")

# =========================================================
# GENERATE QUERY PLAN
# =========================================================

# =========================================================
# CRICKET TERMINOLOGY DICTIONARY
# Maps natural language terms to SQL patterns
# =========================================================

CRICKET_TERMS = {

    # ── BATTING TERMS ──────────────────────────────────
    "opener":           "batting_position IN (1, 2)",
    "openers":          "batting_position IN (1, 2)",
    "top order":        "batting_position IN (1, 2, 3)",
    "top-order":        "batting_position IN (1, 2, 3)",
    "upper order":      "batting_position IN (1, 2, 3)",
    "middle order":     "batting_position IN (4, 5, 6)",
    "middle-order":     "batting_position IN (4, 5, 6)",
    "lower order":      "batting_position IN (7, 8, 9, 10, 11)",
    "lower-order":      "batting_position IN (7, 8, 9, 10, 11)",
    "tail":             "batting_position IN (9, 10, 11)",
    "tailender":        "batting_position IN (9, 10, 11)",
    "pinch hitter":     "batting_position <= 4 AND strike_rate > 150",
    "anchor":           "batting_position IN (3, 4) AND dot_ball_pct < 40",
    "finisher":         "batting_position IN (5, 6, 7)",

    # ── BOWLING TERMS ───────────────────────────────────
    "powerplay bowler": "over_number BETWEEN 1 AND 6",
    "death bowler":     "over_number BETWEEN 16 AND 20",
    "death overs":      "over_number BETWEEN 16 AND 20",
    "powerplay":        "over_number BETWEEN 1 AND 6",
    "middle overs":     "over_number BETWEEN 7 AND 15",
    "yorker":           "runs_total = 0 AND wicket IS NOT NULL",
    "dot ball":         "runs_total = 0 AND legal_ball = TRUE",
    "dot balls":        "runs_total = 0 AND legal_ball = TRUE",
    "boundary":         "runs_batter IN (4, 6)",
    "six":              "runs_batter = 6",
    "sixes":            "runs_batter = 6",
    "four":             "runs_batter = 4",
    "fours":            "runs_batter = 4",
    "maiden":           "SUM(runs_total) = 0 GROUP BY over_number",
    "wide":             "legal_ball = FALSE AND wide_runs > 0",
    "no ball":          "legal_ball = FALSE AND noball_runs > 0",
    "extras":           "runs_extras > 0",
    "free hit":         "free_hit = TRUE",

    # ── DISMISSAL TERMS ─────────────────────────────────
    "caught":           "wicket = 'Caught'",
    "bowled":           "wicket = 'Bowled'",
    "lbw":              "wicket = 'LBW'",
    "run out":          "wicket = 'Run Out'",
    "stumped":          "wicket = 'Stumped'",
    "hit wicket":       "wicket = 'Hit Wicket'",
    "duck":             "runs_batter = 0 AND wicket IS NOT NULL",
    "golden duck":      "runs_batter = 0 AND wicket IS NOT NULL AND ball_number = 1",
    "dismissed":        "wicket IS NOT NULL",
    "not out":          "wicket IS NULL",
    "survived":         "wicket IS NULL",

    # ── PHASE TERMS ─────────────────────────────────────
    "pp":               "over_number BETWEEN 1 AND 6",
    "pp1":              "over_number BETWEEN 1 AND 6",
    "death":            "over_number BETWEEN 16 AND 20",
    "slog overs":       "over_number BETWEEN 16 AND 20",
    "final over":       "over_number = 20",
    "last over":        "over_number = 20",
    "super over":       "over_number = 21",
    "first over":       "over_number = 1",

    # ── MATCH TERMS ─────────────────────────────────────
    "first innings":    "innings_number = 1",
    "second innings":   "innings_number = 2",
    "chase":            "innings_number = 2",
    "chasing":          "innings_number = 2",
    "defending":        "innings_number = 1",
    "batting first":    "innings_number = 1",
    "batting second":   "innings_number = 2",

    # ── FIELDING TERMS ──────────────────────────────────
    "keeper":           "keeper_up IS NOT NULL",
    "keeper up":        "keeper_up = TRUE",
    "keeper back":      "keeper_up = FALSE",
    "around wicket":    "around_the_wicket = TRUE",
    "over wicket":      "around_the_wicket = FALSE",

    # ── PERFORMANCE TERMS ───────────────────────────────
    "economy":          "ROUND((SUM(runs_total)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball = TRUE), 0)) * 6, 2)",
    "strike rate":      "ROUND((SUM(runs_batter)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball = TRUE), 0)) * 100, 2)",
    "batting average":  "ROUND(SUM(runs_batter)::numeric / NULLIF(COUNT(*) FILTER (WHERE wicket IS NOT NULL), 0), 2)",
    "bowling average":  "ROUND(SUM(runs_total)::numeric / NULLIF(COUNT(*) FILTER (WHERE wicket IS NOT NULL), 0), 2)",
    "dot percentage":   "ROUND((COUNT(*) FILTER (WHERE runs_total = 0 AND legal_ball = TRUE)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball = TRUE), 0)) * 100, 2)",
    "boundary percentage": "ROUND((COUNT(*) FILTER (WHERE runs_batter IN (4,6))::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball = TRUE), 0)) * 100, 2)",
}


def build_cricket_terms_context():

    lines = ["CRICKET TERMINOLOGY -> SQL TRANSLATION:"]

    lines.append("\nBATTING POSITIONS:")
    lines.append("  opener / openers       -> batting_position IN (1, 2)")
    lines.append("  top order / top-order  -> batting_position IN (1, 2, 3)")
    lines.append("  middle order           -> batting_position IN (4, 5, 6)")
    lines.append("  lower order / tail     -> batting_position IN (7, 8, 9, 10, 11)")
    lines.append("  finisher               -> batting_position IN (5, 6, 7)")

    lines.append("\nPHASES (T20):")
    lines.append("  powerplay / pp / pp1   -> over_number BETWEEN 1 AND 6")
    lines.append("  middle overs           -> over_number BETWEEN 7 AND 15")
    lines.append("  death / slog overs     -> over_number BETWEEN 16 AND 20")
    lines.append("  first over             -> over_number = 1")
    lines.append("  last / final over      -> over_number = 20")

    lines.append("\nDISMISSALS:")
    lines.append("  caught                 -> wicket = 'Caught'")
    lines.append("  bowled                 -> wicket = 'Bowled'")
    lines.append("  lbw                    -> wicket = 'LBW'")
    lines.append("  run out                -> wicket = 'Run Out'")
    lines.append("  stumped                -> wicket = 'Stumped'")
    lines.append("  hit wicket             -> wicket = 'Hit Wicket'")
    lines.append("  duck                   -> runs_batter = 0 AND wicket IS NOT NULL")
    lines.append("  dismissed              -> wicket IS NOT NULL")
    lines.append("  not out                -> wicket IS NULL")

    lines.append("\nDELIVERY TYPES:")
    lines.append("  dot ball               -> runs_total = 0 AND legal_ball = TRUE")
    lines.append("  boundary               -> runs_batter IN (4, 6)")
    lines.append("  six / sixes            -> runs_batter = 6")
    lines.append("  four / fours           -> runs_batter = 4")
    lines.append("  free hit               -> free_hit = TRUE")
    lines.append("  wide                   -> legal_ball = FALSE (wide)")
    lines.append("  no ball                -> legal_ball = FALSE (no-ball)")

    lines.append("\nMATCH CONTEXT:")
    lines.append("  first innings          -> innings_number = 1")
    lines.append("  second innings / chase -> innings_number = 2")
    lines.append("  batting first          -> innings_number = 1")
    lines.append("  batting second         -> innings_number = 2")

    lines.append("\nFIELDING / BOWLING ANGLE:")
    lines.append("  keeper up              -> keeper_up = TRUE")
    lines.append("  keeper back            -> keeper_up = FALSE")
    lines.append("  around the wicket      -> around_the_wicket = TRUE")
    lines.append("  over the wicket        -> around_the_wicket = FALSE")

    lines.append("\nPERFORMANCE FORMULAS:")
    lines.append("  economy rate           -> ROUND((SUM(runs_total)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*6, 2)")
    lines.append("  batting strike rate    -> ROUND((SUM(runs_batter)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*100, 2)")
    lines.append("  batting average        -> ROUND(SUM(runs_batter)::numeric / NULLIF(COUNT(*) FILTER (WHERE wicket IS NOT NULL),0), 2)")
    lines.append("  bowling average        -> ROUND(SUM(runs_total)::numeric / NULLIF(COUNT(*) FILTER (WHERE wicket IS NOT NULL),0), 2)")
    lines.append("  dot ball %             -> ROUND((COUNT(*) FILTER (WHERE runs_total=0 AND legal_ball=TRUE)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*100, 2)")
    lines.append("  boundary %             -> ROUND((COUNT(*) FILTER (WHERE runs_batter IN (4,6))::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*100, 2)")

    return "\n".join(lines)


# =========================================================
# STRIP HAVING CLAUSES (for empty result retry)
# =========================================================

import re

def strip_having_clauses(sql_query):
    """Remove HAVING clauses to relax thresholds when results are empty."""
    cleaned = re.sub(
        r'\bHAVING\b.*?(?=\bORDER\b|\bLIMIT\b|\bGROUP\b|;|$)',
        '',
        sql_query,
        flags=re.IGNORECASE | re.DOTALL
    )
    return " ".join(cleaned.split())


def results_are_empty(query_results):
    """Check if ALL queries returned zero rows."""
    return all(
        len(item.get("results", [])) == 0
        for item in query_results
    )


# =========================================================
# GENERATE QUERY PLAN
# =========================================================

def generate_query_plan(question, relax_thresholds=False):

    schema_context      = build_schema_context(question)
    metrics_context     = build_metrics_context()
    rules_context       = build_rules_context()
    cricket_terms       = build_cricket_terms_context()

    threshold_note = ""
    if relax_thresholds:
        threshold_note = """
IMPORTANT: This is a RETRY after empty results.
DO NOT use any HAVING clauses or minimum ball/innings thresholds.
Return ALL available data regardless of sample size.
Let the insight layer handle the small sample size caveat.
"""

    prompt = f"""
You are Cricket_Scorer_AI - an elite cricket data engineer and analyst with
encyclopaedic knowledge of T20, ODI, Test, and domestic cricket formats.
Your job is to translate any natural language cricket question - including
questions using cricket slang, terminology, and abbreviations - into a
precise, multi-query PostgreSQL plan.

{threshold_note}

DATABASE
Table: public.nv_play
Each row = one delivery (ball) in a cricket match.

SCHEMA
{schema_context}

DERIVED METRICS
{metrics_context}

CRICKET RULES & BENCHMARKS
{rules_context}

CRICKET TERMINOLOGY TRANSLATION
{cricket_terms}

CRITICAL COLUMN RULES

WICKETS:
  - wicket column is TEXT: 'Caught', 'Bowled', 'LBW', 'Run Out', 'Stumped', 'Hit Wicket'
  - NEVER use: wicket = TRUE or wicket = FALSE
  - dismissed    -> wicket IS NOT NULL
  - survived     -> wicket IS NULL
  - specific     -> wicket = 'Caught'
  - dismissed_batter IS NOT NULL also confirms dismissal

BOOLEANS (use TRUE / FALSE):
  - legal_ball, free_hit, around_the_wicket, keeper_up

RUNS:
  - Always SUM, never assume one row = match total
  - Strike rate  = (SUM(runs_batter) / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE), 0)) * 100
  - Economy      = (SUM(runs_total)  / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE), 0)) * 6

PLAYER/TEAM IDENTITY:
  - Use ILIKE for all name matching (handles case differences)
  - Team names: use ILIKE '%wimbledon%' style for partial matching

QUERY DESIGN RULES

1.  SELECT or WITH...SELECT only. PostgreSQL syntax.
2.  Never invent column names. Use only schema columns.
3.  Meaningful aliases on every computed column.
4.  ROUND(value::numeric, 2) for all decimals.
5.  NULLIF(..., 0) on every denominator to prevent division-by-zero.
6.  LIMIT: leaderboards/rankings -> 15 or 20. Single entity -> no LIMIT.
7.  ORDER BY results meaningfully.
8.  TEAM analysis  -> GROUP BY batting_team or bowling_team.
9.  PLAYER analysis -> GROUP BY batter or GROUP BY bowler.
10. THRESHOLDS (only apply when question is about leaderboards/all-time rankings,
    NOT when question is about a specific team, player, or match):
    - Batting leaderboards: HAVING COUNT(*) FILTER (WHERE legal_ball=TRUE) >= 30
    - Bowling leaderboards: HAVING COUNT(*) FILTER (WHERE legal_ball=TRUE) >= 24
    - NEVER apply thresholds for specific team/player/match queries
11. COMPLEX QUESTIONS -> always use MULTIPLE queries:
    Query 1: Overall / career summary stats
    Query 2: Phase-wise breakdown (powerplay / middle / death)
    Query 3: Comparison, ranking, or opposition breakdown
    Query 4: (optional) Dismissal types or special situations
12. Always include contextual columns: player name, team, balls_faced/balls_bowled, matches.
13. COUNT(DISTINCT match_id) AS matches_played for match counts.
14. For batting_position use it directly if available, else use row_number logic.

COMPLEX QUERY PATTERNS

PARTNERSHIP ANALYSIS:
  WITH partnerships AS (
    SELECT batter, non_striker, SUM(runs_batter + runs_extras) AS runs,
           COUNT(*) FILTER (WHERE legal_ball=TRUE) AS balls
    FROM public.nv_play
    GROUP BY batter, non_striker
  )

PHASE COMPARISON (single player):
  SELECT
    CASE WHEN over_number BETWEEN 1  AND 6  THEN 'Powerplay'
         WHEN over_number BETWEEN 7  AND 15 THEN 'Middle'
         WHEN over_number BETWEEN 16 AND 20 THEN 'Death'
    END AS phase,
    SUM(runs_batter) AS runs,
    ROUND((SUM(runs_batter)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*100,2) AS sr
  FROM public.nv_play
  WHERE batter ILIKE '%name%'
  GROUP BY phase

HEAD-TO-HEAD (batter vs bowler):
  SELECT batter, bowler,
    COUNT(*) FILTER (WHERE legal_ball=TRUE) AS balls,
    SUM(runs_batter) AS runs,
    COUNT(*) FILTER (WHERE wicket IS NOT NULL) AS dismissals,
    ROUND((SUM(runs_batter)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*100,2) AS sr
  FROM public.nv_play
  WHERE batter ILIKE '%x%' AND bowler ILIKE '%y%'

OVER-BY-OVER ANALYSIS:
  SELECT over_number,
    ROUND(AVG(runs_total),2) AS avg_runs_per_ball,
    COUNT(*) FILTER (WHERE wicket IS NOT NULL) AS wickets
  FROM public.nv_play
  GROUP BY over_number ORDER BY over_number

MATCH SUMMARY:
  SELECT match_id, batting_team, bowling_team,
    SUM(runs_total) AS total_runs,
    COUNT(*) FILTER (WHERE wicket IS NOT NULL) AS wickets,
    ROUND((SUM(runs_total)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*6,2) AS run_rate
  FROM public.nv_play
  GROUP BY match_id, batting_team, bowling_team, innings_number

ANALYSIS TYPE DETECTION

Detect intent and plan queries accordingly:

INDIVIDUAL BATTING    -> career stats + phase breakdown + dismissal types
INDIVIDUAL BOWLING    -> career stats + phase breakdown + dismissal methods + head-to-head
TEAM BATTING          -> team totals + phase breakdown + top contributors
TEAM BOWLING          -> team economy + wicket takers + phase breakdown
HEAD TO HEAD          -> batter vs bowler balls/runs/dismissals/SR
PARTNERSHIP           -> runs/balls together, run rate during partnership
MATCH SUMMARY         -> innings total, run rate, wickets, top scorer/bowler
LEADERBOARD           -> ranked list with multiple metrics + thresholds
PHASE ANALYSIS        -> powerplay / middle / death comparison
DISMISSAL ANALYSIS    -> type breakdown + which bowler causes what dismissal
BATTING POSITION      -> top order / middle order / tail performance
OVER ANALYSIS         -> over-by-over run rates, economy, wickets
VENUE / CONDITION     -> ground-wise, keeper position, bowling angle impact
SPECIAL SITUATIONS    -> free hits, super overs, last over, chasing analysis

THRESHOLD RULES (CRITICAL)

DO use HAVING thresholds ONLY when:
  - Question is an open leaderboard (top N players across entire dataset)
  - No specific team, player, or match is named

DO NOT use HAVING thresholds when:
  - A specific team is named (e.g. Wimbledon, MI, CSK)
  - A specific player is named
  - A specific match or date is mentioned
  - The question uses batting positions (top order, openers etc.)
  - The word "all" players of a team is used

OUTPUT FORMAT

Return ONLY valid JSON. No markdown. No explanation. No extra text.

{{
  "analysis_type": "single | multi",
  "intent": "precise one-line description of what user is asking",
  "threshold_applied": true or false,
  "queries": [
    {{
      "name": "descriptive_snake_case_name",
      "purpose": "exactly what this query computes",
      "sql": "SELECT ..."
    }}
  ]
}}

USER QUESTION
{question}
"""

    raw = llm(prompt).replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw)
    except Exception:
        return {
            "analysis_type": "single",
            "threshold_applied": False,
            "queries": [
                {
                    "name":    "fallback_query",
                    "purpose": "Fallback cricket analysis",
                    "sql":     generate_fallback_sql(question)
                }
            ]
        }


# =========================================================
# FALLBACK SQL
# =========================================================

def generate_fallback_sql(question):

    schema_context  = build_schema_context(question)
    cricket_terms   = build_cricket_terms_context()

    prompt = f"""
You are a PostgreSQL expert for cricket ball-by-ball data.

TABLE: public.nv_play (each row = one delivery)

SCHEMA:
{schema_context}

CRICKET TERMINOLOGY:
{cricket_terms}

STRICT RULES:
1.  Return ONLY the raw SQL — no explanation, no markdown, no backticks.
2.  SELECT only.
3.  PostgreSQL syntax.
4.  Only schema columns — never invent columns.
5.  wicket is TEXT — wicket IS NOT NULL for dismissals.
6.  legal_ball, free_hit, around_the_wicket, keeper_up are BOOLEAN.
7.  ROUND(value::numeric, 2) for all decimals.
8.  NULLIF(..., 0) on all denominators.
9.  Meaningful aliases.
10. LIMIT 20 for open rankings.
11. ILIKE for name matching.
12. NO HAVING clauses — return all available data.
13. Translate cricket terms using the terminology guide above.

QUESTION:
{question}
"""

    sql_query = clean_sql(llm(prompt))

    if not sql_query.endswith(";"):
        sql_query += ";"

    return sql_query

# =========================================================
# EXECUTE SQL
# =========================================================

def execute_sql(sql_query):

    validate_sql(sql_query)

    conn   = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(sql_query)
    rows    = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]

    results = []

    for row in rows:
        row_dict = {}
        for i, value in enumerate(row):
            if isinstance(value, float):
                value = round(value, 2)
            row_dict[columns[i]] = value
        results.append(row_dict)

    cursor.close()
    conn.close()

    return results

# =========================================================
# EXECUTE QUERY PLAN
# =========================================================

def execute_query_plan(query_plan):

    all_results = []
    all_sql     = []

    for q in query_plan.get("queries", []):

        sql_query = clean_sql(q["sql"])

        if not sql_query.endswith(";"):
            sql_query += ";"

        try:
            results = execute_sql(sql_query)

            all_results.append({
                "query_name": q["name"],
                "purpose":    q["purpose"],
                "sql":        sql_query,
                "results":    results
            })

            all_sql.append(sql_query)

        except Exception as e:
            all_results.append({
                "query_name": q["name"],
                "purpose":    q["purpose"],
                "sql":        sql_query,
                "results":    [],
                "error":      str(e)
            })

    return all_results, all_sql


# =========================================================
# EXECUTE QUERY PLAN WITH RETRY
# If all results empty -> strip HAVING -> retry once
# =========================================================

def execute_query_plan_with_retry(query_plan, question):

    all_results, all_sql = execute_query_plan(query_plan)

    # ── If all queries returned 0 rows, retry without thresholds ──
    if results_are_empty(all_results):

        # First try: just strip HAVING clauses from existing SQL
        retry_plan = {
            "analysis_type": query_plan.get("analysis_type", "single"),
            "queries": []
        }

        for q in query_plan.get("queries", []):
            relaxed_sql = strip_having_clauses(q["sql"])
            retry_plan["queries"].append({
                "name":    q["name"] + "_relaxed",
                "purpose": q["purpose"] + " (thresholds relaxed)",
                "sql":     relaxed_sql
            })

        retry_results, retry_sql = execute_query_plan(retry_plan)

        # If still empty -> regenerate plan without thresholds
        if results_are_empty(retry_results):

            fresh_plan   = generate_query_plan(question, relax_thresholds=True)
            fresh_results, fresh_sql = execute_query_plan(fresh_plan)

            return fresh_results, fresh_sql, True  # True = thresholds were relaxed

        return retry_results, retry_sql, True

    return all_results, all_sql, False

# =========================================================
# FORMAT RESULTS
# =========================================================

def format_results(results):

    if not results:
        return "No results found."

    headers    = list(results[0].keys())
    header_row = " | ".join(headers)
    separator  = "-" * len(header_row)

    table = [header_row, separator]

    for row in results:
        table.append(" | ".join(str(v) for v in row.values()))

    return "\n".join(table)

# =========================================================
# GENERATE CRICKET INSIGHT
# =========================================================

def generate_cricket_insight(question, query_results, small_sample=False):

    compact_data          = json.dumps(query_results, default=str)[:12000]
    insight_rules_context = build_insight_rules_context()

    small_sample_note = ""
    if small_sample:
        small_sample_note = """
NOTE: The data returned has a small sample size (minimum thresholds were relaxed
to retrieve results). Mention this naturally in your analysis — e.g. "based on
limited data" or "small sample caveat applies" — but still provide the best
analysis possible from the available numbers. Do not refuse to analyse.
"""

    prompt = f"""
You are Cricket_Scorer_AI - a world-class cricket analyst combining the
tactical depth of a coaching analyst, the storytelling of a top commentator,
and the precision of a statistician.

{small_sample_note}

USER QUESTION
{question}

DATA FROM DATABASE
{compact_data}

CRICKET RULES, BENCHMARKS & INSIGHT GUIDELINES
{insight_rules_context}

YOUR TASK

Produce a rich, structured cricket analysis report. Follow this layout exactly:

---

## 🏏 [Write a compelling headline summarising the key finding]

### 📊 Key Numbers
Present the most important statistics as a clean markdown table.
Include columns like Player/Team, Metric 1, Metric 2, Metric 3 etc.
Always show actual numbers from the data.

### 🔍 Analysis

Write 3-5 paragraphs covering:

**Overall Performance**
Summarise what the numbers say at a high level. Quote exact figures.
Classify performance using tier labels (elite/excellent/good/average/poor).

**Strengths**
What is the player/team doing exceptionally well?
Reference specific stats. Use cricket-specific language.

**Weaknesses / Vulnerabilities**
Where are the cracks? Reference numbers.

**Tactical Insights**
Phase-wise, around/over wicket, keeper position, scoring zones.
Only mention what the data supports.

**Context & Comparisons** (if multiple players/teams)
Compare performers directly. What separates the top from the rest?

### 📈 Standout Moments / Records
2-4 bullet points highlighting the most impressive stat or anomaly.

### 💡 Verdict
One punchy paragraph (3-5 sentences). Expert panel verdict tone.

---

STRICT RULES

- Always cite exact numbers (e.g. SR of 187.4, economy of 6.23)
- Use cricket terminology naturally
- Render tables in proper markdown with | pipes and --- separators
- Bold key player/team names using **Name**
- If multiple players/teams: compare explicitly, not in isolation
- If phase-wise data exists: dedicate a paragraph to it
- Classify every key metric against benchmark tiers
- If data is empty: acknowledge gracefully
- Under 700 words unless data demands more
- Never mention SQL, database, queries, or technical implementation
- Never fabricate stats not in the data
- Never use filler phrases like it is worth noting
"""

    return llm(prompt)

# =========================================================
# GENERATE CHART CONFIG
# =========================================================

def generate_chart_config(question, query_results):

    compact_data = json.dumps(query_results, default=str)[:8000]

    prompt = f"""
You are a cricket data visualization expert.

Decide which chart type best communicates the insight.
If no chart adds value, return null.

CHART TYPE DECISION GUIDE

bar
  WHEN: Side-by-side multi-metric comparison across players/teams/phases
  EXAMPLES: Compare runs, SR, average for top 5 batters
  REQUIRES: Multiple y_keys, categorical x_axis
  MAX ROWS: 10

bar_colored
  WHEN: Single metric leaderboard - each entry gets a unique color
  EXAMPLES: Top 10 run scorers, top wicket takers, economy comparison
  REQUIRES: Exactly 1 y_key, categorical x_axis
  MAX ROWS: 15

line
  WHEN: Trends or progressions over a sequence
  EXAMPLES: Run rate over-by-over, economy trend across matches
  REQUIRES: Ordered/sequential x_axis
  MAX ROWS: 25

area
  WHEN: Volume or accumulation trends
  EXAMPLES: Cumulative runs over overs, total wickets over time
  REQUIRES: Ordered/sequential x_axis
  MAX ROWS: 25

pie
  WHEN: Distribution or share of a whole
  EXAMPLES: Dismissal type breakdown, run distribution 4s vs 6s vs singles
  REQUIRES: Exactly 1 y_key, x_key is label/category
  MAX ROWS: 8

radar
  WHEN: Multi-dimensional profile of 2-4 players across 4-6 metrics
  EXAMPLES: Bowler profile across economy, wickets, dot%, SR, average
  REQUIRES: Multiple y_keys, 2-4 rows only
  MAX ROWS: 4

SELECTION RULES

1. distribution / breakdown / share / percentage of total -> pie
2. trend / over time / over-by-over / progression         -> line or area
3. compare / vs / top N + ONE metric                      -> bar_colored
4. compare / vs / top N + MULTIPLE metrics                -> bar
5. profile / all-round / 2-4 players x many dimensions    -> radar
6. Only 1 row of data                                     -> null
7. No numeric columns                                     -> null
8. Single stat lookup                                     -> null

DATA RULES

1. x_key must be a STRING column
2. y_keys must be NUMERIC columns only
3. All values must be actual numbers, not strings
4. Float values rounded to 2 decimal places
5. title must be short and specific
6. subtitle is optional one-line context

RETURN FORMAT

{{
  "chart_type": "bar | bar_colored | line | area | pie | radar",
  "title": "Short descriptive title",
  "subtitle": "Optional one-line context",
  "x_key": "column_used_as_x_axis_or_label",
  "y_keys": ["metric1", "metric2"],
  "data": [
    {{ "category": "Player A", "metric1": 45.2, "metric2": 132.5 }},
    {{ "category": "Player B", "metric1": 38.1, "metric2": 119.0 }}
  ]
}}

If no chart adds value return exactly:
null

No markdown. No explanation. Only JSON or null.

USER QUESTION
{question}

DATABASE RESULTS
{compact_data}
"""

    raw = llm(prompt).replace("```json", "").replace("```", "").strip()

    if raw.lower() == "null":
        return None

    try:
        return json.loads(raw)
    except Exception:
        return None

# =========================================================
# GENERATE SQL ROUTE
# =========================================================

@app.post("/generate-sql")
def generate_sql(req: QueryRequest):

    try:
        return {
            "question":   req.question,
            "query_plan": generate_query_plan(req.question)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# =========================================================
# MAIN ASK ROUTE
# =========================================================

@app.post("/ask")
def ask_question(req: QueryRequest):

    try:

        # ─────────────────────────────────────────
        # STEP 1 — Generate SQL query plan
        # ─────────────────────────────────────────

        query_plan = generate_query_plan(req.question)

        # ─────────────────────────────────────────
        # STEP 2 — Execute with auto-retry
        # Retries without HAVING if results empty
        # ─────────────────────────────────────────

        query_results, sql_queries, thresholds_relaxed = execute_query_plan_with_retry(
            query_plan,
            req.question
        )

        # ─────────────────────────────────────────
        # STEP 3 — Format plain-text tables
        # ─────────────────────────────────────────

        tables = [
            {
                "query_name": item["query_name"],
                "purpose":    item["purpose"],
                "table":      format_results(item["results"])
            }
            for item in query_results
        ]

        # ─────────────────────────────────────────
        # STEP 4 — Generate chart config
        # ─────────────────────────────────────────

        chart_config = generate_chart_config(req.question, query_results)

        # ─────────────────────────────────────────
        # STEP 5 — Generate cricket insight
        # Pass relaxed flag so insight mentions small sample if needed
        # ─────────────────────────────────────────

        insight = generate_cricket_insight(
            req.question,
            query_results,
            small_sample=thresholds_relaxed
        )

        # ─────────────────────────────────────────
        # STEP 6 — Return full response
        # ─────────────────────────────────────────

        return {
            "question":           req.question,
            "analysis_type":      query_plan.get("analysis_type", "single"),
            "thresholds_relaxed": thresholds_relaxed,
            "sql_queries":        sql_queries,
            "results":            query_results,
            "tables":             tables,
            "chart_config":       chart_config,
            "insight":            insight
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}