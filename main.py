from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import os
import re
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

    lines.append("MINIMUM SAMPLE SIZES:")
    for category, values in rules["minimum_sample_size"].items():
        for k, v in values.items():
            lines.append(f"  {category} -> {k}: {v}")

    lines.append("\nMATCH PHASES (T20):")
    for phase, data in rules["match_phases"]["T20"].items():
        lines.append(
            f"  {phase}: overs {data['start_over']}-{data['end_over']} -- {data['description']}"
        )

    lines.append("\nBATTING BENCHMARKS (T20 Strike Rate):")
    for tier, data in rules["batting_benchmarks"]["T20"]["strike_rate"].items():
        lines.append(f"  {tier}: {data}")

    lines.append("\nBOWLING BENCHMARKS (T20 Economy):")
    for tier, data in rules["bowling_benchmarks"]["T20"]["economy_rate"].items():
        lines.append(f"  {tier}: {data}")

    lines.append("\nDISMISSAL TYPES:")
    for dtype, meta in rules["dismissal_types"].items():
        lines.append(f"  {dtype}: {meta['tactical_note']}")

    lines.append("\nLEGAL BALL:")
    lines.append(f"  {rules['legal_ball_rules']['analysis_note']}")

    lines.append("\nKEEPER POSITION:")
    lines.append(f"  keeper_up=TRUE: {rules['keeper_position_rules']['keeper_up']['impact']}")
    lines.append(f"  keeper_up=FALSE: {rules['keeper_position_rules']['keeper_back']['impact']}")

    lines.append("\nBOWLING ANGLE:")
    lines.append(f"  around_the_wicket: {rules['bowling_angle_rules']['around_the_wicket']['right_arm_to_right_batter']}")

    lines.append("\nFREE HIT:")
    lines.append(f"  {rules['free_hit_rules']['analysis_note']}")

    lines.append("\nALL ROUNDER THRESHOLDS (T20):")
    for k, v in rules["all_rounder_thresholds"]["T20"].items():
        lines.append(f"  {k}: {v}")

    lines.append("\nCOMPARISON RULES:")
    for rule in rules["comparison_rules"]:
        lines.append(f"  - {rule}")

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

    lines.append("PERFORMANCE TIER LABELS (use these in analysis):")
    lines.append("  BATTING:")
    for tier, label in rules["performance_labels"]["batting"].items():
        lines.append(f"    {tier}: {label}")
    lines.append("  BOWLING:")
    for tier, label in rules["performance_labels"]["bowling"].items():
        lines.append(f"    {tier}: {label}")

    lines.append("\nBATTING BENCHMARKS (T20):")
    for metric, tiers in rules["batting_benchmarks"]["T20"].items():
        lines.append(f"  {metric}:")
        for tier, data in tiers.items():
            lines.append(f"    {tier}: {data}")

    lines.append("\nBOWLING BENCHMARKS (T20):")
    for metric, tiers in rules["bowling_benchmarks"]["T20"].items():
        lines.append(f"  {metric}:")
        for tier, data in tiers.items():
            lines.append(f"    {tier}: {data}")

    lines.append("\nINSIGHT RULES (follow all of these):")
    for rule in rules["insight_rules"]:
        lines.append(f"  - {rule}")

    lines.append("\nMATCH SITUATION CONTEXT:")
    for situation, data in rules["match_situations"].items():
        lines.append(f"  {situation}: {data['insight']}")

    lines.append("\nDISMISSAL TACTICAL CONTEXT:")
    for dtype, meta in rules["dismissal_types"].items():
        lines.append(f"  {dtype}: {meta['tactical_note']}")

    lines.append("\nSCORING ZONE CONTEXT:")
    for zone, data in rules["scoring_zone_context"].items():
        lines.append(f"  {zone}: {data['insight']}")

    lines.append("\nHEAD TO HEAD INSIGHT RULES:")
    for note in rules["head_to_head_rules"]["insight_notes"]:
        lines.append(f"  - {note}")

    return "\n".join(lines)

# =========================================================
# CRICKET TERMINOLOGY DICTIONARY
# =========================================================

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
    lines.append("  wide                   -> legal_ball = FALSE (wide type)")
    lines.append("  no ball                -> legal_ball = FALSE (no-ball type)")

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
# DATE / TIME CONTEXT
# Critical for year, season, month queries
# =========================================================

def build_date_context():

    lines = []

    lines.append("DATE & TIME COLUMN RULES:")
    lines.append("")
    lines.append("PRIMARY DATE COLUMN:")
    lines.append("  match_date   -> DATE type (e.g. '2023-07-15')")
    lines.append("  Use match_date for ALL date, year, season, month filtering")
    lines.append("")
    lines.append("YEAR EXTRACTION:")
    lines.append("  EXTRACT(YEAR FROM match_date)          -> integer year (2021, 2022, 2023...)")
    lines.append("  DATE_PART('year', match_date)          -> same as above")
    lines.append("  TO_CHAR(match_date, 'YYYY')            -> year as text string")
    lines.append("")
    lines.append("MONTH EXTRACTION:")
    lines.append("  EXTRACT(MONTH FROM match_date)         -> integer month (1-12)")
    lines.append("  TO_CHAR(match_date, 'Month')           -> month name (January...)")
    lines.append("  TO_CHAR(match_date, 'Mon')             -> short month (Jan, Feb...)")
    lines.append("")
    lines.append("SEASON / YEAR GROUPING PATTERNS:")
    lines.append("  GROUP BY EXTRACT(YEAR FROM match_date)")
    lines.append("  GROUP BY TO_CHAR(match_date, 'YYYY')")
    lines.append("  ORDER BY EXTRACT(YEAR FROM match_date)")
    lines.append("")
    lines.append("DATE FILTERING PATTERNS:")
    lines.append("  Specific year:    WHERE EXTRACT(YEAR FROM match_date) = 2023")
    lines.append("  Year range:       WHERE EXTRACT(YEAR FROM match_date) BETWEEN 2021 AND 2023")
    lines.append("  Specific month:   WHERE EXTRACT(MONTH FROM match_date) = 6")
    lines.append("  Date range:       WHERE match_date BETWEEN '2023-01-01' AND '2023-12-31'")
    lines.append("  After date:       WHERE match_date >= '2023-01-01'")
    lines.append("  Recent N years:   WHERE EXTRACT(YEAR FROM match_date) >= EXTRACT(YEAR FROM CURRENT_DATE) - N")
    lines.append("")
    lines.append("YEARLY PERFORMANCE PATTERN (most common):")
    lines.append("  SELECT")
    lines.append("    EXTRACT(YEAR FROM match_date) AS year,")
    lines.append("    COUNT(DISTINCT match_id)       AS matches,")
    lines.append("    SUM(runs_batter)               AS total_runs,")
    lines.append("    ROUND((SUM(runs_batter)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*100, 2) AS strike_rate,")
    lines.append("    ROUND(SUM(runs_batter)::numeric / NULLIF(COUNT(*) FILTER (WHERE wicket IS NOT NULL),0), 2) AS avg")
    lines.append("  FROM public.nv_play")
    lines.append("  WHERE batter ILIKE '%player%'")
    lines.append("  GROUP BY EXTRACT(YEAR FROM match_date)")
    lines.append("  ORDER BY year")
    lines.append("")
    lines.append("TEAM YEARLY PERFORMANCE PATTERN:")
    lines.append("  SELECT")
    lines.append("    EXTRACT(YEAR FROM match_date)  AS year,")
    lines.append("    COUNT(DISTINCT match_id)        AS matches,")
    lines.append("    SUM(runs_total)                 AS total_runs,")
    lines.append("    COUNT(*) FILTER (WHERE wicket IS NOT NULL) AS wickets_lost,")
    lines.append("    ROUND((SUM(runs_total)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*6, 2) AS run_rate")
    lines.append("  FROM public.nv_play")
    lines.append("  WHERE batting_team ILIKE '%team%'")
    lines.append("  GROUP BY EXTRACT(YEAR FROM match_date)")
    lines.append("  ORDER BY year")
    lines.append("")
    lines.append("NATURAL LANGUAGE -> SQL DATE MAPPING:")
    lines.append("  'every year'          -> GROUP BY EXTRACT(YEAR FROM match_date) ORDER BY year")
    lines.append("  'year by year'        -> GROUP BY EXTRACT(YEAR FROM match_date) ORDER BY year")
    lines.append("  'each season'         -> GROUP BY EXTRACT(YEAR FROM match_date) ORDER BY year")
    lines.append("  'annual'              -> GROUP BY EXTRACT(YEAR FROM match_date)")
    lines.append("  'in 2023'             -> WHERE EXTRACT(YEAR FROM match_date) = 2023")
    lines.append("  'last year'           -> WHERE EXTRACT(YEAR FROM match_date) = EXTRACT(YEAR FROM CURRENT_DATE) - 1")
    lines.append("  'this year'           -> WHERE EXTRACT(YEAR FROM match_date) = EXTRACT(YEAR FROM CURRENT_DATE)")
    lines.append("  'last 2 years'        -> WHERE match_date >= CURRENT_DATE - INTERVAL '2 years'")
    lines.append("  'recent'              -> ORDER BY match_date DESC LIMIT 10")
    lines.append("  'latest'              -> ORDER BY match_date DESC LIMIT 1")
    lines.append("  'over the years'      -> GROUP BY EXTRACT(YEAR FROM match_date) ORDER BY year")
    lines.append("  'trend'               -> GROUP BY EXTRACT(YEAR FROM match_date) ORDER BY year")
    lines.append("  'since 2021'          -> WHERE match_date >= '2021-01-01'")
    lines.append("  'before 2022'         -> WHERE match_date < '2022-01-01'")
    lines.append("  'in June'             -> WHERE EXTRACT(MONTH FROM match_date) = 6")
    lines.append("  'summer'              -> WHERE EXTRACT(MONTH FROM match_date) BETWEEN 4 AND 8")
    lines.append("")
    lines.append("ALWAYS include year/date in SELECT when grouping by time:")
    lines.append("  EXTRACT(YEAR FROM match_date) AS year   -- always alias as 'year'")
    lines.append("  TO_CHAR(match_date, 'YYYY-MM') AS month -- for monthly breakdown")
    lines.append("")
    lines.append("NEVER use:")
    lines.append("  YEAR(match_date)        -- MySQL syntax, not PostgreSQL")
    lines.append("  match_date.year         -- not valid SQL")
    lines.append("  DATEPART(year, ...)     -- SQL Server syntax, not PostgreSQL")

    return "\n".join(lines)

# =========================================================
# STRIP HAVING CLAUSES
# =========================================================

def strip_having_clauses(sql_query):
    cleaned = re.sub(
        r'\bHAVING\b.*?(?=\bORDER\b|\bLIMIT\b|\bGROUP\b|;|$)',
        '',
        sql_query,
        flags=re.IGNORECASE | re.DOTALL
    )
    return " ".join(cleaned.split())


def results_are_empty(query_results):
    return all(
        len(item.get("results", [])) == 0
        for item in query_results
    )

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
# GEMINI TEST
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

    # Fix MySQL-style YEAR() function -> PostgreSQL EXTRACT
    sql_query = re.sub(
        r'\bYEAR\s*\(\s*(\w+)\s*\)',
        r'EXTRACT(YEAR FROM \1)',
        sql_query,
        flags=re.IGNORECASE
    )

    # Fix MONTH() -> EXTRACT
    sql_query = re.sub(
        r'\bMONTH\s*\(\s*(\w+)\s*\)',
        r'EXTRACT(MONTH FROM \1)',
        sql_query,
        flags=re.IGNORECASE
    )

    # Fix DAY() -> EXTRACT
    sql_query = re.sub(
        r'\bDAY\s*\(\s*(\w+)\s*\)',
        r'EXTRACT(DAY FROM \1)',
        sql_query,
        flags=re.IGNORECASE
    )

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

def generate_query_plan(question, relax_thresholds=False):

    schema_context  = build_schema_context(question)
    metrics_context = build_metrics_context()
    rules_context   = build_rules_context()
    cricket_terms   = build_cricket_terms_context()
    date_context    = build_date_context()

    threshold_note = ""
    if relax_thresholds:
        threshold_note = """
IMPORTANT - RETRY MODE:
This is a retry after empty results.
DO NOT use any HAVING clauses or minimum ball/innings thresholds.
Return ALL available data regardless of sample size.
The insight layer will handle small sample size caveats.
"""

    prompt = f"""
You are Cricket_Scorer_AI - an elite cricket data engineer and analyst with
encyclopaedic knowledge of T20, ODI, Test, and domestic cricket formats.

Your job: translate ANY natural language cricket question into a precise,
correct, multi-query PostgreSQL plan — including questions about time periods,
years, seasons, date ranges, cricket terminology, and complex statistics.

{threshold_note}

═══════════════════════════════════════
DATABASE
═══════════════════════════════════════
Table: public.nv_play
Each row = one delivery (ball) in a cricket match.

═══════════════════════════════════════
SCHEMA
═══════════════════════════════════════
{schema_context}

═══════════════════════════════════════
DERIVED METRICS
═══════════════════════════════════════
{metrics_context}

═══════════════════════════════════════
CRICKET RULES & BENCHMARKS
═══════════════════════════════════════
{rules_context}

═══════════════════════════════════════
CRICKET TERMINOLOGY -> SQL
═══════════════════════════════════════
{cricket_terms}

═══════════════════════════════════════
DATE & TIME RULES  ← READ CAREFULLY
═══════════════════════════════════════
{date_context}

═══════════════════════════════════════
CRITICAL COLUMN RULES
═══════════════════════════════════════

WICKETS:
  - wicket column is TEXT: 'Caught', 'Bowled', 'LBW', 'Run Out', 'Stumped', 'Hit Wicket'
  - NEVER: wicket = TRUE / wicket = FALSE
  - dismissed  -> wicket IS NOT NULL
  - not out    -> wicket IS NULL
  - specific   -> wicket = 'Caught'

BOOLEANS: legal_ball, free_hit, around_the_wicket, keeper_up  -> use TRUE / FALSE

RUNS:
  - Always SUM — one row = one ball, not a match
  - Strike rate = (SUM(runs_batter) / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0)) * 100
  - Economy     = (SUM(runs_total)  / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0)) * 6

NAMES: Use ILIKE '%name%' for all player and team name matching

═══════════════════════════════════════
QUERY DESIGN RULES
═══════════════════════════════════════

1.  SELECT or WITH...SELECT only. PostgreSQL syntax ONLY.
2.  Never invent column names. Schema columns only.
3.  Meaningful alias on every computed column.
4.  ROUND(value::numeric, 2) for all decimals.
5.  NULLIF(..., 0) on EVERY denominator.
6.  LIMIT: leaderboards -> 15-20. Single entity -> no LIMIT.
7.  ORDER BY meaningfully. Time queries -> ORDER BY year ASC.
8.  TEAM   -> GROUP BY batting_team or bowling_team.
9.  PLAYER -> GROUP BY batter or GROUP BY bowler.
10. TIME   -> GROUP BY EXTRACT(YEAR FROM match_date) AS year
              Always alias as 'year', always ORDER BY year ASC.

THRESHOLDS — CRITICAL RULE:
  APPLY HAVING only for open leaderboards (no specific entity named):
    Batting: HAVING COUNT(*) FILTER (WHERE legal_ball=TRUE) >= 30
    Bowling: HAVING COUNT(*) FILTER (WHERE legal_ball=TRUE) >= 24
  NEVER apply HAVING when:
    - Specific team named (Wimbledon, MI, CSK, any team name)
    - Specific player named
    - Specific year, date, or season mentioned
    - Batting position terms used (top order, openers, etc.)
    - Question asks about ALL players/matches of an entity

COMPLEX QUESTIONS — use MULTIPLE queries:
  Query 1: Overall / career / full-period summary
  Query 2: Year-by-year OR phase breakdown
  Query 3: Ranking, comparison, or opponent breakdown
  Query 4: (optional) Dismissal types or special situations

Always include: player/team name, match count, balls faced/bowled, year if time query.
Use COUNT(DISTINCT match_id) AS matches_played for match counts.

═══════════════════════════════════════
COMPLEX QUERY PATTERNS
═══════════════════════════════════════

YEARLY TEAM PERFORMANCE (use for "every year", "year by year", "each season"):
  SELECT
    EXTRACT(YEAR FROM match_date)                                              AS year,
    COUNT(DISTINCT match_id)                                                   AS matches,
    SUM(runs_total)                                                            AS total_runs,
    COUNT(*) FILTER (WHERE wicket IS NOT NULL)                                 AS wickets_lost,
    ROUND((SUM(runs_total)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*6, 2) AS run_rate,
    ROUND(SUM(runs_total)::numeric / NULLIF(COUNT(DISTINCT match_id),0), 2)    AS avg_score_per_match
  FROM public.nv_play
  WHERE batting_team ILIKE '%team_name%'
  GROUP BY EXTRACT(YEAR FROM match_date)
  ORDER BY year ASC

YEARLY PLAYER PERFORMANCE:
  SELECT
    EXTRACT(YEAR FROM match_date)                                              AS year,
    COUNT(DISTINCT match_id)                                                   AS matches,
    SUM(runs_batter)                                                           AS runs,
    COUNT(*) FILTER (WHERE legal_ball=TRUE)                                    AS balls_faced,
    ROUND((SUM(runs_batter)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*100, 2) AS strike_rate,
    ROUND(SUM(runs_batter)::numeric / NULLIF(COUNT(*) FILTER (WHERE wicket IS NOT NULL),0), 2)     AS average
  FROM public.nv_play
  WHERE batter ILIKE '%player%'
  GROUP BY EXTRACT(YEAR FROM match_date)
  ORDER BY year ASC

PHASE COMPARISON:
  SELECT
    CASE WHEN over_number BETWEEN 1  AND 6  THEN 'Powerplay'
         WHEN over_number BETWEEN 7  AND 15 THEN 'Middle'
         WHEN over_number BETWEEN 16 AND 20 THEN 'Death'
    END AS phase, ...
  GROUP BY phase

HEAD-TO-HEAD:
  SELECT batter, bowler,
    COUNT(*) FILTER (WHERE legal_ball=TRUE) AS balls,
    SUM(runs_batter) AS runs,
    COUNT(*) FILTER (WHERE wicket IS NOT NULL) AS dismissals,
    ROUND((SUM(runs_batter)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*100,2) AS sr
  FROM public.nv_play
  WHERE batter ILIKE '%x%' AND bowler ILIKE '%y%'

OVER-BY-OVER:
  SELECT over_number,
    ROUND(AVG(runs_total),2) AS avg_runs,
    COUNT(*) FILTER (WHERE wicket IS NOT NULL) AS wickets
  FROM public.nv_play
  GROUP BY over_number ORDER BY over_number

═══════════════════════════════════════
ANALYSIS TYPE DETECTION
═══════════════════════════════════════

YEARLY / SEASONAL     -> GROUP BY EXTRACT(YEAR FROM match_date) — ALWAYS for "every year", "each season", "year by year", "annual", "trend over years"
INDIVIDUAL BATTING    -> career stats + phase breakdown + dismissal types
INDIVIDUAL BOWLING    -> career stats + phase breakdown + dismissal methods
TEAM BATTING          -> team totals + phase breakdown + top contributors
TEAM BOWLING          -> team economy + wicket takers + phase breakdown
HEAD TO HEAD          -> batter vs bowler matchup stats
PARTNERSHIP           -> runs/balls together, run rate as a pair
MATCH SUMMARY         -> innings totals, run rate, wickets, top performers
LEADERBOARD           -> ranked list with thresholds applied
PHASE ANALYSIS        -> powerplay / middle / death comparison
DISMISSAL ANALYSIS    -> type breakdown + bowler causing each type
BATTING POSITION      -> top order / middle order / tail analysis
OVER ANALYSIS         -> over-by-over run rates and wickets
DATE RANGE            -> filter by specific date range or period
RECENT FORM           -> last N matches, ORDER BY match_date DESC

═══════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════

Return ONLY valid JSON. No markdown. No explanation. No extra text.

{{
  "analysis_type": "single | multi",
  "intent": "precise description of what user is asking including time dimension if present",
  "has_time_dimension": true or false,
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
            "has_time_dimension": False,
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

    schema_context = build_schema_context(question)
    cricket_terms  = build_cricket_terms_context()
    date_context   = build_date_context()

    prompt = f"""
You are a PostgreSQL expert for cricket ball-by-ball data.

TABLE: public.nv_play (each row = one delivery)

SCHEMA:
{schema_context}

CRICKET TERMINOLOGY:
{cricket_terms}

DATE & TIME RULES:
{date_context}

STRICT RULES:
1.  Return ONLY the raw SQL — no explanation, no markdown, no backticks.
2.  SELECT only. PostgreSQL syntax ONLY.
3.  Only schema columns — never invent columns.
4.  wicket is TEXT — use wicket IS NOT NULL, never wicket = TRUE.
5.  legal_ball, free_hit, around_the_wicket, keeper_up are BOOLEAN.
6.  ROUND(value::numeric, 2) for all decimals.
7.  NULLIF(..., 0) on all denominators.
8.  Meaningful aliases on every column.
9.  LIMIT 20 for open rankings.
10. ILIKE for all name matching.
11. NO HAVING clauses — return all available data.
12. DATE QUERIES: use EXTRACT(YEAR FROM match_date) — NEVER use YEAR() function.
13. For yearly grouping: GROUP BY EXTRACT(YEAR FROM match_date) ORDER BY year ASC.
14. Always alias EXTRACT(YEAR FROM match_date) AS year.

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
# =========================================================

def execute_query_plan_with_retry(query_plan, question):

    all_results, all_sql = execute_query_plan(query_plan)

    if results_are_empty(all_results):

        # Stage 1 — strip HAVING from existing SQL
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

        # Stage 2 — regenerate entire plan without thresholds
        if results_are_empty(retry_results):

            fresh_plan = generate_query_plan(question, relax_thresholds=True)
            fresh_results, fresh_sql = execute_query_plan(fresh_plan)

            return fresh_results, fresh_sql, True

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
NOTE: Small sample size — thresholds were relaxed to retrieve this data.
Mention naturally (e.g. "based on limited data") but still analyse fully.
Do not refuse to analyse.
"""

    # Detect if this is a time-based query for extra insight instructions
    time_keywords = ["year", "season", "annual", "trend", "every", "each year",
                     "monthly", "over time", "progression", "2021", "2022",
                     "2023", "2024", "2025", "recent", "latest", "history"]

    is_time_query = any(kw in question.lower() for kw in time_keywords)

    time_insight_note = ""
    if is_time_query:
        time_insight_note = """
TIME-BASED ANALYSIS INSTRUCTIONS:
- This question has a time/year dimension — structure your analysis accordingly
- Identify the TREND: is performance improving, declining, or consistent year-over-year?
- Highlight the BEST year and WORST year with exact figures
- Mention year-on-year changes (e.g. "SR improved from 124 in 2022 to 141 in 2023")
- Note any significant dips or spikes and what they might indicate
- In the Key Numbers table include a 'Year' column as the first column
- In Standout Moments highlight the peak year performance
- In Verdict summarise the overall trajectory (rising/declining/consistent)
"""

    prompt = f"""
You are Cricket_Scorer_AI - a world-class cricket analyst combining the
tactical depth of a coaching analyst, the storytelling of a top commentator,
and the precision of a statistician.

{small_sample_note}
{time_insight_note}

USER QUESTION
{question}

DATA FROM DATABASE
{compact_data}

CRICKET RULES, BENCHMARKS & INSIGHT GUIDELINES
{insight_rules_context}

YOUR TASK

Produce a rich, structured cricket analysis report. Follow this layout exactly:

---

## 🏏 [Compelling headline — include time period if relevant e.g. "Wimbledon's Batting Through the Years"]

### 📊 Key Numbers
Clean markdown table with actual numbers.
For time queries: Year | Matches | Runs | SR | Average (etc.)
For player queries: Player | Matches | Runs | SR | Average | 4s | 6s
For team queries: Team | Matches | Runs | Run Rate | Wickets Lost

### 🔍 Analysis

**Overall Performance**
High-level summary with exact figures. Classify using tier labels.
For time queries: describe the overall trajectory across years.

**Strengths**
Specific stats backing every claim. Cricket-specific language.
For time queries: identify the strongest period and why.

**Weaknesses / Vulnerabilities**
Reference exact numbers. For time queries: weakest period.

**Year-by-Year Trend** (include ONLY if time-based data present)
Describe the progression year by year. Identify peaks, troughs, and inflection points.
Mention specific year comparisons: "2022 was their best year with X runs at SR of Y"

**Tactical Insights**
Phase-wise, bowling angle, keeper position, scoring zones.
Only from data that exists.

**Context & Comparisons** (if multiple entities)
Compare directly. What separates top from rest?

### 📈 Standout Moments / Records
2-4 bullet points. For time queries include best/worst year stat.

### 💡 Verdict
3-5 sentence expert panel verdict.
For time queries: is the team/player on an upward or downward trajectory?

---

STRICT RULES
- Cite exact numbers always (SR of 187.4, economy of 6.23, year 2023)
- Cricket terminology naturally
- Markdown tables with | pipes and --- separators
- Bold key names: **Name**, bold years: **2023**
- Multiple entities: compare explicitly not in isolation
- Phase-wise data: dedicate paragraph
- Classify metrics against benchmark tiers
- Empty data: acknowledge gracefully
- Under 700 words unless data demands more
- Never mention SQL, database, queries
- Never fabricate stats
- No filler phrases like it is worth noting
"""

    return llm(prompt)

# =========================================================
# GENERATE CHART CONFIG
# =========================================================

def generate_chart_config(question, query_results):

    compact_data = json.dumps(query_results, default=str)[:8000]

    # Detect time dimension for chart type hints
    time_keywords = ["year", "season", "annual", "trend", "every", "each year",
                     "monthly", "over time", "progression", "history"]
    is_time_query = any(kw in question.lower() for kw in time_keywords)

    time_chart_hint = ""
    if is_time_query:
        time_chart_hint = """
TIME QUERY DETECTED:
- If data has a 'year' column with multiple years -> prefer 'line' or 'area'
- line: for single metric trends over years (e.g. SR by year)
- area: for volume/cumulative trends (e.g. total runs by year)
- bar: for year-by-year comparison of multiple metrics side by side
- x_key should be 'year'
- y_keys should be numeric performance columns (runs, strike_rate, average, etc.)
"""

    prompt = f"""
You are a cricket data visualization expert.

Decide which chart type best communicates the insight.
If no chart adds value return null.

{time_chart_hint}

CHART TYPE GUIDE

bar
  WHEN: Side-by-side multi-metric comparison across players/teams/phases/years
  REQUIRES: Multiple y_keys, categorical x_axis
  MAX ROWS: 10

bar_colored
  WHEN: Single metric leaderboard — each entry a unique color
  REQUIRES: Exactly 1 y_key
  MAX ROWS: 15

line
  WHEN: Trends over time, year-by-year progression, over-by-over
  REQUIRES: Ordered x_axis (year, over_number, match sequence)
  MAX ROWS: 25

area
  WHEN: Volume or cumulative trends over time
  REQUIRES: Ordered x_axis
  MAX ROWS: 25

pie
  WHEN: Distribution or share of a whole (dismissal types, run breakdown)
  REQUIRES: 1 y_key, categorical x_key
  MAX ROWS: 8

radar
  WHEN: Multi-dimension profile of 2-4 players across 4-6 metrics
  REQUIRES: Multiple y_keys, 2-4 rows
  MAX ROWS: 4

SELECTION RULES
1. year/time data with trend             -> line or area
2. year/time data multiple metrics       -> bar (grouped by year)
3. distribution/breakdown/share          -> pie
4. compare/vs/top N + ONE metric         -> bar_colored
5. compare/vs/top N + MULTIPLE metrics   -> bar
6. player profile multi-dimension        -> radar
7. 1 row only                            -> null
8. no numeric columns                    -> null
9. single stat lookup                    -> null

DATA RULES
1. x_key must be STRING or year (integer)
2. y_keys must be NUMERIC only
3. All values actual numbers not strings
4. Floats rounded to 2 decimal places
5. For year data: x_key = 'year', values are integers like 2021, 2022, 2023
6. title short and specific
7. subtitle optional one-line context

RETURN FORMAT
{{
  "chart_type": "bar | bar_colored | line | area | pie | radar",
  "title": "Short descriptive title",
  "subtitle": "Optional one-line context",
  "x_key": "year or category column",
  "y_keys": ["metric1", "metric2"],
  "data": [
    {{ "year": 2021, "runs": 450, "strike_rate": 124.5 }},
    {{ "year": 2022, "runs": 612, "strike_rate": 138.2 }}
  ]
}}

If no chart adds value return exactly: null
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

        # Step 1 — Generate SQL query plan
        query_plan = generate_query_plan(req.question)

        # Step 2 — Execute with auto-retry on empty results
        query_results, sql_queries, thresholds_relaxed = execute_query_plan_with_retry(
            query_plan,
            req.question
        )

        # Step 3 — Format plain-text tables
        tables = [
            {
                "query_name": item["query_name"],
                "purpose":    item["purpose"],
                "table":      format_results(item["results"])
            }
            for item in query_results
        ]

        # Step 4 — Generate chart config
        chart_config = generate_chart_config(req.question, query_results)

        # Step 5 — Generate cricket insight
        insight = generate_cricket_insight(
            req.question,
            query_results,
            small_sample=thresholds_relaxed
        )

        # Step 6 — Return full response
        return {
            "question":           req.question,
            "analysis_type":      query_plan.get("analysis_type", "single"),
            "has_time_dimension": query_plan.get("has_time_dimension", False),
            "thresholds_relaxed": thresholds_relaxed,
            "sql_queries":        sql_queries,
            "results":            query_results,
            "tables":             tables,
            "chart_config":       chart_config,
            "insight":            insight
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}