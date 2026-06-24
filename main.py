from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import os
import re
import json
import base64
import io
import traceback
from functools import lru_cache
from typing import Optional, Tuple, List, Dict, Any

import psycopg2
from psycopg2 import pool as pg_pool

import pandas as pd
import numpy as np
import duckdb

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

from schema_context import NV_PLAY_DICTIONARY
from derived_metrics import DERIVED_METRICS
from cricket_rules import CRICKET_RULES


# =========================================================
# LOAD ENV
# =========================================================
load_dotenv()

APP_VERSION = "3.1.0"

# =========================================================
# VERTEX AI SETUP
# =========================================================
vertexai.init(
    project=os.getenv("GCP_PROJECT"),
    location=os.getenv("GCP_LOCATION", "us-central1")
)

model = GenerativeModel("gemini-2.5-flash")

PLAN_CONFIG    = GenerationConfig(temperature=0.1, top_p=0.9,  max_output_tokens=8192)
INSIGHT_CONFIG = GenerationConfig(temperature=0.3, top_p=0.95, max_output_tokens=8192)
CHART_CONFIG_CFG = GenerationConfig(temperature=0.1, top_p=0.9, max_output_tokens=2048)

# =========================================================
# FASTAPI
# =========================================================
app = FastAPI(title="Cricket_Scorer_AI", version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# DB CONNECTION POOL
# =========================================================
DB_POOL: Optional[pg_pool.SimpleConnectionPool] = None


def init_db_pool():
    global DB_POOL
    if DB_POOL is None:
        try:
            DB_POOL = pg_pool.SimpleConnectionPool(
                minconn=1,
                maxconn=int(os.getenv("DB_POOL_MAX", "8")),
                host=os.getenv("DB_HOST"),
                port=os.getenv("DB_PORT"),
                database=os.getenv("DB_NAME"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                connect_timeout=10,
            )
        except Exception:
            DB_POOL = None


def get_db_conn():
    if DB_POOL is None:
        init_db_pool()
    if DB_POOL is None:
        return psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            connect_timeout=10,
        )
    return DB_POOL.getconn()


def release_db_conn(conn):
    if conn is None:
        return
    if DB_POOL is not None:
        try:
            DB_POOL.putconn(conn)
            return
        except Exception:
            pass
    try:
        conn.close()
    except Exception:
        pass


@app.on_event("startup")
def _on_startup():
    init_db_pool()


# =========================================================
# REQUEST MODELS
# =========================================================
class QueryRequest(BaseModel):
    question: str
    session_context: list = []


class ExcelQueryRequest(BaseModel):
    question: str
    file_base64: str
    file_name: str
    file_ext: str


# =========================================================
# LLM WRAPPER
# =========================================================
def llm(prompt: str, cfg: Optional[GenerationConfig] = None, retries: int = 2) -> str:
    last_err = None
    for attempt in range(retries + 1):
        try:
            if cfg is not None:
                resp = model.generate_content(prompt, generation_config=cfg)
            else:
                resp = model.generate_content(prompt)
            return (resp.text or "").strip()
        except Exception as e:
            last_err = e
    raise RuntimeError(f"LLM call failed: {last_err}")


def llm_json(prompt: str, cfg: Optional[GenerationConfig] = None) -> Optional[dict]:
    raw = llm(prompt, cfg=cfg or PLAN_CONFIG)
    raw = raw.replace("```json", "").replace("```", "").strip()
    m = re.search(r'\{[\s\S]*\}', raw)
    if m:
        raw = m.group(0)
    try:
        return json.loads(raw)
    except Exception:
        try:
            fixed = re.sub(r",\s*([\]}])", r"\1", raw)
            return json.loads(fixed)
        except Exception:
            return None


# =========================================================
# INTENT CLASSIFICATION
# =========================================================
INTENT_KEYWORDS = {
    "is_batting":    ["bat","runs","strike rate","average","century","fifty","boundary","four","six","scorer","openers","batting","innings","batter"],
    "is_bowling":    ["bowl","wicket","economy","dot ball","yorker","delivery","spin","pace","seam","swing","bowler","bowling"],
    "is_fielding":   ["catch","field","run out","direct hit","dropped","boundary save","fielder"],
    "is_team":       ["team","squad","side","xi","playing eleven","franchise"],
    "is_h2h":        ["head to head","h2h","matchup","batter vs bowler","vs bowler","vs batter"],
    "is_allrounder": ["all-rounder","allrounder","all rounder","both bat and bowl"],
    "is_form":       ["recent","last ","form","current form","in form","last 5","last 10","this year","this season","current"],
    "is_trend":      ["year","season","annual","trend","every year","each year","monthly","over time","progression","history","since","before","2021","2022","2023","2024","2025"],
    "is_phase":      ["powerplay","pp ","middle over","death over","slog","phase","over 1","over 6","over 16","over 20","first 6","last 5 overs"],
    "is_comparison": [" vs "," versus ","compare","comparison","between","better","worse","who is better","which team"],
    "is_opponent":   ["against ","opponent","facing","when bowling to","when batting against","bogey","favourite opponent"],
    "is_consistency":["consistent","consistency","reliable","duck","fifty","century","milestone","how often"],
    "is_chase":      ["chase","chasing","run chase","target","second innings","batting second","defending","first innings","batting first","setting"],
    "is_pressure":   ["pressure","clutch","crunch","crucial","eliminate","final","knockout","must win"],
    "is_leaderboard":["top","best","highest","most","ranking","rank","leaderboard","who has most","who scored most","who took most"],
    "is_milestone":  ["century","centuries","hundred","50s","fifties","duck","golden duck","hat-trick","five-for","5 wickets","ten wickets"],
    "is_over_by_over":["over by over","each over","per over","over number","which over","best over"],
    "is_venue":      ["venue","ground","stadium","pitch","home","away"],
    "is_fantasy":    ["fantasy","dream11","dream 11","pick","differential","captain","vice captain","points"],
    "is_predictive": ["predict","likely","probability","expect","forecast","will","chances","should i pick"],
    "is_time_based": ["time","duration","how long","longest","slowest","fastest","minutes","seconds","hours",
                      "pace of play","over rate","time per over","time taken","took longest","crease time",
                      "time spent","how much time","elapsed","between deliveries","delay","slow over rate"],
}

FORMAT_KEYWORDS = {
    "fmt_t20":  ["t20","t-20","ipl","bbl","psl","cpl","sa20","hundred","the hundred"],
    "fmt_odi":  ["odi","one day","one-day","50 over","50-over","list a","world cup odi"],
    "fmt_test": ["test","red ball","test match","test cricket","test series"],
}


def extract_player_names(text: str) -> List[str]:
    keywords = {"how","what","when","which","who","where","best","top","most","has","does",
                "in","at","against","for","is","are","was","were","with","from","by","of",
                "the","a","an","and","or","not","but"}
    tokens = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', text)
    return [t for t in tokens if t.lower() not in keywords]


def classify_intent(question: str) -> dict:
    q = question.lower()
    out: Dict[str, Any] = {}
    for k, words in INTENT_KEYWORDS.items():
        out[k] = any(w in q for w in words)
    for k, words in FORMAT_KEYWORDS.items():
        out[k] = any(w in q for w in words)
    out["candidate_names"] = extract_player_names(question)
    return out


# =========================================================
# CONTEXT BUILDERS — CACHED
# =========================================================
@lru_cache(maxsize=1)
def build_schema_context() -> str:
    lines = []
    for col, meta in NV_PLAY_DICTIONARY.items():
        lines.append(
            f"\nColumn: {col}"
            f"\nDescription: {meta['description']}"
            f"\nDatatype: {meta['datatype']}"
            f"\nCategory: {meta['category']}"
            f"\nAggregation: {meta['aggregation']}"
            f"\nSynonyms: {', '.join(meta['synonyms'])}"
        )
    return "\n".join(lines)


@lru_cache(maxsize=1)
def build_metrics_context() -> str:
    lines = []
    for m, meta in DERIVED_METRICS.items():
        lines.append(
            f"\nMetric: {m}"
            f"\nDescription: {meta['description']}"
            f"\nFormula: {meta['formula']}"
            f"\nCategory: {meta['category']}"
            f"\nSynonyms: {', '.join(meta['synonyms'])}"
        )
    return "\n".join(lines)


@lru_cache(maxsize=1)
def build_rules_context() -> str:
    rules = CRICKET_RULES
    lines = []
    lines.append("MINIMUM SAMPLE SIZES:")
    for category, values in rules["minimum_sample_size"].items():
        for k, v in values.items():
            lines.append(f"  {category} -> {k}: {v}")

    for fmt in ["T20", "ODI"]:
        lines.append(f"\nMATCH PHASES ({fmt}):")
        for phase, data in rules["match_phases"][fmt].items():
            lines.append(f"  {phase}: overs {data['start_over']}-{data['end_over']} -- {data['description']}")

    lines.append("\nBATTING BENCHMARKS:")
    for fmt in ["T20", "ODI", "TEST"]:
        if fmt in rules["batting_benchmarks"]:
            lines.append(f"  [{fmt}]")
            for metric, tiers in rules["batting_benchmarks"][fmt].items():
                lines.append(f"    {metric}:")
                for tier, data in tiers.items():
                    lines.append(f"      {tier}: {data}")

    lines.append("\nBOWLING BENCHMARKS:")
    for fmt in ["T20", "ODI", "TEST"]:
        if fmt in rules["bowling_benchmarks"]:
            lines.append(f"  [{fmt}]")
            for metric, tiers in rules["bowling_benchmarks"][fmt].items():
                lines.append(f"    {metric}:")
                for tier, data in tiers.items():
                    lines.append(f"      {tier}: {data}")

    return "\n".join(lines)


# =========================================================
# SESSION CONTEXT
# =========================================================
def build_session_context(session_context: list) -> str:
    if not session_context:
        return ""
    lines = ["\nPREVIOUS CONVERSATION (use to resolve indirect references):"]
    for i, turn in enumerate(session_context[-4:], 1):
        q = turn.get("question", "")
        a = (turn.get("summary", "") or "")[:300]
        lines.append(f"  Turn {i}: Q: {q}")
        if a:
            lines.append(f"           A: {a}")
    return "\n".join(lines)


# =========================================================
# SQL CLEANING + VALIDATION
# =========================================================
COMMON_COL_FIXES = [
    (r'\bmatch_id\b',       'match'),
    (r'\binnings_number\b', 'innings'),
    (r'\bover_number\b',    'over'),
    (r'\bball_number\b',    'ball'),
    (r'\bruns_batter\b',    'runs'),
    (r'\bmatch_date\b',     'date'),
]


def clean_sql_postgres(sql: str) -> str:
    sql = sql.replace("```sql", "").replace("```", "").strip()
    sql = " ".join(sql.split())

    for pat, rep in COMMON_COL_FIXES:
        sql = re.sub(pat, rep, sql, flags=re.IGNORECASE)

    sql = re.sub(r'\bSUM\s*\(\s*runs_total\s*\)', 'SUM(runs + extra_runs)', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bruns_total\b', '(runs + extra_runs)', sql, flags=re.IGNORECASE)
    sql = re.sub(r"TO_TIMESTAMP\s*\(\s*timestamp\s*,\s*'[^']+'\s*\)", 'timestamp', sql, flags=re.IGNORECASE)

    for wrong, right in [
        ("wicket = TRUE",  "wicket IS NOT NULL"),
        ("wicket=TRUE",    "wicket IS NOT NULL"),
        ("wicket = FALSE", "wicket IS NULL"),
        ("wicket=FALSE",   "wicket IS NULL"),
        ("wicket = 1",     "wicket IS NOT NULL"),
        ("wicket=1",       "wicket IS NOT NULL"),
        ("wicket = 0",     "wicket IS NULL"),
        ("wicket=0",       "wicket IS NULL"),
    ]:
        sql = sql.replace(wrong, right)

    sql = re.sub(r'\bYEAR\s*\(\s*(\w+)\s*\)',  r'EXTRACT(YEAR FROM \1)', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bMONTH\s*\(\s*(\w+)\s*\)', r'EXTRACT(MONTH FROM \1)', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bDAY\s*\(\s*(\w+)\s*\)',   r'EXTRACT(DAY FROM \1)', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bDATEPART\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)', r'EXTRACT(\1 FROM \2)', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bIFNULL\s*\(', 'COALESCE(', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bNVL\s*\(',    'COALESCE(', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bISNULL\s*\(', 'COALESCE(', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bDATE_ADD\s*\(\s*(\w+)\s*,\s*INTERVAL\s+(-?\d+)\s+(\w+)\s*\)',
                 r"\1 + INTERVAL '\2 \3'", sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bDATEADD\s*\(\s*(\w+)\s*,\s*(-?\d+)\s*,\s*(\w+)\s*\)',
                 r"\3 + INTERVAL '\2 \1'", sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bSELECT\s+TOP\s+(\d+)\s+', r'SELECT ', sql, flags=re.IGNORECASE)
    sql = re.sub(r'`(\w+)`', r'"\1"', sql)
    sql = re.sub(r'EXTRACT\s*\(\s*(YEAR|MONTH|DAY)\s+FROM\s+(\w+)\s*\)\s+AS\s+(\w+)',
                 r'EXTRACT(\1 FROM \2)::int AS \3', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bINSTR\s*\(', 'POSITION(', sql, flags=re.IGNORECASE)
    return sql.strip()


def clean_sql_duckdb(sql: str) -> str:
    sql = sql.replace("```sql", "").replace("```", "").strip()
    sql = " ".join(sql.split())

    sql = re.sub(r'\bpublic\.nv_play\b', 'nv_play', sql, flags=re.IGNORECASE)
    for pat, rep in COMMON_COL_FIXES:
        sql = re.sub(pat, rep, sql, flags=re.IGNORECASE)

    sql = re.sub(r'::\s*numeric', '', sql, flags=re.IGNORECASE)
    sql = re.sub(r'::\s*int\b', '', sql, flags=re.IGNORECASE)
    sql = re.sub(r'::\s*float\b', '', sql, flags=re.IGNORECASE)

    for wrong, right in [
        ("wicket = TRUE",  "wicket IS NOT NULL"),
        ("wicket=TRUE",    "wicket IS NOT NULL"),
        ("wicket = FALSE", "wicket IS NULL"),
        ("wicket=FALSE",   "wicket IS NULL"),
    ]:
        sql = sql.replace(wrong, right)

    sql = re.sub(r'\bYEAR\s*\(\s*(\w+)\s*\)',  r'EXTRACT(YEAR FROM \1)', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bMONTH\s*\(\s*(\w+)\s*\)', r'EXTRACT(MONTH FROM \1)', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bIFNULL\s*\(', 'COALESCE(', sql, flags=re.IGNORECASE)
    return sql.strip()


def validate_sql(sql: str) -> None:
    first = sql.strip().split()[0].upper() if sql.strip() else ""
    if first not in ("SELECT", "WITH"):
        raise Exception("Only SELECT queries are allowed.")
    blocked = ["DELETE", "DROP", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "CREATE", "GRANT", "REVOKE"]
    upper = sql.upper()
    for kw in blocked:
        if re.search(rf'\b{kw}\b', upper):
            raise Exception(f"{kw} operation not allowed.")


def strip_having_clauses(sql: str) -> str:
    cleaned = re.sub(
        r'\bHAVING\b.*?(?=\bORDER\b|\bLIMIT\b|\bGROUP\b|;|$)',
        '',
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return " ".join(cleaned.split())


def results_are_empty(query_results: list) -> bool:
    return all(len(item.get("results", [])) == 0 for item in query_results)


# =========================================================
# INTENT HINT BUILDER
# =========================================================
def build_intent_hint(intent: Optional[dict]) -> str:
    if not intent:
        return ""
    flags = [k for k, v in intent.items() if v is True]
    out = f"\nDETECTED INTENT FLAGS: {', '.join(flags)}\n"
    if intent.get("candidate_names"):
        out += f"CANDIDATE ENTITY NAMES: {', '.join(intent['candidate_names'])}\n"
    if intent.get("is_time_based"):
        out += ("\n⚠️ TIME-BASED QUERY: Use timestamp column with EPOCH arithmetic. "
                "Never use balls_faced as a duration proxy.\n")
    return out


# =========================================================
# CRICKET TERMINOLOGY — SQL TRANSLATION (FIXED DISMISSAL FORMULA)
# =========================================================
@lru_cache(maxsize=1)
def build_cricket_terms_context() -> str:
    return """CRICKET TERMINOLOGY -> SQL TRANSLATION:

BATTING POSITIONS:
  opener/openers -> batting_position IN (1,2)
  top order -> batting_position IN (1,2,3)
  middle order -> batting_position IN (4,5,6)
  lower order/tail -> batting_position IN (7,8,9,10,11)
  finisher -> batting_position IN (5,6,7)

PHASES (T20): pp/powerplay = over 1-6 | middle = 7-15 | death/slog = 16-20
PHASES (ODI): pp = 1-10 | middle = 11-40 | death = 41-50

DISMISSALS:
  caught -> wicket='Caught' | bowled -> wicket='Bowled' | lbw -> wicket='LBW'
  run out -> wicket='Run Out' | stumped -> wicket='Stumped' | hit wicket -> wicket='Hit Wicket'
  duck -> runs=0 AND wicket IS NOT NULL
  not out -> wicket IS NULL
  dismissed -> wicket IS NOT NULL
  golden duck -> runs=0 AND wicket IS NOT NULL AND ball=1

DELIVERY:
  dot ball -> (runs+extra_runs)=0 AND legal_ball=TRUE
  boundary -> runs IN (4,6) | six -> runs=6 | four -> runs=4
  free hit -> free_hit=TRUE
  wide/no ball -> legal_ball=FALSE
  scoring shot -> runs>0 AND legal_ball=TRUE

MATCH CONTEXT:
  first innings -> innings=1 | second innings/chase -> innings=2

=== CRITICAL: DISMISSAL COUNTING IN BALL-BY-BALL DATA ===

The table has ONE ROW PER DELIVERY. The 'wicket' column is ONLY populated on
the specific delivery where the dismissal occurred. Therefore:

  ✅ CORRECT — count dismissals (innings where batter was out):
     COUNT(DISTINCT CASE WHEN wicket IS NOT NULL THEN match || '-' || innings::text ELSE NULL END)

  ✅ CORRECT — innings played (whether out or not out):
     COUNT(DISTINCT match || '-' || innings::text)

  ❌ WRONG — this counts delivery rows, NOT dismissals:
     COUNT(*) FILTER (WHERE wicket IS NOT NULL)
     COUNT(wicket)

FORMULAS (PostgreSQL — use NULLIF to avoid divide-by-zero):

  batting avg =
    SUM(runs) / NULLIF(
      COUNT(DISTINCT CASE WHEN wicket IS NOT NULL THEN match || '-' || innings::text ELSE NULL END)
    , 0)

  strike rate =
    ROUND((SUM(runs)::numeric / NULLIF(SUM(CASE WHEN legal_ball THEN 1 ELSE 0 END), 0)) * 100, 2)

  economy =
    ROUND((SUM(runs + extra_runs)::numeric / NULLIF(SUM(CASE WHEN legal_ball THEN 1 ELSE 0 END), 0)) * 6, 2)

  bowling avg =
    SUM(runs + extra_runs) / NULLIF(
      COUNT(DISTINCT CASE WHEN wicket IS NOT NULL THEN match || '-' || innings::text || '-' || wicket ELSE NULL END)
    , 0)

  bowling SR =
    SUM(CASE WHEN legal_ball THEN 1 ELSE 0 END)::numeric /
    NULLIF(COUNT(DISTINCT CASE WHEN wicket IS NOT NULL THEN match || '-' || innings::text || '-' || wicket ELSE NULL END), 0)

  dot % =
    ROUND((SUM(CASE WHEN (runs+extra_runs)=0 AND legal_ball THEN 1 ELSE 0 END)::numeric
           / NULLIF(SUM(CASE WHEN legal_ball THEN 1 ELSE 0 END), 0)) * 100, 2)

  boundary % =
    ROUND((SUM(CASE WHEN runs IN (4,6) THEN 1 ELSE 0 END)::numeric
           / NULLIF(SUM(CASE WHEN legal_ball THEN 1 ELSE 0 END), 0)) * 100, 2)

  innings count (all innings) =
    COUNT(DISTINCT match || '-' || innings::text)

  not_outs =
    COUNT(DISTINCT match || '-' || innings::text)
    - COUNT(DISTINCT CASE WHEN wicket IS NOT NULL THEN match || '-' || innings::text ELSE NULL END)

NOTE for DuckDB (Excel route): replace ::numeric and ::int with CAST(x AS DOUBLE),
and use match || '-' || CAST(innings AS VARCHAR) for concatenation.
"""


@lru_cache(maxsize=1)
def build_date_context() -> str:
    return """DATE & TIME RULES (PostgreSQL):
  YEAR: EXTRACT(YEAR FROM date)::int AS year
  MONTH: EXTRACT(MONTH FROM date)::int
  'last N years' -> WHERE date >= CURRENT_DATE - INTERVAL 'N years'
  'this year' -> WHERE EXTRACT(YEAR FROM date) = EXTRACT(YEAR FROM CURRENT_DATE)
  'in YYYY' -> WHERE EXTRACT(YEAR FROM date) = YYYY
  Last N matches -> CTE with DISTINCT match ORDER BY date DESC LIMIT N
  NEVER: YEAR()/MONTH()/DAY()/DATEADD()/DATEPART()
  ALWAYS ORDER BY year ASC for trends.
"""


@lru_cache(maxsize=1)
def build_format_context() -> str:
    return """FORMAT RULES:
  T20/T20I -> T20 benchmarks | ODI/50 Over/List A -> ODI | Test/First Class/Timed -> TEST
  No format mentioned -> do NOT filter match_type; derive format from data's match_type column.
  T20: SR elite >160 | eco elite <6.5 | PP overs 1-6 | death overs 16-20
  ODI: SR elite >110 | eco elite <4.5 | PP overs 1-10 | death overs 41-50
  Test: avg elite >55 | eco elite <2.0
"""


@lru_cache(maxsize=1)
def build_advanced_patterns_context() -> str:
    return """ADVANCED SQL PATTERNS (PostgreSQL):

1. CAREER BASELINE (correct dismissal counting):
  SELECT
    batter,
    COUNT(DISTINCT match) AS matches,
    COUNT(DISTINCT match || '-' || innings::text) AS innings_played,
    SUM(runs) AS total_runs,
    SUM(CASE WHEN legal_ball THEN 1 ELSE 0 END) AS balls_faced,
    COUNT(DISTINCT CASE WHEN wicket IS NOT NULL THEN match || '-' || innings::text ELSE NULL END) AS dismissals,
    COUNT(DISTINCT match || '-' || innings::text)
      - COUNT(DISTINCT CASE WHEN wicket IS NOT NULL THEN match || '-' || innings::text ELSE NULL END) AS not_outs,
    ROUND(SUM(runs)::numeric / NULLIF(
      COUNT(DISTINCT CASE WHEN wicket IS NOT NULL THEN match || '-' || innings::text ELSE NULL END)
    , 0), 2) AS batting_avg,
    ROUND((SUM(runs)::numeric / NULLIF(SUM(CASE WHEN legal_ball THEN 1 ELSE 0 END), 0)) * 100, 2) AS strike_rate,
    COUNT(CASE WHEN runs = 4 THEN 1 END) AS fours,
    COUNT(CASE WHEN runs = 6 THEN 1 END) AS sixes
  FROM public.nv_play
  WHERE batter ILIKE '%name%'
  GROUP BY batter;

2. INNINGS LIST (for verifying dismissal counts):
  SELECT
    match, innings,
    SUM(runs) AS runs_scored,
    MAX(CASE WHEN wicket IS NOT NULL THEN wicket ELSE 'not out' END) AS how_out,
    COUNT(DISTINCT CASE WHEN wicket IS NOT NULL THEN wicket ELSE NULL END) AS wickets_in_innings
  FROM public.nv_play
  WHERE batter ILIKE '%name%'
  GROUP BY match, innings
  ORDER BY match, innings;

3. PHASE SPLIT — use FILTER (WHERE over BETWEEN x AND y) per phase.

4. RECENT FORM — CTE: SELECT DISTINCT match, date ORDER BY date DESC LIMIT N, then JOIN.

5. YEARLY TREND — GROUP BY EXTRACT(YEAR FROM date)::int ORDER BY year ASC.

6. HEAD-TO-HEAD — WHERE batter ILIKE ... AND bowler ILIKE ...,
   dismissals = COUNT(DISTINCT CASE WHEN wicket IS NOT NULL THEN match || '-' || innings::text ELSE NULL END)

7. DISMISSAL BREAKDOWN — GROUP BY wicket WHERE wicket IS NOT NULL.

8. BOWLER IMPACT vs INNINGS RUN RATE (complex CTE pattern):
  -- Compares each bowler's economy against the innings overall run rate.
  WITH innings_rr AS (
    SELECT
      match,
      innings,
      ROUND((SUM(runs + extra_runs)::numeric / NULLIF(SUM(CASE WHEN legal_ball THEN 1 ELSE 0 END), 0)) * 6, 2) AS innings_rr
    FROM public.nv_play
    GROUP BY match, innings
  ),
  bowler_stats AS (
    SELECT
      bowler,
      p.match,
      p.innings,
      ROUND((SUM(p.runs + p.extra_runs)::numeric / NULLIF(SUM(CASE WHEN p.legal_ball THEN 1 ELSE 0 END), 0)) * 6, 2) AS bowler_economy,
      SUM(CASE WHEN p.legal_ball THEN 1 ELSE 0 END) AS balls_bowled
    FROM public.nv_play p
    GROUP BY bowler, p.match, p.innings
    HAVING SUM(CASE WHEN p.legal_ball THEN 1 ELSE 0 END) >= 18  -- min 3 overs per spell
  )
  SELECT
    b.bowler,
    COUNT(DISTINCT b.match || '-' || b.innings::text) AS spells,
    ROUND(AVG(i.innings_rr - b.bowler_economy), 2) AS avg_rr_suppression,
    ROUND(AVG(b.bowler_economy), 2) AS avg_economy,
    SUM(b.balls_bowled) AS total_balls
  FROM bowler_stats b
  JOIN innings_rr i ON i.match = b.match AND i.innings = b.innings
  GROUP BY b.bowler
  HAVING SUM(b.balls_bowled) >= 18  -- min 3 overs total across tournament
  ORDER BY avg_rr_suppression DESC
  LIMIT 15;

9. CUSTOM INFLUENCE SCORE per player per match (decomposed CTE):
  -- Use CTEs for each component to avoid complex single-query failures.
  WITH batting AS (
    SELECT match, batter AS player,
      SUM(runs) AS batting_runs
    FROM public.nv_play
    GROUP BY match, batter
  ),
  bowling AS (
    SELECT match, bowler AS player,
      COUNT(DISTINCT CASE WHEN wicket IS NOT NULL THEN match || '-' || innings::text || '-' || over::text || '-' || ball::text ELSE NULL END) AS wickets_taken,
      SUM(CASE WHEN (runs + extra_runs) = 0 AND legal_ball THEN 1 ELSE 0 END) AS dot_balls
    FROM public.nv_play
    GROUP BY match, bowler
  ),
  fielding AS (
    SELECT match, fielder AS player,
      COUNT(*) AS catches
    FROM public.nv_play
    WHERE fielder IS NOT NULL AND wicket ILIKE '%caught%'
    GROUP BY match, fielder
  ),
  combined AS (
    SELECT
      COALESCE(bat.match, bowl.match, fld.match) AS match,
      COALESCE(bat.player, bowl.player, fld.player) AS player,
      COALESCE(bat.batting_runs, 0) AS batting_runs,
      COALESCE(bowl.wickets_taken, 0) AS wickets_taken,
      COALESCE(bowl.dot_balls, 0) AS dot_balls,
      COALESCE(fld.catches, 0) AS catches,
      COALESCE(bat.batting_runs, 0)
        + (COALESCE(bowl.wickets_taken, 0) * 20)
        + (COALESCE(bowl.dot_balls, 0) * 0.5)
        + (COALESCE(fld.catches, 0) * 10) AS influence_score
    FROM batting bat
    FULL OUTER JOIN bowling bowl ON bat.match = bowl.match AND bat.player = bowl.player
    FULL OUTER JOIN fielding fld ON COALESCE(bat.match, bowl.match) = fld.match
                                 AND COALESCE(bat.player, bowl.player) = fld.player
  )
  SELECT c.match, c.player, c.batting_runs, c.wickets_taken, c.dot_balls, c.catches,
    ROUND(c.influence_score::numeric, 1) AS influence_score
  FROM combined c
  -- To filter for losing team: join a match_result subquery or use batting_team/bowling_team columns
  ORDER BY c.influence_score DESC
  LIMIT 20;

10. IDENTIFICATION + DETAIL PATTERN:
  -- When a question asks to IDENTIFY a match/player AND then describe it,
  -- always produce TWO queries: one to identify, one to drill into the detail.
  -- Example: "find the match with largest momentum swing" needs:
  --   Query 1: identify the match (returns match name)
  --   Query 2: over-by-over breakdown of THAT match (use a subquery or CTE)
  WITH identified AS (
    SELECT match, MAX(some_metric) - MIN(some_metric) AS swing
    FROM public.nv_play GROUP BY match ORDER BY swing DESC LIMIT 1
  )
  SELECT p.match, p.over,
    SUM(p.runs + p.extra_runs) AS runs_in_over,
    SUM(SUM(p.runs + p.extra_runs)) OVER (PARTITION BY p.match ORDER BY p.over) AS cumulative_runs,
    COUNT(DISTINCT CASE WHEN p.wicket IS NOT NULL THEN p.over ELSE NULL END) AS wickets_in_over
  FROM public.nv_play p
  JOIN identified i ON p.match = i.match AND p.innings = 1
  GROUP BY p.match, p.over
  ORDER BY p.over;
"""


@lru_cache(maxsize=1)
def build_baseline_queries_hint() -> str:
    return """PROACTIVE BASELINE STRATEGY:

  Player question      -> career_baseline + innings_list + question_specific
  Team question        -> team_batting + team_bowling + question_specific
  Recent form          -> career_baseline + last_N_innings + question_specific
  H2H                  -> h2h_summary + batter_career + bowler_career
  Leaderboard          -> main_leaderboard (no per-player baselines needed)

ALWAYS include innings_list when dismissal counts or averages are requested.

IDENTIFICATION + DETAIL RULE (critical for Q4-type questions):
  When the question asks to IDENTIFY a match/record AND understand what happened,
  ALWAYS produce two queries:
    1. identify_query  — finds the answer (match name, player, record)
    2. detail_query    — drills into that specific result (over-by-over, innings breakdown, etc.)
  Examples:
    "Find the match with largest swing" -> identify match + over-by-over breakdown of that match
    "Find the batter who scored most runs without..." -> identify player + their innings list
    "Find the bowler with best impact" -> identify bowler + their spell-by-spell breakdown
  Without the detail query, the insight has only a name/ID — not enough for meaningful analysis.

COMPOUND FORMULA QUERIES — decompose into CTEs:
  If the question involves a custom formula combining batting + bowling + fielding,
  ALWAYS use separate CTEs for each component (see pattern 9 in ADVANCED PATTERNS),
  then JOIN them. Never write one giant SELECT — it fails with syntax errors.
"""


# =========================================================
# QUERY PLAN GENERATION (PostgreSQL)
# =========================================================
def generate_query_plan(question: str,
                        relax_thresholds: bool = False,
                        intent: Optional[dict] = None,
                        session_context: Optional[list] = None,
                        error_feedback: Optional[str] = None) -> dict:

    intent_hint = build_intent_hint(intent)
    session_ctx = build_session_context(session_context or [])

    threshold_note = ""
    if relax_thresholds:
        threshold_note = (
            "RETRY MODE — previous attempt returned zero rows.\n"
            "- Do NOT add HAVING / minimum thresholds.\n"
            "- Return ALL rows regardless of sample size.\n"
            "- Broaden ILIKE patterns (use shorter fragments).\n"
            "- If you filtered by year, try without the year filter.\n"
        )

    feedback_note = ""
    if error_feedback:
        feedback_note = (
            f"PREVIOUS SQL FAILED WITH ERROR:\n{error_feedback}\n"
            "Fix the SQL using ONLY the schema columns listed below. "
            "Do not use unknown columns. Use PostgreSQL syntax.\n"
        )

    prompt = f"""
You are an expert cricket data analyst and SQL engineer.

TASK: Translate the user's cricket question into a precise multi-query PostgreSQL plan.

DATABASE:
  Table : public.nv_play
  Grain : ONE ROW = ONE DELIVERY in a cricket match
  Engine: PostgreSQL 14+

SCHEMA:
{build_schema_context()}

DERIVED METRICS:
{build_metrics_context()}

CRICKET RULES & BENCHMARKS:
{build_rules_context()}

CRICKET TERMINOLOGY -> SQL (READ CAREFULLY — especially dismissal formulas):
{build_cricket_terms_context()}

DATE RULES:
{build_date_context()}

FORMAT CONTEXT:
{build_format_context()}

ADVANCED PATTERNS:
{build_advanced_patterns_context()}

BASELINE STRATEGY:
{build_baseline_queries_hint()}

{threshold_note}
{feedback_note}
{intent_hint}
{session_ctx}

HARD RULES:
  1. SELECT or WITH...SELECT only.
  2. Use exact schema column names. NEVER invent columns.
  3. Every numeric metric: ROUND(value::numeric, 2).
  4. Every denominator: NULLIF(..., 0).
  5. wicket is TEXT: dismissed -> wicket IS NOT NULL.
  6. Booleans: legal_ball, free_hit, around_the_wicket, keeper_up.
  7. Names: ILIKE '%name%'.
  8. EXTRACT(YEAR FROM date)::int AS year.
  9. LIMIT 15-20 for open leaderboards. No LIMIT for named entities.
  10. Do NOT apply HAVING when a specific player/team is named.
  11. CRITICAL: Count dismissals with COUNT(DISTINCT CASE WHEN wicket IS NOT NULL
      THEN match || '-' || innings::text ELSE NULL END) — NEVER COUNT(*) FILTER (WHERE wicket IS NOT NULL).
  12. Always include an innings_list query when average or dismissal count is asked.
  13. HAVING FILTER SCOPE: Any HAVING clause (e.g. HAVING balls_faced >= 30) must filter on
      the SAME aggregation as the metric being measured. Never apply a global career filter
      when the query aggregates a specific phase or subset — a batter may have 30+ career balls
      but only 3 balls in the death overs. Filter in the WHERE or use a subquery.
  14. ALWAYS include COUNT(*) or a row count alongside any aggregate metric — never return
      a single percentage/rate without the sample size that produced it.
  15. MATCH WINNER / RESULT: To filter by which team won, look for columns like
      winning_team, result, match_result, or winner in the schema. If none exist,
      derive the winner by comparing innings totals: the team with the higher total in innings 2
      (or innings 1 if innings 2 never reached the target) won. Use a CTE:
      WITH match_totals AS (
        SELECT match, innings, batting_team,
          SUM(runs + extra_runs) AS total_runs
        FROM public.nv_play GROUP BY match, innings, batting_team
      ),
      match_winner AS (
        SELECT match,
          MAX(CASE WHEN innings=1 THEN batting_team END) AS team1,
          MAX(CASE WHEN innings=2 THEN batting_team END) AS team2,
          MAX(CASE WHEN innings=1 THEN total_runs END) AS total1,
          MAX(CASE WHEN innings=2 THEN total_runs END) AS total2,
          CASE WHEN MAX(CASE WHEN innings=2 THEN total_runs END) >
                    MAX(CASE WHEN innings=1 THEN total_runs END)
               THEN MAX(CASE WHEN innings=2 THEN batting_team END)
               ELSE MAX(CASE WHEN innings=1 THEN batting_team END)
          END AS winning_team
        FROM match_totals GROUP BY match
      )

OUTPUT — STRICT JSON, NO PREAMBLE, NO MARKDOWN:

{{
  "analysis_type": "single | multi",
  "intent": "1-2 precise sentences describing what is asked",
  "has_time_dimension": true | false,
  "has_phase_dimension": true | false,
  "has_comparison": true | false,
  "has_recent_form": true | false,
  "has_fantasy_dimension": true | false,
  "has_predictive_dimension": true | false,
  "has_time_based_dimension": true | false,
  "threshold_applied": true | false,
  "queries": [
    {{"name": "snake_case_name", "purpose": "what & why", "sql": "SELECT ..."}}
  ]
}}

USER QUESTION:
{question}
"""

    parsed = llm_json(prompt, cfg=PLAN_CONFIG)
    if not parsed or "queries" not in parsed:
        return {
            "analysis_type": "single",
            "queries": [{
                "name": "fallback_query",
                "purpose": "Fallback",
                "sql": generate_fallback_sql_pg(question),
            }],
        }
    return parsed


def generate_fallback_sql_pg(question: str) -> str:
    prompt = f"""
PostgreSQL expert. TABLE: public.nv_play (each row = one delivery).

SCHEMA: {build_schema_context()}
TERMINOLOGY: {build_cricket_terms_context()}
DATE RULES: {build_date_context()}

Return ONLY raw SQL — no markdown, no explanation.
SELECT only | PostgreSQL syntax | use schema columns only.
CRITICAL: Count dismissals with COUNT(DISTINCT CASE WHEN wicket IS NOT NULL THEN match || '-' || innings::text ELSE NULL END).

QUESTION: {question}
"""
    sql = clean_sql_postgres(llm(prompt, cfg=PLAN_CONFIG))
    if not sql.endswith(";"):
        sql += ";"
    return sql


# =========================================================
# EXECUTE SQL (PostgreSQL)
# =========================================================
def execute_sql_pg(sql: str) -> List[dict]:
    validate_sql(sql)
    conn = get_db_conn()
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        out = []
        for r in rows:
            row = {}
            for i, v in enumerate(r):
                if isinstance(v, float):
                    v = round(v, 2)
                row[cols[i]] = v
            out.append(row)
        cur.close()
        return out
    finally:
        release_db_conn(conn)


def execute_query_plan_pg(plan: dict, auto_repair: bool = True,
                          question: str = "",
                          intent: Optional[dict] = None,
                          session_context: Optional[list] = None) -> Tuple[list, list]:
    all_results = []
    all_sql = []
    for q in plan.get("queries", []):
        sql = clean_sql_postgres(q["sql"])
        if not sql.endswith(";"):
            sql += ";"
        try:
            results = execute_sql_pg(sql)
            all_results.append({"query_name": q["name"], "purpose": q["purpose"], "sql": sql, "results": results})
            all_sql.append(sql)
        except Exception as e:
            err = str(e)
            repaired_ok = False
            if auto_repair:
                try:
                    fixed_plan = generate_query_plan(
                        question or q.get("purpose", ""),
                        intent=intent,
                        session_context=session_context,
                        error_feedback=f"Query '{q['name']}' failed: {err}\nOriginal SQL:\n{sql}",
                    )
                    if fixed_plan.get("queries"):
                        fixed_sql = clean_sql_postgres(fixed_plan["queries"][0]["sql"])
                        if not fixed_sql.endswith(";"):
                            fixed_sql += ";"
                        results = execute_sql_pg(fixed_sql)
                        all_results.append({
                            "query_name": q["name"] + "_repaired",
                            "purpose": q["purpose"],
                            "sql": fixed_sql,
                            "results": results,
                        })
                        all_sql.append(fixed_sql)
                        repaired_ok = True
                except Exception:
                    repaired_ok = False
            if not repaired_ok:
                all_results.append({
                    "query_name": q["name"], "purpose": q["purpose"],
                    "sql": sql, "results": [], "error": err,
                })
    return all_results, all_sql


def execute_query_plan_with_retry(plan: dict, question: str,
                                  intent: Optional[dict] = None,
                                  session_context: Optional[list] = None) -> Tuple[list, list, bool]:
    results, sql_list = execute_query_plan_pg(
        plan, auto_repair=True, question=question, intent=intent, session_context=session_context
    )

    if not results_are_empty(results):
        return results, sql_list, False

    retry_plan = {
        "analysis_type": plan.get("analysis_type", "single"),
        "queries": [
            {"name": q["name"] + "_relaxed",
             "purpose": q["purpose"] + " (thresholds relaxed)",
             "sql": strip_having_clauses(q["sql"])}
            for q in plan.get("queries", [])
        ],
    }
    results2, sql2 = execute_query_plan_pg(
        retry_plan, auto_repair=False, question=question, intent=intent, session_context=session_context
    )
    if not results_are_empty(results2):
        return results2, sql2, True

    fresh_plan = generate_query_plan(
        question, relax_thresholds=True, intent=intent, session_context=session_context
    )
    results3, sql3 = execute_query_plan_pg(
        fresh_plan, auto_repair=True, question=question, intent=intent, session_context=session_context
    )
    return results3, sql3, True


# =========================================================
# EXCEL: LOAD + PROFILE
# =========================================================
def load_dataframe(file_base64: str, file_ext: str) -> pd.DataFrame:
    raw = base64.b64decode(file_base64)
    buf = io.BytesIO(raw)
    ext = file_ext.lower().strip()

    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(buf, engine="openpyxl")
    elif ext == ".csv":
        df = pd.read_csv(buf)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]

    for col in ["legal_ball", "free_hit", "around_the_wicket", "keeper_up"]:
        if col in df.columns:
            try:
                df[col] = df[col].astype(bool)
            except Exception:
                df[col] = df[col].map(lambda x: str(x).strip().lower() in ("true", "1", "yes", "t"))

    for col in ["date", "timestamp"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in ["runs", "extra_runs", "over", "ball", "innings", "batting_position"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def profile_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    profile = {
        "row_count": int(len(df)),
        "columns": [],
        "match_types": [],
        "competitions": [],
        "date_range": None,
        "unique_batters": 0,
        "unique_bowlers": 0,
        "unique_teams": 0,
    }

    for c in df.columns:
        s = df[c]
        col_info = {
            "name": c,
            "dtype": str(s.dtype),
            "null_pct": round(float(s.isna().mean() * 100), 1),
            "n_unique": int(s.nunique(dropna=True)),
        }
        if s.dtype == object and col_info["n_unique"] <= 25:
            try:
                col_info["sample_values"] = [str(v) for v in s.dropna().unique()[:25].tolist()]
            except Exception:
                pass
        profile["columns"].append(col_info)

    if "match_type" in df.columns:
        profile["match_types"] = [str(v) for v in df["match_type"].dropna().unique().tolist()[:20]]
    if "competition" in df.columns:
        profile["competitions"] = [str(v) for v in df["competition"].dropna().unique().tolist()[:20]]
    if "date" in df.columns and df["date"].notna().any():
        profile["date_range"] = [str(df["date"].min().date()), str(df["date"].max().date())]
    if "batter" in df.columns:
        profile["unique_batters"] = int(df["batter"].nunique())
    if "bowler" in df.columns:
        profile["unique_bowlers"] = int(df["bowler"].nunique())
    if "batting_team" in df.columns:
        profile["unique_teams"] = int(df["batting_team"].nunique())

    return profile


def resolve_entities_in_df(df: pd.DataFrame, candidate_names: List[str]) -> Dict[str, List[str]]:
    resolved: Dict[str, List[str]] = {}
    if not candidate_names:
        return resolved

    name_cols = [c for c in ["batter", "bowler", "fielder", "non_striker"] if c in df.columns]
    team_cols = [c for c in ["batting_team", "bowling_team", "team"] if c in df.columns]

    def find_matches(name: str, cols: list) -> List[str]:
        matches = set()
        name_low = name.lower()
        for col in cols:
            try:
                vals = df[col].dropna().astype(str).unique()
                for v in vals:
                    if name_low in v.lower() or v.lower() in name_low:
                        matches.add(v)
            except Exception:
                continue
        return list(matches)[:5]

    for name in candidate_names:
        player_hits = find_matches(name, name_cols)
        team_hits = find_matches(name, team_cols)
        if player_hits or team_hits:
            resolved[name] = list(set(player_hits + team_hits))[:5]
    return resolved


# =========================================================
# EXCEL: DUCKDB QUERY PLAN — with fixed dismissal formula
# =========================================================
@lru_cache(maxsize=1)
def build_cricket_terms_duckdb() -> str:
    """DuckDB-specific version of the cricket terms, with correct dismissal formula."""
    return """CRICKET TERMINOLOGY -> DUCKDB SQL:

BATTING POSITIONS:
  opener/openers -> batting_position IN (1,2)
  top order -> batting_position IN (1,2,3)
  middle order -> batting_position IN (4,5,6)

PHASES (T20): powerplay = over 1-6 | middle = 7-15 | death = 16-20
PHASES (ODI): powerplay = 1-10 | middle = 11-40 | death = 41-50

DELIVERY:
  dot ball -> (runs+extra_runs)=0 AND legal_ball=TRUE
  boundary -> runs IN (4,6) | six -> runs=6 | four -> runs=4
  dismissed -> wicket IS NOT NULL | not out -> wicket IS NULL

=== CRITICAL: DISMISSAL COUNTING IN BALL-BY-BALL DATA (DuckDB) ===

ONE ROW = ONE DELIVERY. 'wicket' is populated ONLY on the delivery where dismissal occurred.

  ✅ CORRECT dismissal count:
     COUNT(DISTINCT CASE WHEN wicket IS NOT NULL
           THEN match || '-' || CAST(innings AS VARCHAR)
           ELSE NULL END)

  ✅ CORRECT innings played:
     COUNT(DISTINCT match || '-' || CAST(innings AS VARCHAR))

  ❌ WRONG (counts delivery rows, not innings):
     COUNT(*) FILTER (WHERE wicket IS NOT NULL)
     SUM(CASE WHEN wicket IS NOT NULL THEN 1 ELSE 0 END)

DUCKDB FORMULAS:

  batting_avg =
    ROUND(SUM(runs) / NULLIF(
      COUNT(DISTINCT CASE WHEN wicket IS NOT NULL
            THEN match || '-' || CAST(innings AS VARCHAR) ELSE NULL END)
    , 0), 2)

  strike_rate =
    ROUND((CAST(SUM(runs) AS DOUBLE) / NULLIF(
      SUM(CASE WHEN legal_ball = TRUE THEN 1 ELSE 0 END), 0)) * 100, 2)

  economy =
    ROUND((CAST(SUM(runs + extra_runs) AS DOUBLE) / NULLIF(
      SUM(CASE WHEN legal_ball = TRUE THEN 1 ELSE 0 END), 0)) * 6, 2)

  innings_played =
    COUNT(DISTINCT match || '-' || CAST(innings AS VARCHAR))

  dismissals =
    COUNT(DISTINCT CASE WHEN wicket IS NOT NULL
          THEN match || '-' || CAST(innings AS VARCHAR) ELSE NULL END)

  not_outs =
    COUNT(DISTINCT match || '-' || CAST(innings AS VARCHAR))
    - COUNT(DISTINCT CASE WHEN wicket IS NOT NULL
            THEN match || '-' || CAST(innings AS VARCHAR) ELSE NULL END)

DUCKDB SYNTAX RULES:
  - Table: nv_play (no schema prefix)
  - NEVER use ::numeric or ::int — use CAST(x AS DOUBLE) or CAST(x AS INTEGER)
  - String concat: col1 || '-' || CAST(col2 AS VARCHAR)
  - Booleans: legal_ball = TRUE
  - Strings: ILIKE '%name%'
  - No DATEADD / YEAR() — use EXTRACT(YEAR FROM date)
  - FILTER syntax: COUNT(*) FILTER (WHERE condition) is supported
"""


def generate_duckdb_query_plan(question: str,
                                profile: Dict[str, Any],
                                resolved_entities: Dict[str, List[str]],
                                intent: Optional[dict] = None,
                                error_feedback: Optional[str] = None) -> dict:

    columns = [c["name"] for c in profile["columns"]]
    available_cols = ", ".join(columns)

    profile_summary_lines = [f"  Rows: {profile['row_count']}"]
    if profile["match_types"]:
        profile_summary_lines.append(f"  Match types in file: {profile['match_types']}")
    if profile["competitions"]:
        profile_summary_lines.append(f"  Competitions: {profile['competitions'][:10]}")
    if profile["date_range"]:
        profile_summary_lines.append(f"  Date range: {profile['date_range'][0]} -> {profile['date_range'][1]}")
    if profile["unique_batters"]:
        profile_summary_lines.append(f"  Unique batters: {profile['unique_batters']}")
    if profile["unique_bowlers"]:
        profile_summary_lines.append(f"  Unique bowlers: {profile['unique_bowlers']}")

    col_hints = []
    for col in profile["columns"]:
        line = f"  {col['name']} ({col['dtype']}, nulls={col['null_pct']}%, unique={col['n_unique']})"
        if "sample_values" in col:
            line += f" sample={col['sample_values'][:6]}"
        col_hints.append(line)

    entity_hint = ""
    if resolved_entities:
        entity_hint = "\nENTITY RESOLUTION (use these EXACT values in ILIKE patterns):\n"
        for asked, hits in resolved_entities.items():
            entity_hint += f"  '{asked}' -> {hits}\n"
    elif intent and intent.get("candidate_names"):
        entity_hint = (f"\nNote: question mentions {intent['candidate_names']} but no matches found in file. "
                       "Use broad ILIKE patterns.\n")

    intent_hint = build_intent_hint(intent)

    feedback = ""
    if error_feedback:
        feedback = f"\nPREVIOUS SQL FAILED:\n{error_feedback}\nFix using only the columns above.\n"

    prompt = (
        "You are an expert cricket data analyst and SQL engineer.\n\n"
        "TASK: Write a precise multi-query DuckDB SQL plan for the user question.\n"
        "Data is in a DuckDB in-memory table called: nv_play\n"
        "One row = one delivery (ball).\n\n"
        "AVAILABLE COLUMNS:\n" + available_cols + "\n\n"
        "COLUMN PROFILE:\n" + "\n".join(col_hints) + "\n\n"
        "FILE SUMMARY:\n" + "\n".join(profile_summary_lines) + "\n"
        + entity_hint + intent_hint + feedback
        + "\nCRICKET TERMINOLOGY AND FORMULAS (DuckDB):\n"
        + build_cricket_terms_duckdb()
        + "\nCRICKET BENCHMARKS:\n" + build_rules_context()
        + "\nFORMAT CONTEXT:\n" + build_format_context()
        + """
QUERY STRATEGY:
  - Player named -> career_baseline + innings_list + question_specific
  - Averages/dismissals asked -> always include innings_list
  - Phase breakdown asked -> add phase_split query
  - Trend asked -> add yearly_trend query (if date column exists)
  - Identification question -> add detail_query for the identified item

HARD SQL RULES (DuckDB):
  1. Table: nv_play (NEVER public.nv_play)
  2. NEVER ::numeric or ::int — use CAST(x AS DOUBLE) or CAST(x AS INTEGER)
  3. Dismissals: COUNT(DISTINCT CASE WHEN wicket IS NOT NULL
         THEN match || '-' || CAST(innings AS VARCHAR) ELSE NULL END)
     NEVER COUNT(*) FILTER (WHERE wicket IS NOT NULL)
  4. Innings played: COUNT(DISTINCT match || '-' || CAST(innings AS VARCHAR))
  5. Use ONLY columns from AVAILABLE COLUMNS. Skip dimensions if column missing.
  6. Named entities: use resolved values with ILIKE '%value%'
  7. HAVING filter scope: apply HAVING on the SAME aggregation as the metric.
     Do NOT filter on global career balls when measuring phase-specific balls.
  8. Always include sample size (ball count or innings count) alongside every rate/%.
  9. Match winner (if needed): derive from innings totals —
     team batting second wins if their total > team batting first total.
     Use a CTE: WITH match_totals AS (SELECT match, innings, batting_team,
       SUM(runs+extra_runs) AS total FROM nv_play GROUP BY match,innings,batting_team),
     winner AS (SELECT match,
       MAX(CASE WHEN innings=1 THEN batting_team END) AS batting_first_team,
       MAX(CASE WHEN innings=2 THEN batting_team END) AS chasing_team,
       MAX(CASE WHEN innings=1 THEN total END) AS total1,
       MAX(CASE WHEN innings=2 THEN total END) AS total2,
       CASE WHEN MAX(CASE WHEN innings=2 THEN total END) >
                 MAX(CASE WHEN innings=1 THEN total END)
            THEN MAX(CASE WHEN innings=2 THEN batting_team END)
            ELSE MAX(CASE WHEN innings=1 THEN batting_team END) END AS winning_team
     FROM match_totals GROUP BY match)
  10. Required run rate (if needed): at start of each over in chase, compute
      (target - runs_so_far) / overs_remaining * (balls_per_over).

OUTPUT — STRICT JSON ONLY, NO PREAMBLE, NO MARKDOWN FENCES:
{"analysis_type": "single|multi", "intent": "1-2 sentences", "queries": [{"name": "snake_case", "purpose": "what and why", "sql": "SELECT ..."}]}

USER QUESTION: """ + question
    )

    parsed = llm_json(prompt, cfg=PLAN_CONFIG)
    if not parsed or "queries" not in parsed:
        fb_sql = "SELECT * FROM nv_play LIMIT 20"
        if "batter" in columns and "runs" in columns:
            innings_col = "match || '-' || CAST(innings AS VARCHAR)" if "innings" in columns else "match"
            fb_sql = (
                f"SELECT batter, "
                f"COUNT(DISTINCT match) AS matches, "
                f"COUNT(DISTINCT {innings_col}) AS innings_played, "
                f"SUM(runs) AS total_runs, "
                f"COUNT(DISTINCT CASE WHEN wicket IS NOT NULL THEN {innings_col} ELSE NULL END) AS dismissals, "
                f"ROUND(CAST(SUM(runs) AS DOUBLE) / NULLIF(COUNT(DISTINCT CASE WHEN wicket IS NOT NULL THEN {innings_col} ELSE NULL END), 0), 2) AS batting_avg "
                f"FROM nv_play GROUP BY batter ORDER BY total_runs DESC LIMIT 20"
            )
        return {
            "analysis_type": "single",
            "queries": [{"name": "fallback_query", "purpose": "General analysis", "sql": fb_sql}],
        }
    return parsed


def execute_duckdb_plan(plan: dict, df: pd.DataFrame, auto_repair: bool = True,
                        question: str = "", profile: Optional[dict] = None,
                        resolved_entities: Optional[dict] = None,
                        intent: Optional[dict] = None) -> list:
    con = duckdb.connect(database=":memory:")
    con.register("nv_play", df)
    all_results = []

    def run_one(name: str, purpose: str, sql: str, allow_repair: bool):
        sql_clean = clean_sql_duckdb(sql)
        if not sql_clean.endswith(";"):
            sql_clean += ";"
        try:
            res = con.execute(sql_clean).fetchdf()
            recs = []
            for row in res.to_dict(orient="records"):
                clean = {}
                for k, v in row.items():
                    if hasattr(v, "item"):
                        try:
                            v = v.item()
                        except Exception:
                            pass
                    if hasattr(v, "isoformat"):
                        v = str(v)
                    if isinstance(v, float):
                        if v != v:
                            v = None
                        else:
                            v = round(v, 2)
                    clean[k] = v
                recs.append(clean)
            return {"query_name": name, "purpose": purpose, "sql": sql_clean, "results": recs}
        except Exception as e:
            err = str(e)
            if allow_repair and auto_repair and profile is not None:
                try:
                    fixed_plan = generate_duckdb_query_plan(
                        question, profile=profile,
                        resolved_entities=resolved_entities or {},
                        intent=intent,
                        error_feedback=f"Query '{name}' failed: {err}\nSQL was:\n{sql_clean}",
                    )
                    if fixed_plan.get("queries"):
                        return run_one(
                            name + "_repaired",
                            purpose,
                            fixed_plan["queries"][0]["sql"],
                            allow_repair=False,
                        )
                except Exception:
                    pass
            return {"query_name": name, "purpose": purpose, "sql": sql_clean, "results": [], "error": err}

    for q in plan.get("queries", []):
        all_results.append(run_one(q["name"], q["purpose"], q["sql"], allow_repair=True))

    con.close()
    return all_results


# =========================================================
# CHART CONFIG
# =========================================================
def generate_chart_config(question: str, query_results: list, intent: Optional[dict] = None):
    best = None
    best_size = 0
    for r in query_results:
        rows = r.get("results", [])
        if rows and len(rows) > best_size:
            best = r
            best_size = len(rows)
    if best is None:
        return None

    compact = json.dumps({"primary": best, "context": [r for r in query_results if r is not best][:2]},
                         default=str)[:8000]

    if intent is None:
        intent = classify_intent(question)

    hints = []
    if intent.get("is_trend"):        hints.append("TIME: x_key='year', use 'line' for 1 metric, 'bar' for >=2")
    if intent.get("is_phase"):        hints.append("PHASE: x_key='phase' (PP/Middle/Death), use 'bar'")
    if intent.get("is_comparison"):   hints.append("COMPARE: 2-5 entities, use 'bar' or 'radar'")
    if intent.get("is_over_by_over"): hints.append("OVERS: x_key='over', use 'line'")
    if intent.get("is_consistency"):  hints.append("DISTRIBUTION: x_key='score_band', use 'bar_colored'")
    if intent.get("is_milestone"):    hints.append("MILESTONE: x_key='year', use 'bar'")
    if intent.get("is_venue"):        hints.append("VENUE: x_key='venue', use 'bar_colored'")
    if any(w in question.lower() for w in ["dismissal","how out","bowled","caught","lbw","wicket type"]):
        hints.append("DISMISSAL: use 'pie' for share, 'bar_colored' for count")

    hints_text = "\n".join(hints) if hints else "(no specific hint)"

    prompt = f"""
Cricket data visualisation expert. Select the best chart type.
Return null if no chart adds meaningful value (e.g. single number).

HINTS:
{hints_text}

CHART TYPES:
  bar         : Multi-metric comparison (2+ y_keys). Max 12 rows.
  bar_colored : Single-metric leaderboard. Max 15 rows.
  line        : Time/ordered trend. Max 25 rows.
  area        : Cumulative trend. Max 25 rows.
  pie         : Share of whole. 1 y_key. 3-8 categories.
  radar       : Multi-dimension profile. 2-4 entities, 4-6 metrics.

RULES:
  - y_keys: NUMERIC only (exclude IDs, names).
  - title: specific, include entity name.
  - Use the primary (densest) result set.

RETURN STRICT JSON (no preamble, no markdown):
{{
  "chart_type": "bar|bar_colored|line|area|pie|radar",
  "title": "...",
  "subtitle": "...",
  "x_key": "column_name",
  "y_keys": ["metric1","metric2"],
  "data": [{{"x_value": ..., "metric1": ..., "metric2": ...}}]
}}

Return exactly: null   if no chart is appropriate.

USER QUESTION: {question}
DATABASE RESULTS: {compact}
"""

    raw = llm(prompt, cfg=CHART_CONFIG_CFG)
    raw = raw.replace("```json", "").replace("```", "").strip()
    if raw.lower().strip() == "null":
        return None
    m = re.search(r'\{[\s\S]*\}', raw)
    if m:
        raw = m.group(0)
    try:
        return json.loads(raw)
    except Exception:
        return None


# =========================================================
# INSIGHT GENERATION — DATA-FAITHFUL, NO TRUNCATION
# =========================================================
def generate_cricket_insight(question: str,
                             query_results: list,
                             small_sample: bool = False,
                             intent: Optional[dict] = None,
                             session_context: Optional[list] = None,
                             file_context: Optional[dict] = None) -> str:

    session_ctx = build_session_context(session_context or [])

    # ── Truncate large result sets before building the prompt ─────────────────
    MAX_ROWS_IN_PROMPT = 20
    truncated_results = []
    any_truncated = False
    for r in query_results:
        rows = r.get("results", [])
        if len(rows) > MAX_ROWS_IN_PROMPT:
            truncated = dict(r)
            truncated["results"] = rows[:MAX_ROWS_IN_PROMPT]
            truncated["_truncated_to"] = MAX_ROWS_IN_PROMPT
            truncated["_total_rows"]   = len(rows)
            truncated_results.append(truncated)
            any_truncated = True
        else:
            truncated_results.append(r)

    compact_data = json.dumps(truncated_results, default=str)[:18000]

    # ── Result shape detection ────────────────────────────────────────────────
    all_failed = all(
        r.get("error") or len(r.get("results", [])) == 0
        for r in query_results
    )
    errors = [str(r.get("error", "")) for r in query_results if r.get("error")]
    has_partial_zero = (not all_failed) and any(
        len(r.get("results", [])) == 0 for r in query_results
    )
    total_result_rows = sum(len(r.get("results", [])) for r in query_results)
    is_tiny_result    = (total_result_rows <= 3) and not all_failed
    is_large_result   = any_truncated

    # ── Context blocks ────────────────────────────────────────────────────────
    file_block = ""
    if file_context:
        file_block = (
            "\nDATA SOURCE: User-uploaded spreadsheet."
            f"\n  Rows: {file_context.get('row_count')}"
            f" | Match types: {file_context.get('match_types')}"
            f" | Date range: {file_context.get('date_range')}"
            f" | Competitions: {file_context.get('competitions')}"
            "\nAnalyse ONLY this data. Do not reference external matches.\n"
        )

    truncation_note = ""
    if any_truncated:
        truncation_note = (
            f"\nDATA NOTE: Some result sets were capped at {MAX_ROWS_IN_PROMPT} rows "
            "(_total_rows shows the real count). Acknowledge the full count in your analysis "
            "and summarise the full distribution — do not pretend only the shown rows exist.\n"
        )

    small_sample_note = (
        "\nNOTE: Small sample size — mention caveats naturally.\n" if small_sample else ""
    )

    # ── Dynamic analysis section based on intent ─────────────────────────────
    analysis_section_label = "Performance Analysis"
    if intent:
        if intent.get("is_bowling") and not intent.get("is_batting"):
            analysis_section_label = "Bowling Performance"
        elif intent.get("is_batting") and not intent.get("is_bowling"):
            analysis_section_label = "Batting Performance"
        elif intent.get("is_allrounder"):
            analysis_section_label = "All-Round Performance"
        elif intent.get("is_comparison"):
            analysis_section_label = "Comparison"
        elif intent.get("is_leaderboard"):
            analysis_section_label = "Rankings"
        elif intent.get("is_trend"):
            analysis_section_label = "Trend Analysis"
        elif intent.get("is_phase"):
            analysis_section_label = "Phase Breakdown"
        elif intent.get("is_chase"):
            analysis_section_label = "Chase vs Setting"
        elif intent.get("is_venue"):
            analysis_section_label = "Venue Analysis"

    # ── Special case: all queries failed ─────────────────────────────────────
    if all_failed:
        error_detail = "; ".join(errors) if errors else "no data returned"
        failed_prompt = (
            "You are an expert cricket analyst.\n\n"
            "The user asked: " + question + "\n\n"
            "Unfortunately the data retrieval failed with this error: " + error_detail + "\n\n"
            "Write a response with EXACTLY this structure:\n\n"
            "---\n\n"
            "## 🏏 [Descriptive headline about what was attempted]\n\n"
            "### ⚠️ Data Retrieval Issue\n"
            "In 2-3 plain sentences (no technical jargon): explain what was asked, "
            "that the data could not be retrieved, and what specific information "
            "would be needed to answer it.\n\n"
            "### 💡 Suggested Next Steps\n"
            "- One specific simpler rephrasing of the question that might work\n"
            "- One alternative related question that could be answered\n\n"
            "### 🔗 Related Analyses\n"
            "2 alternative questions that are achievable with ball-by-ball cricket data.\n\n"
            "---\n\n"
            "Keep total response under 200 words. Do not mention SQL or databases."
        )
        return llm(failed_prompt, cfg=INSIGHT_CONFIG)

    # ── Build the main prompt ─────────────────────────────────────────────────
    tiny_instruction = (
        "RESPONSE LENGTH: This is a small/single-stat result. Write a CONCISE response:\n"
        "- Headline: specific with the actual number/name\n"
        "- Key Numbers: show the data clearly (table if multiple columns, single line if one number)\n"
        "- Analysis: 3-5 sentences maximum. Answer directly. No padding.\n"
        "- Verdict: 2-3 sentences.\n"
        "- Additional Context: include all 5 sections but keep each to 1-2 sentences.\n"
        "- Total length: 200-350 words maximum.\n"
    ) if is_tiny_result else (
        "RESPONSE LENGTH: Write a comprehensive analysis:\n"
        "- Total length: 600-900 words.\n"
        "- Every section must be substantive and data-grounded.\n"
        "- Complete ALL sections — do not stop early.\n"
    )

    large_instruction = (
        "LARGE DATASET NOTE: The table was capped at " + str(MAX_ROWS_IN_PROMPT) + " rows for display. "
        "In your analysis, explicitly state the full count from _total_rows, "
        "identify the top 3 performers clearly, and describe the overall distribution range.\n"
    ) if is_large_result else ""

    partial_note = (
        "PARTIAL DATA NOTE: One or more specific filters returned 0 rows. "
        "State clearly which criteria found no data (e.g. 'No batter met the RRR > 10 threshold'). "
        "Then report the available broader results. Do NOT write hypothetical analysis.\n"
    ) if has_partial_zero else ""

    prompt = (
        "You are an expert cricket analyst. Answer the user's question using ONLY the data "
        "in the query results below. Be accurate, direct, and insightful.\n\n"
        + file_block
        + small_sample_note
        + truncation_note
        + session_ctx
        + "\n\nUSER QUESTION:\n" + question
        + "\n\nQUERY RESULTS (exact numbers — do not alter or invent any figures):\n"
        + compact_data
        + "\n\n" + tiny_instruction
        + large_instruction
        + partial_note
        + """
CRITICAL RULES (follow every one):
1. ACCURACY: Every number must come directly from the query results. Never invent or estimate.
2. FORMAT DETECTION: Check for match_type in the results.
   - If T20/T20I found -> apply T20 benchmarks.
   - If ODI/List A/50-over found -> apply ODI benchmarks.
   - If Test/First Class found -> apply Test benchmarks.
   - If only division names (Division 3A, Premier Division, etc.) or no match_type -> 
     write "Format: local/club cricket — standard benchmarks not applied."
3. DISMISSALS: If innings_list is in the results, use it as ground truth for dismissal counts.
4. NO PADDING: Every sentence must reference a specific number or fact from the data.
   Do not write generic cricket knowledge not supported by the data.
5. NO TRUNCATION: Complete every section. Never stop mid-sentence or mid-table.
6. SAMPLE SIZE: Always state how many matches/innings/balls the conclusion is based on.
7. Do not mention SQL, database, queries, or column names.

REQUIRED OUTPUT FORMAT (complete every section, never omit or abbreviate):

---

## 🏏 [Specific headline — include the actual name/number/match found]

### 📊 Key Numbers
Present ALL relevant data as a clean markdown table.
If the dataset was truncated, show the top """ + str(MAX_ROWS_IN_PROMPT) + """ rows and state the full count.
Column headers must be readable (no underscores). Round all decimals to 2 places.

### 🔍 Analysis

**Summary**
2-3 sentences directly answering the question with exact numbers from the data.
State format detected and sample size.

**""" + analysis_section_label + """**
Detailed analysis of the key metrics. Compare against benchmarks only if format was detected.
For leaderboards: identify top 3 clearly, note the gap between 1st and 2nd.
For trends: describe direction, peak, and trough with specific values.
For phase analysis: cover each phase separately with its numbers.

**Notable Points**
3-5 bullet points. Each bullet must cite a specific number from the data.
No bullet should contain generic cricket statements without a supporting number.

### 💡 Verdict
3-5 sentences. Direct expert conclusion grounded entirely in the data.
State tier classification only if format was confirmed.

---

### ℹ️ Additional Context

**🎯 Query Context**
What was asked | type of analysis | format detected | sample size (matches/innings/balls) | any filters broadened.

**📅 Data Coverage**
Date range from the data | competitions covered | any notable gaps.

**📐 Benchmarks Applied**
Format: [detected or "not applicable — local cricket"] | List each metric with its tier thresholds used.
If format not detected, write: "Standard T20/ODI/Test benchmarks not applied to this dataset."

**📝 Summary**
2-3 sentences: direct answer to the question + one key supporting fact.

**🔗 Related Analyses**
Exactly 2 specific follow-up questions using the actual player/team names and format from this data.

---
"""
    )

    return llm(prompt, cfg=INSIGHT_CONFIG)



# =========================================================
# INTENT SUMMARY
# =========================================================
def generate_intent_summary(question: str, intent: dict) -> str:
    label_map = {
        "is_batting":     "Batting analysis",
        "is_bowling":     "Bowling analysis",
        "is_fielding":    "Fielding analysis",
        "is_team":        "Team analysis",
        "is_h2h":         "Head-to-head matchup",
        "is_allrounder":  "All-rounder assessment",
        "is_form":        "Recent form",
        "is_trend":       "Year-by-year trend",
        "is_phase":       "Phase breakdown (PP/Middle/Death)",
        "is_comparison":  "Player/team comparison",
        "is_opponent":    "Opponent-specific analysis",
        "is_consistency": "Consistency & milestones",
        "is_chase":       "Chase vs Setting analysis",
        "is_pressure":    "Pressure/clutch performance",
        "is_leaderboard": "Rankings/leaderboard",
        "is_milestone":   "Career milestones",
        "is_over_by_over":"Over-by-over breakdown",
        "is_venue":       "Venue analysis",
        "is_fantasy":     "Fantasy cricket recommendation",
        "is_predictive":  "Predictive insight",
        "is_time_based":  "Time / duration / pace-of-play",
    }
    flags = [v for k, v in label_map.items() if intent.get(k)]
    fmt = []
    if intent.get("fmt_t20"):  fmt.append("T20")
    if intent.get("fmt_odi"):  fmt.append("ODI")
    if intent.get("fmt_test"): fmt.append("Test")

    parts = []
    if flags: parts.append(", ".join(flags))
    if fmt:   parts.append(f"Format: {'/'.join(fmt)}")
    if intent.get("candidate_names"):
        parts.append(f"Entities: {', '.join(intent['candidate_names'])}")
    return " | ".join(parts) if parts else "General cricket query"


# =========================================================
# FORMAT RESULTS (markdown table)
# =========================================================
def format_results(results: list) -> str:
    if not results:
        return "No results found."
    headers = list(results[0].keys())
    rows = [" | ".join(headers), "-" * max(8, len(" | ".join(headers)))]
    for r in results:
        rows.append(" | ".join(str(v) for v in r.values()))
    return "\n".join(rows)


# =========================================================
# API ROUTES
# =========================================================
@app.get("/health")
def health_check():
    try:
        conn = get_db_conn()
        release_db_conn(conn)
        return {"status": "ok", "database": "connected", "version": APP_VERSION}
    except Exception as e:
        return {"status": "error", "message": str(e), "traceback": traceback.format_exc()}


@app.get("/version")
def version():
    import datetime
    return {
        "version": APP_VERSION,
        "app": "Cricket_Scorer_AI",
        "model": "gemini-2.5-flash",
        "status": "running",
        "server_time": datetime.datetime.utcnow().isoformat() + "Z",
    }


@app.get("/gemini-test")
def gemini_test():
    return {"response": llm("Say hello from Cricket_Scorer_AI")}


@app.post("/generate-sql")
def generate_sql(req: QueryRequest):
    try:
        intent = classify_intent(req.question)
        return {
            "question": req.question,
            "intent_flags": {k: v for k, v in intent.items() if v is True},
            "intent_summary": generate_intent_summary(req.question, intent),
            "query_plan": generate_query_plan(
                req.question, intent=intent, session_context=req.session_context
            ),
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "traceback": traceback.format_exc()}


@app.post("/ask")
def ask_question(req: QueryRequest):
    try:
        intent = classify_intent(req.question)
        intent_summary = generate_intent_summary(req.question, intent)

        plan = generate_query_plan(req.question, intent=intent, session_context=req.session_context)
        results, sql_list, relaxed = execute_query_plan_with_retry(
            plan, req.question, intent=intent, session_context=req.session_context
        )

        tables = [
            {"query_name": r["query_name"], "purpose": r["purpose"], "table": format_results(r["results"])}
            for r in results
        ]
        chart_cfg = generate_chart_config(req.question, results, intent=intent)
        insight = generate_cricket_insight(
            req.question, results, small_sample=relaxed,
            intent=intent, session_context=req.session_context,
        )

        return {
            "question": req.question,
            "intent_summary": intent_summary,
            "intent_flags": {
                "analysis_type":       plan.get("analysis_type", "single"),
                "has_time_dimension":  plan.get("has_time_dimension", False),
                "has_phase_dimension": plan.get("has_phase_dimension", False),
                "has_comparison":      plan.get("has_comparison", False),
                "has_recent_form":     plan.get("has_recent_form", False),
                "has_fantasy":         plan.get("has_fantasy_dimension", False),
                "has_predictive":      plan.get("has_predictive_dimension", False),
                "has_time_based":      plan.get("has_time_based_dimension", False),
            },
            "thresholds_relaxed": relaxed,
            "sql_queries": sql_list,
            "results": results,
            "tables": tables,
            "chart_config": chart_cfg,
            "insight": insight,
            "session_turn": {"question": req.question, "summary": (insight[:400] if insight else "")},
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "traceback": traceback.format_exc()}


@app.post("/ask-excel")
def ask_excel(req: ExcelQueryRequest):
    try:
        intent = classify_intent(req.question)
        intent_summary = generate_intent_summary(req.question, intent)

        df = load_dataframe(req.file_base64, req.file_ext)
        if df.empty:
            return {
                "status": "error",
                "message": "Uploaded file is empty after parsing.",
                "file_name": req.file_name,
            }
        profile = profile_dataframe(df)
        resolved = resolve_entities_in_df(df, intent.get("candidate_names", []))

        plan = generate_duckdb_query_plan(
            req.question, profile=profile, resolved_entities=resolved, intent=intent
        )

        results = execute_duckdb_plan(
            plan, df, auto_repair=True, question=req.question,
            profile=profile, resolved_entities=resolved, intent=intent,
        )

        all_empty = all(len(q.get("results", [])) == 0 for q in results)
        if all_empty:
            relaxed_plan = {
                "analysis_type": plan.get("analysis_type", "single"),
                "queries": [
                    {"name": q["name"] + "_relaxed",
                     "purpose": q["purpose"] + " (thresholds relaxed)",
                     "sql": strip_having_clauses(q["sql"])}
                    for q in plan.get("queries", [])
                ],
            }
            results = execute_duckdb_plan(
                relaxed_plan, df, auto_repair=False, question=req.question,
                profile=profile, resolved_entities=resolved, intent=intent,
            )
            all_empty = all(len(q.get("results", [])) == 0 for q in results)

            if all_empty:
                fresh = generate_duckdb_query_plan(
                    req.question, profile=profile, resolved_entities=resolved, intent=intent,
                    error_feedback="All previous queries returned 0 rows. Broaden filters and remove HAVING.",
                )
                results = execute_duckdb_plan(
                    fresh, df, auto_repair=True, question=req.question,
                    profile=profile, resolved_entities=resolved, intent=intent,
                )
                all_empty = all(len(q.get("results", [])) == 0 for q in results)

        chart_cfg = generate_chart_config(req.question, results, intent=intent)
        insight = generate_cricket_insight(
            req.question, results,
            small_sample=all_empty, intent=intent, session_context=[],
            file_context=profile,
        )

        tables = [
            {"query_name": r["query_name"], "purpose": r["purpose"], "table": format_results(r["results"])}
            for r in results
        ]
        sql_queries = [q["sql"] for q in results]

        return {
            "question": req.question,
            "intent_summary": intent_summary,
            "intent_flags": {
                "analysis_type":       plan.get("analysis_type", "single"),
                "has_time_dimension":  intent.get("is_trend", False),
                "has_phase_dimension": intent.get("is_phase", False),
                "has_comparison":      intent.get("is_comparison", False),
                "has_recent_form":     intent.get("is_form", False),
                "has_fantasy":         intent.get("is_fantasy", False),
                "has_predictive":      intent.get("is_predictive", False),
                "has_time_based":      intent.get("is_time_based", False),
            },
            "source": "excel",
            "file_name": req.file_name,
            "rows_in_file": profile["row_count"],
            "columns_in_file": [c["name"] for c in profile["columns"]],
            "file_profile": {
                "match_types": profile["match_types"],
                "competitions": profile["competitions"],
                "date_range": profile["date_range"],
                "unique_batters": profile["unique_batters"],
                "unique_bowlers": profile["unique_bowlers"],
            },
            "resolved_entities": resolved,
            "thresholds_relaxed": all_empty,
            "sql_queries": sql_queries,
            "results": results,
            "tables": tables,
            "chart_config": chart_cfg,
            "insight": insight,
            "session_turn": {"question": req.question, "summary": (insight[:400] if insight else "")},
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "traceback": traceback.format_exc()}


@app.post("/interpret")
def interpret_question(req: QueryRequest):
    try:
        intent = classify_intent(req.question)
        summary = generate_intent_summary(req.question, intent)
        prompt = f"""
You are a cricket analyst. A user has asked: "{req.question}"

Explain in 3-4 clear sentences:
1. What cricket analysis this question is asking for
2. Which players/teams/formats are involved
3. What data dimensions will be analysed (phase, form, comparison, etc.)
4. What the key insight will be

Be specific. Use cricket domain language only.
"""
        explanation = llm(prompt)
        return {
            "question": req.question,
            "intent_summary": summary,
            "intent_flags": {k: v for k, v in intent.items() if v is True},
            "explanation": explanation,
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "traceback": traceback.format_exc()}