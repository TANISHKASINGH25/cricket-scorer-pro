from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import os
import re
import json
import base64
import io
import hashlib
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

APP_VERSION = "3.0.0"

# =========================================================
# VERTEX AI SETUP  (UNCHANGED per user request)
# =========================================================
vertexai.init(
    project=os.getenv("GCP_PROJECT"),
    location=os.getenv("GCP_LOCATION", "us-central1")
)

model = GenerativeModel("gemini-2.5-flash")

# Separate generation configs for different task types
PLAN_CONFIG = GenerationConfig(temperature=0.1, top_p=0.9, max_output_tokens=4096)
INSIGHT_CONFIG = GenerationConfig(temperature=0.4, top_p=0.95, max_output_tokens=4096)
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
# DB CONNECTION POOL (much more efficient than per-request)
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
        # fallback to direct connection
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
# LLM WRAPPER (with light retry)
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
    """Call LLM and parse JSON robustly."""
    raw = llm(prompt, cfg=cfg or PLAN_CONFIG)
    raw = raw.replace("```json", "").replace("```", "").strip()
    m = re.search(r'\{[\s\S]*\}', raw)
    if m:
        raw = m.group(0)
    try:
        return json.loads(raw)
    except Exception:
        # try to repair common issues
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
# CONTEXT BUILDERS — CACHED (these were rebuilt on EVERY request before)
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

    lines.append("\nFORMAT BENCHMARK SELECTION:")
    lines.append("  T20/T20I -> T20 | ODI -> ODI | Test -> TEST | Mixed -> per-format separately")
    lines.append("  NEVER apply T20 benchmarks to ODI/Test data.")

    lines.append("\nDISMISSAL TYPES:")
    for dtype, meta in rules["dismissal_types"].items():
        lines.append(f"  {dtype}: {meta['tactical_note']}")

    lines.append("\nALL-ROUNDER THRESHOLDS (T20):")
    for k, v in rules["all_rounder_thresholds"]["T20"].items():
        lines.append(f"  {k}: {v}")

    return "\n".join(lines)


@lru_cache(maxsize=1)
def build_insight_rules_context() -> str:
    rules = CRICKET_RULES
    lines = ["PERFORMANCE TIER LABELS:"]
    for grp in ["batting", "bowling", "allround"]:
        lines.append(f"  {grp.upper()}:")
        for tier, label in rules["performance_labels"][grp].items():
            lines.append(f"    {tier}: {label}")

    for fmt in ["T20", "ODI", "TEST"]:
        if fmt in rules["batting_benchmarks"]:
            lines.append(f"\nBATTING BENCHMARKS [{fmt}]:")
            for metric, tiers in rules["batting_benchmarks"][fmt].items():
                lines.append(f"  {metric}: " + " | ".join(f"{t}={d}" for t, d in tiers.items()))
        if fmt in rules["bowling_benchmarks"]:
            lines.append(f"BOWLING BENCHMARKS [{fmt}]:")
            for metric, tiers in rules["bowling_benchmarks"][fmt].items():
                lines.append(f"  {metric}: " + " | ".join(f"{t}={d}" for t, d in tiers.items()))

    lines.append("\nINSIGHT RULES:")
    for r in rules["insight_rules"]:
        lines.append(f"  - {r}")
    return "\n".join(lines)


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

FORMULAS (PostgreSQL syntax, adapt for DuckDB by removing ::numeric casts):
  economy        = (SUM(runs+extra_runs)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0)) * 6
  strike rate    = (SUM(runs)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0)) * 100
  batting avg    = SUM(runs)::numeric / NULLIF(COUNT(*) FILTER (WHERE wicket IS NOT NULL),0)
  bowling avg    = SUM(runs+extra_runs)::numeric / NULLIF(COUNT(*) FILTER (WHERE wicket IS NOT NULL),0)
  bowling SR     = COUNT(*) FILTER (WHERE legal_ball=TRUE)::numeric / NULLIF(COUNT(*) FILTER (WHERE wicket IS NOT NULL),0)
  dot %          = (COUNT(*) FILTER (WHERE (runs+extra_runs)=0 AND legal_ball=TRUE)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0)) * 100
  boundary %     = (COUNT(*) FILTER (WHERE runs IN (4,6))::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0)) * 100
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
def build_timestamp_context() -> str:
    return """TIMESTAMP RULES:
  timestamp is already TIMESTAMP — NEVER call TO_TIMESTAMP(timestamp, ...)
  Duration: EXTRACT(EPOCH FROM (MAX(timestamp) - MIN(timestamp)))
  /60 for minutes, /3600 for hours
  Ball gaps: LAG(timestamp) OVER (PARTITION BY match, innings ORDER BY timestamp)
  Filter ball gaps BETWEEN 5 AND 300 seconds to exclude breaks.
"""


@lru_cache(maxsize=1)
def build_format_context() -> str:
    return """FORMAT RULES:
  T20/T20I -> T20 benchmarks | ODI/50 Over/List A -> ODI | Test/First Class/Timed -> TEST
  No format mentioned -> do NOT filter match_type
  T20: SR elite >160 | eco elite <6.5 | PP overs 1-6 | death overs 16-20
  ODI: SR elite >110 | eco elite <4.5 | PP overs 1-10 | death overs 41-50
  Test: avg elite >55 | eco elite <2.0
"""


@lru_cache(maxsize=1)
def build_systemic_cricket_knowledge() -> str:
    return """SYSTEMIC CRICKET KNOWLEDGE:
BATTING ARCHETYPES:
  AGGRESSOR: SR >145 T20 / >100 ODI, high boundary%, high 6:4 ratio
  ANCHOR: Moderate SR T20 120-135 / ODI 80-90 / Test avg 40+
  FINISHER: death SR T20 >175 / ODI >120
  ACCUMULATOR: High average, moderate SR, runs in singles/twos
BOWLING ARCHETYPES:
  WICKET-TAKER: SR <15 T20 / <30 ODI / <50 Test
  MISER: Economy <7 T20 / <5 ODI / <2.5 Test
  DEATH SPECIALIST: death eco <9.0 T20 / <7.0 ODI
CHASE: T20 RR >12 = crisis. ODI >9 last 10 = very hard.
FORM vs CLASS: Poor form + strong career = class is permanent, likely to return.
H2H: 3+ dismissals = psychological edge. T20 SR >180 vs bowler = batter dominance.
"""


@lru_cache(maxsize=1)
def build_advanced_patterns_context() -> str:
    # Kept compact; LLM only needs the shape, not full SQL
    return """ADVANCED SQL PATTERNS (PostgreSQL):

1. CAREER BASELINE:
  SELECT batter, COUNT(DISTINCT match) AS matches, SUM(runs) AS runs,
    COUNT(*) FILTER (WHERE legal_ball=TRUE) AS balls,
    ROUND((SUM(runs)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*100, 2) AS sr,
    ROUND(SUM(runs)::numeric / NULLIF(COUNT(*) FILTER (WHERE wicket IS NOT NULL),0), 2) AS avg,
    COUNT(*) FILTER (WHERE runs=4) AS fours, COUNT(*) FILTER (WHERE runs=6) AS sixes
  FROM public.nv_play WHERE batter ILIKE '%name%' GROUP BY batter

2. PHASE SPLIT — use one query with FILTER (WHERE over BETWEEN x AND y) per phase.

3. RECENT FORM — CTE: SELECT DISTINCT match,date ORDER BY date DESC LIMIT N, then JOIN.

4. YEARLY TREND — GROUP BY EXTRACT(YEAR FROM date) ORDER BY year ASC.

5. HEAD-TO-HEAD — WHERE batter ILIKE ... AND bowler ILIKE ..., aggregate balls/runs/dismissals.

6. DISMISSAL BREAKDOWN — GROUP BY wicket WHERE wicket IS NOT NULL with COUNT and pct.
"""


@lru_cache(maxsize=1)
def build_baseline_queries_hint() -> str:
    return """PROACTIVE BASELINE STRATEGY (MANDATORY for rich answers):
  Player question -> career_baseline + phase_split + yearly_trend + question_specific
  Team question -> team_batting + team_bowling + phase_split + question_specific
  Recent form -> career_baseline + last_10_matches + phase_split_recent
  H2H -> h2h_summary + batter_career + bowler_career + h2h_phase_split
  Leaderboard -> main_leaderboard + phase_leaders
Career baseline is the comparison anchor that makes every other number meaningful.
"""


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

    # DuckDB hates ::numeric / ::int
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
# QUERY PLAN GENERATION (PostgreSQL / DB route)
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
You are Cricket_Scorer_AI — an elite cricket data engineer.

TASK: Translate the user's question into a precise, production-grade multi-query
PostgreSQL plan. Include proactive baseline queries that contextualise the answer.

{threshold_note}
{feedback_note}
{intent_hint}
{session_ctx}

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

CRICKET TERMINOLOGY -> SQL:
{build_cricket_terms_context()}

DATE RULES:
{build_date_context()}

TIMESTAMP RULES:
{build_timestamp_context()}

FORMAT CONTEXT:
{build_format_context()}

ADVANCED PATTERNS:
{build_advanced_patterns_context()}

PROACTIVE BASELINE STRATEGY:
{build_baseline_queries_hint()}

HARD RULES:
  1. SELECT or WITH...SELECT only.
  2. Use the exact schema column names. NEVER invent columns.
  3. Every numeric metric: ROUND(value::numeric, 2).
  4. Every denominator: NULLIF(..., 0).
  5. wicket is TEXT: dismissed -> wicket IS NOT NULL.
  6. Booleans: legal_ball, free_hit, around_the_wicket, keeper_up.
  7. Names: ILIKE '%name%'.
  8. EXTRACT(YEAR FROM date)::int AS year — always.
  9. LIMIT 15-20 for open leaderboards. No LIMIT for named entities.
  10. Do NOT apply HAVING when a specific player/team is named.
  11. For complex questions, decompose into 3-5 focused queries instead of one giant query.

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
FORMAT RULES: {build_format_context()}

Return ONLY raw SQL — no markdown, no explanation.
SELECT only | PostgreSQL syntax | use schema columns only
wicket IS NOT NULL for dismissals | ROUND(x::numeric,2) | NULLIF(x,0) | ILIKE '%name%'

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
                # Single LLM-driven repair attempt
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

    # Retry 1: strip HAVING
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

    # Retry 2: regenerate with relax_thresholds=True
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

    # Normalise column names
    df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]

    # Coerce booleans
    for col in ["legal_ball", "free_hit", "around_the_wicket", "keeper_up"]:
        if col in df.columns:
            try:
                df[col] = df[col].astype(bool)
            except Exception:
                df[col] = df[col].map(lambda x: str(x).strip().lower() in ("true", "1", "yes", "t"))

    # Parse dates/timestamps
    for col in ["date", "timestamp"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Coerce numeric columns we expect to be numeric
    for col in ["runs", "extra_runs", "over", "ball", "innings", "batting_position"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def profile_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Lightweight profile to guide the LLM."""
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
        # add a small sample for low-cardinality string columns
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
    """Match user-mentioned names against actual values in the DataFrame (case-insensitive substring)."""
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
# EXCEL: QUERY PLAN (DuckDB) — schema-aware
# =========================================================
def generate_duckdb_query_plan(question: str,
                                profile: Dict[str, Any],
                                resolved_entities: Dict[str, List[str]],
                                intent: Optional[dict] = None,
                                error_feedback: Optional[str] = None) -> dict:

    columns = [c["name"] for c in profile["columns"]]
    available_cols = ", ".join(columns)

    # Compact profile for prompt
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

    # Per-column hints
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

    prompt = f"""
You are Cricket_Scorer_AI — an elite cricket data engineer.

TASK: Translate the user's question into a multi-query DuckDB SQL plan.
The data lives in DuckDB in-memory table:   nv_play
One row = one delivery (ball).

AVAILABLE COLUMNS:
{available_cols}

COLUMN PROFILE:
{chr(10).join(col_hints)}

FILE SUMMARY:
{chr(10).join(profile_summary_lines)}

{entity_hint}
{intent_hint}
{feedback}

CRICKET TERMINOLOGY:
{build_cricket_terms_context()}

CRICKET RULES:
{build_rules_context()}

FORMAT CONTEXT:
{build_format_context()}

PROACTIVE BASELINE STRATEGY:
{build_baseline_queries_hint()}

DUCKDB-SPECIFIC RULES (CRITICAL):
  1.  Table name is ALWAYS: nv_play  (no schema prefix)
  2.  FILTER syntax:        COUNT(*) FILTER (WHERE condition)
  3.  Booleans:             legal_ball = TRUE
  4.  Strings:              ILIKE '%name%'
  5.  Dismissals:           wicket IS NOT NULL (text column)
  6.  ROUND:                ROUND(value, 2)
  7.  NULLIF:               NULLIF(x, 0)
  8.  Cast:                 CAST(value AS DOUBLE)   -- NEVER ::numeric / ::int
  9.  EXTRACT:              EXTRACT(YEAR FROM date)
  10. CTEs and window functions are fully supported.
  11. DO NOT use: public.nv_play, ::numeric, ::int, DATEADD, YEAR(), MONTH()
  12. ONLY use columns from AVAILABLE COLUMNS above. If a needed column is missing
      (e.g. timestamp, batting_position), gracefully skip that dimension.
  13. For named entities, use the resolved values exactly as shown.

QUERY STRATEGY:
  - Always include a CAREER BASELINE query when a player is named.
  - Add PHASE SPLIT (PP/Middle/Death) when over column is available.
  - Add YEARLY TREND when date column is available and time dimension is detected.
  - Add the QUESTION-SPECIFIC query last.
  - For complex questions, produce 3-5 focused queries instead of one giant query.

OUTPUT — STRICT JSON ONLY, NO PREAMBLE, NO MARKDOWN:

{{
  "analysis_type": "single | multi",
  "intent": "1-2 sentences",
  "queries": [
    {{"name": "snake_case_name", "purpose": "what & why", "sql": "SELECT ... FROM nv_play ..."}}
  ]
}}

USER QUESTION: {question}
"""

    parsed = llm_json(prompt, cfg=PLAN_CONFIG)
    if not parsed or "queries" not in parsed:
        # Smart fallback based on what's in the file
        fb_sql = "SELECT * FROM nv_play LIMIT 20"
        if "batter" in columns and "runs" in columns:
            fb_sql = ("SELECT batter, COUNT(DISTINCT match) AS matches, SUM(runs) AS total_runs, "
                      "COUNT(*) FILTER (WHERE legal_ball=TRUE) AS balls "
                      "FROM nv_play GROUP BY batter ORDER BY total_runs DESC LIMIT 20")
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
                        if v != v:  # NaN
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
                            allow_repair=False,  # only one repair attempt
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
    # Choose the densest non-empty result for charting
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
    if intent.get("is_trend"):       hints.append("TIME: x_key='year', use 'line' for 1 metric, 'bar' for >=2")
    if intent.get("is_phase"):       hints.append("PHASE: x_key='phase' (PP/Middle/Death), use 'bar'")
    if intent.get("is_comparison"):  hints.append("COMPARE: 2-5 entities, use 'bar' or 'radar'")
    if intent.get("is_over_by_over"):hints.append("OVERS: x_key='over', use 'line'")
    if intent.get("is_consistency"): hints.append("DISTRIBUTION: x_key='score_band', use 'bar_colored'")
    if intent.get("is_milestone"):   hints.append("MILESTONE: x_key='year', use 'bar'")
    if intent.get("is_venue"):       hints.append("VENUE: x_key='venue', use 'bar_colored'")
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
# INSIGHT GENERATION
# =========================================================
def generate_cricket_insight(question: str,
                             query_results: list,
                             small_sample: bool = False,
                             intent: Optional[dict] = None,
                             session_context: Optional[list] = None,
                             file_context: Optional[dict] = None) -> str:

    compact_data = json.dumps(query_results, default=str)[:18000]
    session_ctx = build_session_context(session_context or [])

    file_block = ""
    if file_context:
        file_block = (
            "\nFILE CONTEXT (the data was supplied by the user as a spreadsheet):\n"
            f"  Rows: {file_context.get('row_count')}\n"
            f"  Match types: {file_context.get('match_types')}\n"
            f"  Date range: {file_context.get('date_range')}\n"
            f"  Competitions: {file_context.get('competitions')}\n"
            "Treat the file as the complete universe of data for this analysis. "
            "Do NOT reference external matches or players not present in the data.\n"
        )

    small_sample_note = ""
    if small_sample:
        small_sample_note = (
            "SAMPLE NOTE: Thresholds were relaxed. Mention sample-size caveats naturally "
            "but still classify against benchmarks.\n"
        )

    if intent is None:
        intent = classify_intent(question)

    dim_notes = []
    if intent.get("is_time_based"):  dim_notes.append("TIME/DURATION: use actual timestamp measurements, in min+sec.")
    if intent.get("is_trend"):       dim_notes.append("TIME-SERIES: chronological, best/worst year, trajectory.")
    if intent.get("is_phase"):       dim_notes.append("PHASE: each phase separately + tier label + strongest/weakest.")
    if intent.get("is_comparison"):  dim_notes.append("COMPARISON: side-by-side, clear winner per metric + overall.")
    if intent.get("is_form"):        dim_notes.append("FORM: career vs recent. Hot/cold/normal classification.")
    if intent.get("is_opponent"):    dim_notes.append("OPPONENT: best -> worst, bogey + favourite opponent.")
    if intent.get("is_consistency"): dim_notes.append("CONSISTENCY: distribution, duck rate, 30+ rate.")
    if intent.get("is_h2h"):         dim_notes.append("H2H: control of matchup, dismissal pattern, captain pick.")
    if intent.get("is_chase") or intent.get("is_pressure"):
        dim_notes.append("PRESSURE/CHASE: setting vs chasing split, clutch classification.")
    if intent.get("is_allrounder"):  dim_notes.append("ALL-ROUNDER: equal weight batting and bowling.")
    if intent.get("is_leaderboard"): dim_notes.append("LEADERBOARD: rank clearly, gaps between ranks.")
    if intent.get("is_fantasy"):     dim_notes.append("FANTASY: captain, vice-captain, differential.")
    if intent.get("is_predictive"):  dim_notes.append("PREDICTIVE: calibrated probabilities, risk factors.")
    if intent.get("is_milestone"):   dim_notes.append("MILESTONES: year-by-year counts, conversion rates.")
    if intent.get("is_over_by_over"):dim_notes.append("OVER-BY-OVER: link to format benchmarks per over.")
    if intent.get("is_venue"):       dim_notes.append("VENUE: rank by performance, home/away split.")

    dim_notes_str = "\n".join(f"  - {x}" for x in dim_notes) if dim_notes else "  - General analysis"

    prompt = f"""
You are Cricket_Scorer_AI — the world's most advanced cricket analytics engine.

Transform raw numbers into genuine insight. BLEND systemic cricket knowledge with the data.
NEVER fabricate statistics. NEVER mention SQL, database, queries, or columns.

{small_sample_note}
{file_block}
{session_ctx}

USER QUESTION: {question}

DATABASE RESULTS:
{compact_data}

CRICKET RULES, BENCHMARKS & INSIGHT GUIDELINES:
{build_insight_rules_context()}

FORMAT CONTEXT:
{build_format_context()}

SYSTEMIC CRICKET KNOWLEDGE:
{build_systemic_cricket_knowledge()}

DIMENSION-SPECIFIC INSTRUCTIONS:
{dim_notes_str}

REPORT STRUCTURE (follow EXACTLY):

---

## 🏏 [Compelling specific headline]

### 📊 Key Numbers
Markdown table with ALL data rows from the most relevant result.

### 🔍 Deep Analysis

**Overall Picture**
Detect format from match_type in the data. Apply correct benchmarks.
Tier labels: 🔴 Elite / 🟠 Excellent / 🟡 Good / 🟢 Average / ⚪ Below Par

**Strengths**
2-3 evidence-backed strengths with exact numbers.

**Weaknesses / Vulnerabilities**
2-3 evidence-backed weaknesses with exact numbers.

{"**Year-by-Year Trend**" if intent.get('is_trend') else ""}
{"**Phase Breakdown**" if intent.get('is_phase') else ""}
{"**Head-to-Head Breakdown**" if intent.get('is_h2h') else ""}
{"**Comparison**" if intent.get('is_comparison') else ""}
{"**Opponent Split**" if intent.get('is_opponent') else ""}
{"**Consistency Profile**" if intent.get('is_consistency') else ""}
{"**Chase vs Setting**" if intent.get('is_chase') or intent.get('is_pressure') else ""}
{"**Fantasy Recommendation**" if intent.get('is_fantasy') else ""}
{"**Predictive Outlook**" if intent.get('is_predictive') else ""}

**Tactical Intelligence**
Opposition captain plan. Coach recommendations. Phase-specific plans.

### 📈 Standout Moments / Records
4-6 bullets: **[Bold stat]** — explanation

### 💡 Verdict
5-7 sentences. Expert-panel quality. Tier classification. Forward-looking recommendation.

---

### ℹ️ Additional Context

**🎯 Query Context**
Restate the question. Explain analysis type. State format from match_type. State sample size and reliability.

**📅 Data Coverage**
Earliest and latest date. Competitions covered. Flag any gaps.

**📐 Benchmarks Used**
One line per metric: [Metric] — [Format]: [tier thresholds]

**📝 Summary**
3-5 sentences. Direct answer + one supporting fact + one forward-looking note.

**🔗 Related Analyses**
Exactly 2 specific follow-up questions tailored to the entities and format in the data.

---

HARD RULES:
1. All 5 Additional Context sections MUST appear.
2. NEVER end after Verdict.
3. NEVER default to T20 — derive format from match_type in the data.
4. Every tier label must cite the benchmark threshold.
5. Cricket terminology only — never mention SQL / database / columns.
6. 800-1500 words total.
"""

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
    """Full pipeline for user-uploaded Excel/CSV — schema-aware, entity-resolved, with auto-repair."""
    try:
        intent = classify_intent(req.question)
        intent_summary = generate_intent_summary(req.question, intent)

        # 1. Load + profile
        df = load_dataframe(req.file_base64, req.file_ext)
        if df.empty:
            return {
                "status": "error",
                "message": "Uploaded file is empty after parsing.",
                "file_name": req.file_name,
            }
        profile = profile_dataframe(df)

        # 2. Resolve entities from the actual data
        resolved = resolve_entities_in_df(df, intent.get("candidate_names", []))

        # 3. Generate DuckDB plan with full schema awareness
        plan = generate_duckdb_query_plan(
            req.question, profile=profile, resolved_entities=resolved, intent=intent
        )

        # 4. Execute (with per-query auto-repair)
        results = execute_duckdb_plan(
            plan, df, auto_repair=True, question=req.question,
            profile=profile, resolved_entities=resolved, intent=intent,
        )

        # 5. Retry path if everything empty
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
                # Regenerate a fresh broader plan
                fresh = generate_duckdb_query_plan(
                    req.question, profile=profile, resolved_entities=resolved, intent=intent,
                    error_feedback="All previous queries returned 0 rows. Broaden filters and remove HAVING.",
                )
                results = execute_duckdb_plan(
                    fresh, df, auto_repair=True, question=req.question,
                    profile=profile, resolved_entities=resolved, intent=intent,
                )
                all_empty = all(len(q.get("results", [])) == 0 for q in results)

        # 6. Chart + Insight
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
You are Cricket_Scorer_AI. A user has asked: "{req.question}"

Explain in 3-4 clear sentences:
1. What cricket analysis this question is asking for
2. Which players/teams/formats are involved
3. What data dimensions will be analysed (phase, form, comparison, etc.)
4. What the key insight will be

Be specific. Use cricket domain language. No SQL, no database language.
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