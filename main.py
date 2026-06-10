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
"""
def get_db_connection():
    return psycopg2.connect(
        host="/cloudsql/sportsanalytics-495612:europe-west2:sportsdb",
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        connect_timeout=10
    )
"""


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
    lines.append("  golden duck            -> runs_batter = 0 AND wicket IS NOT NULL AND ball_number = 1")

    lines.append("\nDELIVERY TYPES:")
    lines.append("  dot ball               -> runs_total = 0 AND legal_ball = TRUE")
    lines.append("  boundary               -> runs_batter IN (4, 6)")
    lines.append("  six / sixes            -> runs_batter = 6")
    lines.append("  four / fours           -> runs_batter = 4")
    lines.append("  free hit               -> free_hit = TRUE")
    lines.append("  wide                   -> legal_ball = FALSE (wide type)")
    lines.append("  no ball                -> legal_ball = FALSE (no-ball type)")
    lines.append("  scoring shot           -> runs_batter > 0 AND legal_ball = TRUE")

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
    lines.append("  six %                  -> ROUND((COUNT(*) FILTER (WHERE runs_batter=6)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*100, 2)")
    lines.append("  scoring rate           -> ROUND((COUNT(*) FILTER (WHERE runs_batter>0 AND legal_ball=TRUE)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*100, 2)")
    lines.append("  wickets per match      -> ROUND(COUNT(*) FILTER (WHERE wicket IS NOT NULL)::numeric / NULLIF(COUNT(DISTINCT match_id),0), 2)")
    lines.append("  balls per wicket       -> ROUND(COUNT(*) FILTER (WHERE legal_ball=TRUE)::numeric / NULLIF(COUNT(*) FILTER (WHERE wicket IS NOT NULL),0), 2)")

    lines.append("\nPARTNERSHIP PATTERNS:")
    lines.append("  partnership            -> GROUP BY match_id, innings_number, partnership_id (or over/ball window)")
    lines.append("  opening stand          -> batting_position IN (1,2) same innings/match")
    lines.append("  50+ partnership        -> filter partnerships SUM(runs_batter) >= 50")

    lines.append("\nCONDITIONAL AGGREGATIONS (most useful patterns):")
    lines.append("  runs in powerplay      -> SUM(runs_batter) FILTER (WHERE over_number BETWEEN 1 AND 6)")
    lines.append("  runs in death          -> SUM(runs_batter) FILTER (WHERE over_number BETWEEN 16 AND 20)")
    lines.append("  wickets in powerplay   -> COUNT(*) FILTER (WHERE wicket IS NOT NULL AND over_number BETWEEN 1 AND 6)")
    lines.append("  balls in powerplay     -> COUNT(*) FILTER (WHERE legal_ball=TRUE AND over_number BETWEEN 1 AND 6)")
    lines.append("  SR in powerplay        -> ROUND((SUM(runs_batter) FILTER (WHERE over_number BETWEEN 1 AND 6)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE AND over_number BETWEEN 1 AND 6),0))*100,2)")

    return "\n".join(lines)

# =========================================================
# DATE / TIME CONTEXT
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
# BUILD FORMAT CONTEXT  (NEW)
# Teaches the model about multi-format cricket
# =========================================================

def build_format_context():
    lines = []

    lines.append("CRICKET FORMAT RULES:")
    lines.append("")
    lines.append("FORMAT DETECTION from question:")
    lines.append("  T20 / T20I / IPL / Big Bash / PSL / BBL -> match_type ILIKE '%T20%'  OR match_type = 'T20'")
    lines.append("  ODI / one-day / 50-over                  -> match_type ILIKE '%ODI%'  OR match_type = 'ODI'")
    lines.append("  Test / Test match / red-ball              -> match_type ILIKE '%Test%' OR match_type = 'Test'")
    lines.append("  If no format specified -> do NOT filter on match_type (include all formats)")
    lines.append("")
    lines.append("OVER RANGES BY FORMAT:")
    lines.append("  T20  : total overs 1-20 | powerplay 1-6  | middle 7-15  | death 16-20")
    lines.append("  ODI  : total overs 1-50 | powerplay 1-10 | middle 11-40 | death 41-50")
    lines.append("  Test : no fixed over limit; use session or day context if available")
    lines.append("")
    lines.append("BENCHMARK DIFFERENCES BY FORMAT:")
    lines.append("  T20  batting SR elite: >150 | good: 130-150 | average: 110-130")
    lines.append("  ODI  batting SR elite: >100 | good:  85-100 | average:  70-85")
    lines.append("  T20  bowling econ  elite: <6.5 | good: 6.5-8 | expensive: >9")
    lines.append("  ODI  bowling econ  elite: <4.5 | good: 4.5-6 | expensive: >7")
    lines.append("")
    lines.append("TOURNAMENT / COMPETITION FILTERING:")
    lines.append("  Use competition ILIKE '%name%' or match_type for tournament-level filters")
    lines.append("  IPL       -> competition ILIKE '%IPL%' OR competition ILIKE '%Indian Premier%'")
    lines.append("  World Cup -> competition ILIKE '%World Cup%'")
    lines.append("  Champions Trophy -> competition ILIKE '%Champions Trophy%'")
    lines.append("")
    lines.append("VENUE / GROUND FILTERING:")
    lines.append("  home / away / neutral -> venue column if available")
    lines.append("  at <ground>           -> venue ILIKE '%ground_name%'")
    lines.append("")
    lines.append("INNINGS CONTEXT:")
    lines.append("  T20 powerplay batting (1-6)   -> team sets up innings momentum")
    lines.append("  T20 death batting (16-20)     -> finisher role, high SR expected")
    lines.append("  ODI death bowling (41-50)     -> yorker specialists, economy critical")
    lines.append("  Test new ball                 -> over_number BETWEEN 1 AND 10")

    return "\n".join(lines)

# =========================================================
# BUILD ADVANCED QUERY PATTERNS CONTEXT  (NEW)
# Covers complex query scenarios
# =========================================================

def build_advanced_patterns_context():
    lines = []

    lines.append("ADVANCED QUERY PATTERNS:")
    lines.append("")

    lines.append("1. CONSISTENCY / INNINGS PROFILE:")
    lines.append("""
  -- Distribution of scores (how often batter reaches milestones)
  SELECT
    CASE
      WHEN innings_runs < 10  THEN '0-9'
      WHEN innings_runs < 20  THEN '10-19'
      WHEN innings_runs < 30  THEN '20-29'
      WHEN innings_runs < 50  THEN '30-49'
      WHEN innings_runs < 75  THEN '50-74'
      ELSE '75+'
    END AS score_band,
    COUNT(*) AS innings_count
  FROM (
    SELECT match_id, innings_number, batter,
           SUM(runs_batter) AS innings_runs
    FROM public.nv_play
    WHERE batter ILIKE '%name%' AND legal_ball = TRUE
    GROUP BY match_id, innings_number, batter
  ) t
  GROUP BY score_band ORDER BY MIN(innings_runs)
""")

    lines.append("2. OPPONENT-SPECIFIC PERFORMANCE:")
    lines.append("""
  SELECT
    bowling_team                                                               AS opponent,
    COUNT(DISTINCT match_id)                                                   AS matches,
    SUM(runs_batter)                                                           AS runs,
    COUNT(*) FILTER (WHERE legal_ball=TRUE)                                    AS balls,
    ROUND((SUM(runs_batter)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*100,2) AS strike_rate,
    ROUND(SUM(runs_batter)::numeric / NULLIF(COUNT(*) FILTER (WHERE wicket IS NOT NULL),0),2)    AS average,
    COUNT(*) FILTER (WHERE wicket IS NOT NULL)                                 AS dismissals
  FROM public.nv_play
  WHERE batter ILIKE '%name%'
  GROUP BY bowling_team ORDER BY runs DESC
""")

    lines.append("3. BOWLER vs BATTING POSITION:")
    lines.append("""
  SELECT
    batting_position,
    COUNT(*) FILTER (WHERE legal_ball=TRUE)                                    AS balls,
    SUM(runs_total)                                                            AS runs_conceded,
    COUNT(*) FILTER (WHERE wicket IS NOT NULL)                                 AS wickets,
    ROUND((SUM(runs_total)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*6,2) AS economy,
    ROUND(SUM(runs_total)::numeric / NULLIF(COUNT(*) FILTER (WHERE wicket IS NOT NULL),0),2)  AS bowling_avg
  FROM public.nv_play
  WHERE bowler ILIKE '%name%' AND legal_ball = TRUE
  GROUP BY batting_position ORDER BY batting_position
""")

    lines.append("4. PHASE-BY-PHASE SPLIT (single query, multi-column):")
    lines.append("""
  SELECT
    batter,
    -- Powerplay
    SUM(runs_batter) FILTER (WHERE over_number BETWEEN 1 AND 6)               AS pp_runs,
    COUNT(*) FILTER (WHERE legal_ball=TRUE AND over_number BETWEEN 1 AND 6)   AS pp_balls,
    ROUND((SUM(runs_batter) FILTER (WHERE over_number BETWEEN 1 AND 6)::numeric
           / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE AND over_number BETWEEN 1 AND 6),0))*100,2) AS pp_sr,
    -- Middle
    SUM(runs_batter) FILTER (WHERE over_number BETWEEN 7 AND 15)              AS mid_runs,
    COUNT(*) FILTER (WHERE legal_ball=TRUE AND over_number BETWEEN 7 AND 15)  AS mid_balls,
    ROUND((SUM(runs_batter) FILTER (WHERE over_number BETWEEN 7 AND 15)::numeric
           / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE AND over_number BETWEEN 7 AND 15),0))*100,2) AS mid_sr,
    -- Death
    SUM(runs_batter) FILTER (WHERE over_number BETWEEN 16 AND 20)             AS death_runs,
    COUNT(*) FILTER (WHERE legal_ball=TRUE AND over_number BETWEEN 16 AND 20) AS death_balls,
    ROUND((SUM(runs_batter) FILTER (WHERE over_number BETWEEN 16 AND 20)::numeric
           / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE AND over_number BETWEEN 16 AND 20),0))*100,2) AS death_sr
  FROM public.nv_play
  WHERE batter ILIKE '%name%'
  GROUP BY batter
""")

    lines.append("5. RECENT FORM (last N matches):")
    lines.append("""
  WITH ranked_matches AS (
    SELECT match_id, match_date,
           ROW_NUMBER() OVER (ORDER BY match_date DESC) AS rn
    FROM public.nv_play
    WHERE batter ILIKE '%name%'
    GROUP BY match_id, match_date
  )
  SELECT
    p.match_date,
    p.match_id,
    SUM(p.runs_batter)                                                         AS runs,
    COUNT(*) FILTER (WHERE p.legal_ball=TRUE)                                  AS balls,
    ROUND((SUM(p.runs_batter)::numeric / NULLIF(COUNT(*) FILTER (WHERE p.legal_ball=TRUE),0))*100,2) AS strike_rate,
    MAX(p.wicket)                                                              AS dismissal
  FROM public.nv_play p
  JOIN ranked_matches r ON p.match_id = r.match_id
  WHERE p.batter ILIKE '%name%' AND r.rn <= 10
  GROUP BY p.match_date, p.match_id
  ORDER BY p.match_date DESC
""")

    lines.append("6. COMPARISON BETWEEN TWO PLAYERS (side by side):")
    lines.append("""
  SELECT
    batter                                                                     AS player,
    COUNT(DISTINCT match_id)                                                   AS matches,
    SUM(runs_batter)                                                           AS runs,
    COUNT(*) FILTER (WHERE legal_ball=TRUE)                                    AS balls,
    ROUND((SUM(runs_batter)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*100,2) AS strike_rate,
    ROUND(SUM(runs_batter)::numeric / NULLIF(COUNT(*) FILTER (WHERE wicket IS NOT NULL),0),2)    AS average,
    COUNT(*) FILTER (WHERE runs_batter=4)                                      AS fours,
    COUNT(*) FILTER (WHERE runs_batter=6)                                      AS sixes,
    ROUND((COUNT(*) FILTER (WHERE runs_total=0 AND legal_ball=TRUE)::numeric
           / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*100,2)        AS dot_pct
  FROM public.nv_play
  WHERE batter ILIKE '%player1%' OR batter ILIKE '%player2%'
  GROUP BY batter
""")

    lines.append("7. WICKET WICKET CONTRIBUTION (top wicket-takers for a team):")
    lines.append("""
  SELECT
    bowler,
    COUNT(DISTINCT match_id)                                                   AS matches,
    COUNT(*) FILTER (WHERE wicket IS NOT NULL)                                 AS wickets,
    COUNT(*) FILTER (WHERE legal_ball=TRUE)                                    AS balls,
    SUM(runs_total)                                                            AS runs_conceded,
    ROUND((SUM(runs_total)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*6,2) AS economy,
    ROUND(SUM(runs_total)::numeric / NULLIF(COUNT(*) FILTER (WHERE wicket IS NOT NULL),0),2)  AS bowling_avg,
    ROUND(COUNT(*) FILTER (WHERE legal_ball=TRUE)::numeric
          / NULLIF(COUNT(*) FILTER (WHERE wicket IS NOT NULL),0),2)           AS strike_rate
  FROM public.nv_play
  WHERE bowling_team ILIKE '%team%'
  GROUP BY bowler
  ORDER BY wickets DESC LIMIT 15
""")

    lines.append("8. PRESSURE / CHASE ANALYSIS:")
    lines.append("""
  -- Batter performance chasing vs setting
  SELECT
    CASE WHEN innings_number = 1 THEN 'Setting' ELSE 'Chasing' END            AS innings_type,
    COUNT(DISTINCT match_id)                                                   AS matches,
    SUM(runs_batter)                                                           AS runs,
    ROUND((SUM(runs_batter)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*100,2) AS strike_rate,
    ROUND(SUM(runs_batter)::numeric / NULLIF(COUNT(*) FILTER (WHERE wicket IS NOT NULL),0),2)    AS average
  FROM public.nv_play
  WHERE batter ILIKE '%name%'
  GROUP BY innings_number
""")

    lines.append("9. OVER-BY-OVER ECONOMY / RUN RATE:")
    lines.append("""
  SELECT
    over_number,
    ROUND(AVG(over_runs),2)    AS avg_runs_per_over,
    SUM(over_wickets)          AS total_wickets,
    COUNT(*)                   AS overs_bowled
  FROM (
    SELECT match_id, innings_number, over_number,
           SUM(runs_total)                         AS over_runs,
           COUNT(*) FILTER (WHERE wicket IS NOT NULL) AS over_wickets
    FROM public.nv_play
    GROUP BY match_id, innings_number, over_number
  ) t
  GROUP BY over_number ORDER BY over_number
""")

    lines.append("10. DISMISSAL BREAKDOWN BY BOWLER TYPE / PHASE:")
    lines.append("""
  SELECT
    wicket                                                                     AS dismissal_type,
    COUNT(*)                                                                   AS count,
    ROUND(COUNT(*)::numeric / NULLIF(SUM(COUNT(*)) OVER(),0)*100,2)           AS pct
  FROM public.nv_play
  WHERE batter ILIKE '%name%' AND wicket IS NOT NULL
  GROUP BY wicket ORDER BY count DESC
""")

    lines.append("11. HEAD-TO-HEAD DETAILED:")
    lines.append("""
  SELECT
    batter, bowler,
    COUNT(*) FILTER (WHERE legal_ball=TRUE)                                    AS balls,
    SUM(runs_batter)                                                           AS runs,
    COUNT(*) FILTER (WHERE wicket IS NOT NULL)                                 AS dismissals,
    ROUND((SUM(runs_batter)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*100,2) AS batter_sr,
    COUNT(*) FILTER (WHERE runs_total=0 AND legal_ball=TRUE)                   AS dots,
    COUNT(*) FILTER (WHERE runs_batter IN (4,6))                               AS boundaries
  FROM public.nv_play
  WHERE batter ILIKE '%x%' AND bowler ILIKE '%y%'
  GROUP BY batter, bowler
""")

    lines.append("12. TOP SCORERS / WICKET-TAKERS LEADERBOARD:")
    lines.append("""
  -- Batting leaderboard
  SELECT
    batter,
    COUNT(DISTINCT match_id)                                                   AS matches,
    SUM(runs_batter)                                                           AS runs,
    ROUND((SUM(runs_batter)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*100,2) AS strike_rate,
    ROUND(SUM(runs_batter)::numeric / NULLIF(COUNT(*) FILTER (WHERE wicket IS NOT NULL),0),2)    AS average,
    MAX(innings_high_score)                                                    AS high_score  -- if column exists
  FROM public.nv_play
  GROUP BY batter
  HAVING COUNT(*) FILTER (WHERE legal_ball=TRUE) >= 30
  ORDER BY runs DESC LIMIT 20
""")

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

        return {
            "status": "ok",
            "database": "connected"
        }

    except Exception as e:
        import traceback

        return {
            "status": "error",
            "error_type": str(type(e)),
            "message": str(e),
            "traceback": traceback.format_exc()
        }
    

@app.get("/debug")
def debug():
    return {
        "DB_HOST": os.getenv("DB_HOST"),
        "DB_NAME": os.getenv("DB_NAME"),
        "DB_USER": os.getenv("DB_USER")
    }

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

    # Fix MySQL-style YEAR() -> PostgreSQL EXTRACT
    sql_query = re.sub(
        r'\bYEAR\s*\(\s*(\w+)\s*\)',
        r'EXTRACT(YEAR FROM \1)',
        sql_query, flags=re.IGNORECASE
    )
    # Fix MONTH()
    sql_query = re.sub(
        r'\bMONTH\s*\(\s*(\w+)\s*\)',
        r'EXTRACT(MONTH FROM \1)',
        sql_query, flags=re.IGNORECASE
    )
    # Fix DAY()
    sql_query = re.sub(
        r'\bDAY\s*\(\s*(\w+)\s*\)',
        r'EXTRACT(DAY FROM \1)',
        sql_query, flags=re.IGNORECASE
    )
    # Fix IFNULL -> COALESCE
    sql_query = re.sub(
        r'\bIFNULL\s*\(',
        'COALESCE(',
        sql_query, flags=re.IGNORECASE
    )
    # Fix NVL -> COALESCE
    sql_query = re.sub(
        r'\bNVL\s*\(',
        'COALESCE(',
        sql_query, flags=re.IGNORECASE
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
# GENERATE QUERY PLAN  (ENHANCED)
# =========================================================

def generate_query_plan(question, relax_thresholds=False):

    schema_context   = build_schema_context(question)
    metrics_context  = build_metrics_context()
    rules_context    = build_rules_context()
    cricket_terms    = build_cricket_terms_context()
    date_context     = build_date_context()
    format_context   = build_format_context()
    advanced_patterns = build_advanced_patterns_context()

    threshold_note = ""
    if relax_thresholds:
        threshold_note = """
⚠️  RETRY MODE — EMPTY RESULTS PREVIOUSLY:
- DO NOT add any HAVING clauses or minimum ball/innings thresholds.
- Return ALL rows regardless of sample size.
- The insight layer will add small-sample caveats automatically.
- Widen name matching if needed: use ILIKE '%partial_name%'
"""

    prompt = f"""
You are Cricket_Scorer_AI — an elite cricket data engineer and analyst with deep
knowledge of T20, ODI, Test, and domestic cricket formats globally.

Your ONLY job: translate the user's natural language cricket question into a
precise, correct, multi-query PostgreSQL plan that returns ALL data needed
to answer it fully, including complex statistics, time trends, opponent splits,
phase analysis, and comparisons.

{threshold_note}

══════════════════════════════════════════════════════════
DATABASE
══════════════════════════════════════════════════════════
Table  : public.nv_play
Grain  : ONE ROW = ONE DELIVERY (ball) in a cricket match
Engine : PostgreSQL — strict syntax required

══════════════════════════════════════════════════════════
SCHEMA (columns available)
══════════════════════════════════════════════════════════
{schema_context}

══════════════════════════════════════════════════════════
DERIVED METRICS
══════════════════════════════════════════════════════════
{metrics_context}

══════════════════════════════════════════════════════════
CRICKET RULES & BENCHMARKS
══════════════════════════════════════════════════════════
{rules_context}

══════════════════════════════════════════════════════════
CRICKET TERMINOLOGY → SQL
══════════════════════════════════════════════════════════
{cricket_terms}

══════════════════════════════════════════════════════════
DATE & TIME RULES
══════════════════════════════════════════════════════════
{date_context}

══════════════════════════════════════════════════════════
FORMAT & COMPETITION CONTEXT
══════════════════════════════════════════════════════════
{format_context}

══════════════════════════════════════════════════════════
ADVANCED QUERY PATTERNS (use as templates)
══════════════════════════════════════════════════════════
{advanced_patterns}

══════════════════════════════════════════════════════════
CRITICAL COLUMN RULES — NEVER VIOLATE
══════════════════════════════════════════════════════════

WICKETS (TEXT column):
  Values: 'Caught', 'Bowled', 'LBW', 'Run Out', 'Stumped', 'Hit Wicket'
  ✅ dismissed        -> wicket IS NOT NULL
  ✅ specific type    -> wicket = 'Caught'
  ✅ not out          -> wicket IS NULL
  ❌ NEVER: wicket = TRUE / wicket = FALSE / wicket = 1

BOOLEANS (use TRUE/FALSE, not 1/0):
  legal_ball, free_hit, around_the_wicket, keeper_up

RUNS (always aggregate — one row = one ball):
  ✅ Strike rate  = (SUM(runs_batter) / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0)) * 100
  ✅ Economy rate = (SUM(runs_total)  / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0)) * 6
  ❌ NEVER treat a single row as an innings total

NAMES: ALWAYS use ILIKE '%name%' — player names may have spaces, initials, variants
SCHEMA ONLY: NEVER invent column names not in the schema above
POSTGRESQL ONLY: EXTRACT(), COALESCE(), FILTER (WHERE ...) — not MySQL/MSSQL syntax

══════════════════════════════════════════════════════════
QUERY DESIGN RULES
══════════════════════════════════════════════════════════

1.  SELECT or WITH...SELECT only. Pure PostgreSQL.
2.  Every computed column MUST have a meaningful alias.
3.  ROUND(value::numeric, 2) on ALL decimal results.
4.  NULLIF(..., 0) on EVERY denominator — no division-by-zero exceptions.
5.  LIMIT 15-20 for open leaderboards. No LIMIT for specific entity queries.
6.  ORDER BY most informative column. Time queries → ORDER BY year ASC.
7.  COUNT(DISTINCT match_id) AS matches_played for match counts.
8.  Use conditional aggregation FILTER (WHERE ...) to split phases in one query.
9.  CTEs (WITH clauses) for complex multi-step logic (recent form, percentages).
10. When comparing two players/teams, use OR in WHERE and GROUP BY entity.

THRESHOLD RULE (HAVING):
  ✅ APPLY only for open leaderboards (no specific entity named in question):
      Batting leaderboard: HAVING COUNT(*) FILTER (WHERE legal_ball=TRUE) >= 30
      Bowling leaderboard: HAVING COUNT(*) FILTER (WHERE legal_ball=TRUE) >= 24
  ❌ NEVER apply HAVING when:
      - A specific player name is mentioned
      - A specific team name is mentioned
      - A specific year / date / season is mentioned
      - A batting position term is used (top order, openers, etc.)
      - Question is about a named entity's full career / history

MULTI-QUERY STRATEGY — use 2-4 queries for complex questions:
  Q1: Career / overall / full-period summary (always include)
  Q2: Year-by-year breakdown OR phase split OR opponent split
  Q3: Comparison, ranking, or head-to-head breakdown
  Q4 (optional): Dismissal types, special situations, recent form

══════════════════════════════════════════════════════════
ANALYSIS TYPE DETECTION — map question → query strategy
══════════════════════════════════════════════════════════

YEARLY / TREND           -> GROUP BY EXTRACT(YEAR FROM match_date), ORDER BY year ASC
CAREER BATTING           -> career totals + phase split + dismissal breakdown
CAREER BOWLING           -> career totals + phase split + dismissal methods + opponent split
TEAM BATTING             -> team totals + phase split + top run-scorers
TEAM BOWLING             -> team economy + wicket-takers + phase split
HEAD TO HEAD             -> batter vs bowler matchup (h2h query + over-by-over optional)
PARTNERSHIP              -> partnership runs, balls, SR — group by partnership window
MATCH SUMMARY            -> innings totals, RR, wickets, top performers per team
LEADERBOARD              -> ranked list + HAVING thresholds (batting or bowling)
PHASE ANALYSIS           -> powerplay / middle / death split — conditional aggregation
DISMISSAL ANALYSIS       -> wicket type breakdown + % share + by bowling type
BATTING POSITION         -> split by batting_position or position range
OVER ANALYSIS            -> over-by-over run rates, wickets, economy
DATE RANGE               -> filter match_date with BETWEEN or EXTRACT
RECENT FORM              -> CTE ranked by match_date DESC, last N matches
OPPONENT ANALYSIS        -> GROUP BY bowling_team (for batter) or batting_team (for bowler)
CONSISTENCY              -> innings score distribution bands
INNINGS COMPARISON       -> first vs second innings stats
FORMAT COMPARISON        -> split by match_type (T20 vs ODI)
PRESSURE / CHASE         -> innings_number = 1 vs 2 split
ALL-ROUNDER PROFILE      -> batting + bowling stats in same response
MILESTONE TRACKING       -> 50s, 100s, 5-wicket hauls — sub-query or CASE counting

══════════════════════════════════════════════════════════
COMPLEX QUESTION EXAMPLES (for reference)
══════════════════════════════════════════════════════════

"How has [player] performed against pace vs spin?"
→ Q1: vs pace bowlers (bowl_type ILIKE '%pace%' or pace indicator)
→ Q2: vs spin bowlers
→ Q3: Phase split within each

"Which bowlers does [batter] struggle against?"
→ Q1: h2h vs all bowlers faced, ORDER BY batter SR ASC (lowest = struggle)
→ Q2: Dismissal methods breakdown

"How does [team] perform in run-chases vs setting totals?"
→ Q1: innings_number=2 (chasing) stats
→ Q2: innings_number=1 (setting) stats
→ Q3: Year-by-year chase success rate if possible

"Who are the most consistent batters in the powerplay?"
→ Q1: Powerplay batting leaderboard (SR + dot% + boundary%)
→ Q2: Phase comparison (how PP SR compares to their overall SR)

"Compare [player1] and [player2] over the last 3 years"
→ Q1: Side-by-side career comparison (last 3 years)
→ Q2: Year-by-year for both players
→ Q3: Phase split for both

══════════════════════════════════════════════════════════
OUTPUT FORMAT — STRICT
══════════════════════════════════════════════════════════

Return ONLY valid JSON. No markdown. No explanation. No preamble.
The JSON must be parseable directly with json.loads().

{{
  "analysis_type": "single | multi",
  "intent": "precise 1-2 sentence description of what the user is asking, including time/format/phase dimensions",
  "has_time_dimension": true | false,
  "has_phase_dimension": true | false,
  "has_comparison": true | false,
  "threshold_applied": true | false,
  "queries": [
    {{
      "name": "descriptive_snake_case_name",
      "purpose": "exactly what this query computes and why it answers the question",
      "sql": "SELECT ..."
    }}
  ]
}}

USER QUESTION:
{question}
"""

    raw = llm(prompt).replace("```json", "").replace("```", "").strip()

    # Robust JSON extraction — handle LLM preamble
    json_match = re.search(r'\{[\s\S]*\}', raw)
    if json_match:
        raw = json_match.group(0)

    try:
        return json.loads(raw)
    except Exception:
        return {
            "analysis_type": "single",
            "has_time_dimension": False,
            "has_phase_dimension": False,
            "has_comparison": False,
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
# FALLBACK SQL  (ENHANCED)
# =========================================================

def generate_fallback_sql(question):

    schema_context  = build_schema_context(question)
    cricket_terms   = build_cricket_terms_context()
    date_context    = build_date_context()
    format_context  = build_format_context()

    prompt = f"""
You are a PostgreSQL expert for cricket ball-by-ball data.

TABLE: public.nv_play (each row = one delivery)

SCHEMA:
{schema_context}

CRICKET TERMINOLOGY:
{cricket_terms}

DATE & TIME RULES:
{date_context}

FORMAT RULES:
{format_context}

STRICT RULES:
1.  Return ONLY the raw SQL — no explanation, no markdown, no backticks, no preamble.
2.  SELECT or WITH...SELECT only. Pure PostgreSQL syntax.
3.  Only schema columns — NEVER invent column names.
4.  wicket is TEXT — use wicket IS NOT NULL (not wicket = TRUE).
5.  legal_ball, free_hit, around_the_wicket, keeper_up are BOOLEAN — use TRUE/FALSE.
6.  ROUND(value::numeric, 2) for all decimal columns.
7.  NULLIF(..., 0) on ALL denominators.
8.  Meaningful alias on every computed column.
9.  LIMIT 20 for open leaderboards.
10. ILIKE '%name%' for all name matching.
11. NO HAVING clauses — return all available data.
12. DATE QUERIES: EXTRACT(YEAR FROM match_date) — NEVER use YEAR() function.
13. Yearly grouping: GROUP BY EXTRACT(YEAR FROM match_date) ORDER BY year ASC.
14. Always alias EXTRACT(YEAR FROM match_date) AS year.
15. Strike rate = (SUM(runs_batter) / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0)) * 100
16. Economy    = (SUM(runs_total)  / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0)) * 6
17. NEVER use IFNULL, NVL, YEAR(), MONTH(), DATEPART() — these are not PostgreSQL.

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

        # Stage 2 — full regeneration without thresholds
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
    table      = [header_row, separator]

    for row in results:
        table.append(" | ".join(str(v) for v in row.values()))

    return "\n".join(table)

# =========================================================
# GENERATE CRICKET INSIGHT  (ENHANCED)
# =========================================================

def generate_cricket_insight(question, query_results, small_sample=False):

    compact_data          = json.dumps(query_results, default=str)[:14000]
    insight_rules_context = build_insight_rules_context()
    format_context        = build_format_context()

    small_sample_note = ""
    if small_sample:
        small_sample_note = """
⚠️  SMALL SAMPLE SIZE NOTE:
Thresholds were relaxed to retrieve this data. Mention naturally within the analysis
(e.g. "across limited appearances", "based on a small sample") — do NOT refuse to analyse.
Still apply tier benchmarks and give a full verdict.
"""

    # Detect question dimensions for tailored insight instructions
    time_keywords    = ["year", "season", "annual", "trend", "every", "each year",
                        "monthly", "over time", "progression", "2021", "2022",
                        "2023", "2024", "2025", "recent", "latest", "history"]
    phase_keywords   = ["powerplay", "pp", "middle over", "death", "slog", "phase"]
    compare_keywords = ["vs", "versus", "compare", "comparison", "between", "better",
                        "worse", "who is", "which team", "both"]
    form_keywords    = ["recent", "last", "form", "current", "now", "this year",
                        "last 5", "last 10"]
    opponent_keywords = ["against", "vs", "versus", "opponent", "facing", "bowled by"]

    is_time_query     = any(kw in question.lower() for kw in time_keywords)
    is_phase_query    = any(kw in question.lower() for kw in phase_keywords)
    is_comparison     = any(kw in question.lower() for kw in compare_keywords)
    is_form_query     = any(kw in question.lower() for kw in form_keywords)
    is_opponent_query = any(kw in question.lower() for kw in opponent_keywords)

    dimension_notes = ""

    if is_time_query:
        dimension_notes += """
TIME-SERIES INSTRUCTIONS:
- Structure analysis chronologically: earliest year → latest year
- Identify TREND clearly: improving / declining / inconsistent / peaking
- Call out the BEST year and WORST year with exact stats
- Describe year-on-year delta: "SR jumped from 124 in 2022 to 141 in 2023 (+17)"
- Note any inflection points and what may explain them (injuries, team changes, format switch)
- In Key Numbers table: Year | Matches | primary metric columns
- In Verdict: is the trajectory upward, downward, or plateauing?
"""

    if is_phase_query:
        dimension_notes += """
PHASE ANALYSIS INSTRUCTIONS:
- Dedicate a section to each phase (Powerplay / Middle / Death)
- Classify each phase performance against T20 benchmarks
- Identify the player/team's strongest and weakest phase explicitly
- Explain TACTICAL implications: "The high powerplay SR suggests intent to dominate with new ball"
- Link phase weaknesses to dismissal patterns if data supports it
"""

    if is_comparison:
        dimension_notes += """
COMPARISON INSTRUCTIONS:
- Present both entities side-by-side — never analyse them in isolation
- Use a direct comparison table as the first table in Key Numbers
- Determine a clear winner on each metric — state it explicitly
- Give an overall verdict: "Player A is the superior T20 batter based on..."
- Note where each entity has an edge and why it matters tactically
"""

    if is_form_query:
        dimension_notes += """
RECENT FORM INSTRUCTIONS:
- Distinguish clearly between career stats and recent form
- Is the recent form better or worse than career average? By how much?
- Identify if there's a hot streak or a lean patch with specific match data
- In Verdict: comment on current momentum heading into future matches
"""

    if is_opponent_query:
        dimension_notes += """
OPPONENT ANALYSIS INSTRUCTIONS:
- Rank opponents from best-performance to worst-performance
- Identify the "bogey team/bowler" — where the entity clearly struggles
- Identify the "favourite" opponent — where they dominate
- Note if the sample is large enough to draw conclusions per opponent
- Tactical takeaway: what does the opponent split suggest about the entity's game?
"""

    prompt = f"""
You are Cricket_Scorer_AI — a world-class cricket analyst combining the tactical depth
of a coaching analyst, the storytelling of a top commentator, and the precision of a
data scientist. You have deep knowledge of T20, ODI, Test and domestic cricket.

{small_sample_note}

USER QUESTION:
{question}

DATA FROM DATABASE:
{compact_data}

CRICKET RULES, BENCHMARKS & INSIGHT GUIDELINES:
{insight_rules_context}

FORMAT & COMPETITION CONTEXT:
{format_context}

DIMENSION-SPECIFIC INSTRUCTIONS:
{dimension_notes}

══════════════════════════════════════════════════════════
YOUR TASK
══════════════════════════════════════════════════════════

Produce a rich, structured cricket analysis report that FULLY answers the question.
Use ONLY data from the database results above — never fabricate statistics.
Every claim must be backed by a specific number from the data.

Follow this layout exactly:

---

## 🏏 [Compelling, specific headline — include entity name and time/context if relevant]

### 📊 Key Numbers
Markdown table with actual numbers from the data.
Column headers depend on analysis type:
- Time queries  : Year | Matches | Runs/Wickets | SR/Economy | Average
- Player queries: Metric | Value (or side-by-side for comparisons)
- Team queries  : Category | Matches | Runs | Run Rate | Wickets
- Phase queries : Phase | Balls | Runs | SR/Economy | Wickets | Dot%
Include ALL meaningful columns — do not truncate data.

### 🔍 Deep Analysis

**Overall Performance**
1-2 paragraph summary with exact figures. Apply tier labels (Elite / Good / Average / Below Par).
State clearly what the numbers mean in cricket context.

**Strengths**
Specific stats backing every claim. Use cricket language (e.g. "exceptional powerplay aggression",
"death-over yorker accuracy"). Reference exact numbers.

**Weaknesses / Vulnerabilities**
Reference exact numbers. What is the entity's Achilles heel? What would opposition target?

{'''**Year-by-Year Trend**
Describe progression year by year. Identify peaks, troughs, inflection points.
Use format: "In **YEAR**, [entity] scored X runs at SR of Y — their [best/worst] year."''' if is_time_query else ''}

{'''**Phase Breakdown**
Analyse each phase separately. Classify each phase. State tactical implications.''' if is_phase_query else ''}

{'''**Head-to-Head Comparison**
Direct comparison on every key metric. State who leads on what. Overall winner.''' if is_comparison else ''}

{'''**Opponent Split**
Best and worst opponents. Tactical pattern from the data.''' if is_opponent_query else ''}

**Tactical Insights**
Phase-wise patterns, bowling angle, keeper position, scoring zones — only from available data.
What should an opposition captain know? What should the entity's coach focus on?

### 📈 Standout Moments / Records
3-5 bullet points highlighting the most impressive or concerning data points.
{'''Include best-year and worst-year statistics.''' if is_time_query else ''}
Use format: **Bold the key fact**, then explain significance.

### 💡 Verdict
4-6 sentences. Expert-panel quality. Classify overall performance tier.
{'''State trajectory: is the entity on an upward or downward arc?''' if is_time_query else ''}
{'''State who is better and why.''' if is_comparison else ''}
What would you tell the team management or opposition strategist?

---

STRICT RULES:
- Always cite exact numbers: "SR of 187.4", "economy of 6.23", "dismissed 7 times by Caught"
- Use cricket terminology naturally: wagon wheel, hard length, wide yorker, knuckleball, etc.
- Markdown tables with | pipes and --- separators
- Bold key entities: **Rohit Sharma**, **2023**, **Powerplay**
- Never mention SQL, database, tables, queries, or data structures
- Never fabricate stats not in the data
- No filler phrases: "it is worth noting", "interestingly", "it can be seen that"
- Be direct and specific — this is professional cricket analysis
- Word count: 600-1000 words, more if data richness demands it
- Empty or missing data: acknowledge gracefully and analyse what IS available
"""

    return llm(prompt)

# =========================================================
# GENERATE CHART CONFIG  (ENHANCED)
# =========================================================

def generate_chart_config(question, query_results):

    compact_data = json.dumps(query_results, default=str)[:10000]

    # Detect dimensions
    time_keywords    = ["year", "season", "annual", "trend", "every", "each year",
                        "monthly", "over time", "progression", "history"]
    phase_keywords   = ["powerplay", "pp", "middle", "death", "slog", "phase"]
    compare_keywords = ["vs", "versus", "compare", "comparison", "both"]

    is_time_query  = any(kw in question.lower() for kw in time_keywords)
    is_phase_query = any(kw in question.lower() for kw in phase_keywords)
    is_comparison  = any(kw in question.lower() for kw in compare_keywords)

    time_chart_hint = ""
    if is_time_query:
        time_chart_hint = """
TIME QUERY DETECTED:
- Data has a 'year' column with multiple years → prefer 'line' or 'bar'
- line : single metric trend over years (e.g. SR by year)
- area : cumulative/volume metric over years (e.g. total runs by year)
- bar  : multiple metrics per year side-by-side
- x_key = 'year', y_keys = numeric performance columns
"""

    phase_chart_hint = ""
    if is_phase_query:
        phase_chart_hint = """
PHASE QUERY DETECTED:
- Data has phase categories (Powerplay / Middle / Death) → prefer 'bar' or 'radar'
- bar   : compare 2-3 metrics across phases
- radar : multi-metric profile across phases (if 3+ metrics for same entity)
- x_key = 'phase', y_keys = numeric metric columns
"""

    compare_chart_hint = ""
    if is_comparison:
        compare_chart_hint = """
COMPARISON QUERY DETECTED:
- Multiple entities being compared → prefer 'bar' (grouped) or 'radar'
- bar   : 2+ metrics across 2-5 players/teams
- radar : multi-metric profile overlay for 2-4 players (4-6 metrics)
- x_key = player/team name column, y_keys = metrics
"""

    prompt = f"""
You are a cricket data visualization expert. Choose the chart type that best communicates
the primary insight. If no chart adds meaningful value, return null.

{time_chart_hint}
{phase_chart_hint}
{compare_chart_hint}

CHART TYPE GUIDE:

bar
  WHEN: Multi-metric comparison across players/teams/phases/years (2+ y_keys)
  REQUIRES: Multiple y_keys, categorical or year x_axis
  MAX ROWS: 12

bar_colored
  WHEN: Single-metric leaderboard — each bar a unique entity/color
  REQUIRES: Exactly 1 y_key
  MAX ROWS: 15

line
  WHEN: Trends over time, year-by-year progression, over-by-over run rates
  REQUIRES: Ordered x_axis (year, over_number)
  MAX ROWS: 25

area
  WHEN: Cumulative or volume trends over time (total runs, innings count)
  REQUIRES: Ordered x_axis
  MAX ROWS: 25

pie
  WHEN: Distribution or share of a whole (dismissal type %, boundary %)
  REQUIRES: Exactly 1 y_key, 3-8 categories
  MAX ROWS: 8

radar
  WHEN: Multi-dimension profile — compare 2-4 players across 4-6 metrics
  REQUIRES: 2-4 rows (players/teams), 4-6 y_keys
  MAX ROWS: 4

SELECTION PRIORITY:
1. Year/time data with 1 metric trend               → line
2. Year/time data with 2+ metrics                   → bar (grouped by year)
3. Year/time cumulative volume                      → area
4. Distribution / breakdown / share of whole        → pie
5. Compare top-N entities on ONE metric             → bar_colored
6. Compare top-N entities on MULTIPLE metrics       → bar (grouped)
7. Multi-dimension player profile (4-6 metrics)     → radar
8. Phase breakdown (Powerplay/Middle/Death)         → bar
9. Only 1 data row                                  → null
10. No numeric columns                              → null
11. Single stat lookup                              → null

DATA PREPARATION RULES:
1. x_key must map to a STRING, category, or integer year column
2. y_keys must be NUMERIC only — exclude counts if they'd dwarf other metrics
3. All values must be actual numbers, not strings
4. Floats already rounded to 2 decimal places
5. For year data: x_key = 'year', values are integers (2021, 2022...)
6. For phase data: x_key = 'phase', values are strings ('Powerplay', 'Middle', 'Death')
7. Limit data array to the first/most relevant query result when multiple queries exist
8. title: short and specific (include entity name)
9. subtitle: one-line context (optional)

RETURN FORMAT (strict JSON):
{{
  "chart_type": "bar | bar_colored | line | area | pie | radar",
  "title": "Short descriptive title including entity name",
  "subtitle": "Optional one-line context string",
  "x_key": "column_name_for_x_axis",
  "y_keys": ["metric1", "metric2"],
  "data": [
    {{ "year": 2021, "runs": 450, "strike_rate": 124.5 }},
    {{ "year": 2022, "runs": 612, "strike_rate": 138.2 }}
  ]
}}

If no chart adds value return exactly: null
No markdown. No explanation. No preamble. Only valid JSON or the word null.

USER QUESTION:
{question}

DATABASE RESULTS:
{compact_data}
"""

    raw = llm(prompt).replace("```json", "").replace("```", "").strip()

    if raw.lower() == "null":
        return None

    # Robust JSON extraction
    json_match = re.search(r'\{[\s\S]*\}', raw)
    if json_match:
        raw = json_match.group(0)

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
            "question":            req.question,
            "analysis_type":       query_plan.get("analysis_type", "single"),
            "has_time_dimension":  query_plan.get("has_time_dimension", False),
            "has_phase_dimension": query_plan.get("has_phase_dimension", False),
            "has_comparison":      query_plan.get("has_comparison", False),
            "thresholds_relaxed":  thresholds_relaxed,
            "sql_queries":         sql_queries,
            "results":             query_results,
            "tables":              tables,
            "chart_config":        chart_config,
            "insight":             insight
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}