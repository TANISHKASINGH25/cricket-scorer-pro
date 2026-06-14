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
# APP VERSION
# Bump this string on every deployment so you can confirm
# the latest build is live via GET /version
# =========================================================

APP_VERSION = "1.0.0"

# =========================================================
# VERTEX AI SETUP  (unchanged — do not modify)
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
    session_context: list = []   # NEW: carries prior Q&A pairs for conversational memory


# =========================================================
# VERTEX LLM CALL  (unchanged — do not modify)
# =========================================================

def llm(prompt: str) -> str:
    response = model.generate_content(prompt)
    return response.text.strip()

# =========================================================
# INTENT CLASSIFICATION
# Classifies the question across 18 orthogonal dimensions
# before any SQL is generated so every downstream stage
# can adapt its behaviour precisely.
# =========================================================

def classify_intent(question: str) -> dict:
    q = question.lower()

    # ── Primary subject ──────────────────────────────────
    is_batting    = any(w in q for w in ["bat","runs","strike rate","average","century","fifty","boundary","four","six","scorer","openers","batting","innings","batter"])
    is_bowling    = any(w in q for w in ["bowl","wicket","economy","dot ball","yorker","delivery","spin","pace","seam","swing","bowler","bowling"])
    is_fielding   = any(w in q for w in ["catch","field","run out","direct hit","dropped","boundary save","fielder"])
    is_team       = any(w in q for w in ["team","squad","side","xi","playing eleven","franchise"])
    is_h2h        = any(w in q for w in ["head to head","h2h","matchup","batter vs bowler","vs bowler","vs batter"])
    is_allrounder = any(w in q for w in ["all-rounder","allrounder","all rounder","both bat and bowl"])

    # ── Analytical dimensions ─────────────────────────────
    is_form        = any(w in q for w in ["recent","last ","form","current form","in form","last 5","last 10","last 10 matches","this year","this season","current"])
    is_trend       = any(w in q for w in ["year","season","annual","trend","every year","each year","monthly","over time","progression","history","since","before","2021","2022","2023","2024","2025"])
    is_phase       = any(w in q for w in ["powerplay","pp ","middle over","death over","slog","phase","over 1","over 6","over 16","over 20","first 6","last 5 overs","overs 1","overs 16"])
    is_comparison  = any(w in q for w in [" vs "," versus ","compare","comparison","between","better","worse","who is better","which team","both","two players","two teams"])
    is_opponent    = any(w in q for w in ["against ","opponent","facing","when bowling to","when batting against","matchup","bogey","favourite opponent"])
    is_consistency = any(w in q for w in ["consistent","consistency","reliable","duck","fifty","century","milestone","how often","score distribution"])
    is_chase       = any(w in q for w in ["chase","chasing","run chase","target","second innings","batting second","defending","first innings","batting first","setting"])
    is_pressure    = any(w in q for w in ["pressure","clutch","crunch","crucial","eliminate","final","knockout","must win"])
    is_leaderboard = any(w in q for w in ["top","best","highest","most","ranking","rank","leaderboard","who has most","who scored most","who took most"])
    is_milestone   = any(w in q for w in ["century","centuries","hundred","50s","fifties","duck","golden duck","hat-trick","five-for","5 wickets","ten wickets","milestone"])
    is_over_by_over= any(w in q for w in ["over by over","each over","per over","over number","which over","best over"])
    is_venue       = any(w in q for w in ["venue","ground","stadium","pitch","home","away"])
    is_fantasy     = any(w in q for w in ["fantasy","dream11","dream 11","pick","differential","captain","vice captain","points"])
    is_predictive  = any(w in q for w in ["predict","likely","probability","expect","forecast","will","chances","should i pick"])

    # ── Format detection ──────────────────────────────────
    fmt_t20  = any(w in q for w in ["t20","t-20","ipl","bbl","psl","cpl","sa20","hundred","the hundred"])
    fmt_odi  = any(w in q for w in ["odi","one day","one-day","50 over","50-over","list a","world cup odi"])
    fmt_test = any(w in q for w in ["test","red ball","test match","test cricket","test series"])

    # ── Entity extraction helpers ─────────────────────────
    def extract_player_names(text):
        """Heuristic: capitalised multi-word tokens not matching known keywords."""
        keywords = {"how","what","when","which","who","where","best","top","most","has","does",
                    "in","at","against","for","is","are","was","were","with","from","by","of","the","a","an"}
        tokens = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', text)
        return [t for t in tokens if t.lower() not in keywords]

    return {
        # Subject
        "is_batting":     is_batting,
        "is_bowling":     is_bowling,
        "is_fielding":    is_fielding,
        "is_team":        is_team,
        "is_h2h":         is_h2h,
        "is_allrounder":  is_allrounder,
        # Analytical
        "is_form":        is_form,
        "is_trend":       is_trend,
        "is_phase":       is_phase,
        "is_comparison":  is_comparison,
        "is_opponent":    is_opponent,
        "is_consistency": is_consistency,
        "is_chase":       is_chase,
        "is_pressure":    is_pressure,
        "is_leaderboard": is_leaderboard,
        "is_milestone":   is_milestone,
        "is_over_by_over":is_over_by_over,
        "is_venue":       is_venue,
        "is_fantasy":     is_fantasy,
        "is_predictive":  is_predictive,
        # Format
        "fmt_t20":  fmt_t20,
        "fmt_odi":  fmt_odi,
        "fmt_test": fmt_test,
        # Entities
        "candidate_names": extract_player_names(question),
    }


# =========================================================
# BUILD SCHEMA CONTEXT
# =========================================================

def build_schema_context(question=""):
    schema_lines = []
    for column, metadata in NV_PLAY_DICTIONARY.items():
        line = (
            f"\nColumn: {column}"
            f"\nDescription: {metadata['description']}"
            f"\nDatatype: {metadata['datatype']}"
            f"\nCategory: {metadata['category']}"
            f"\nAggregation: {metadata['aggregation']}"
            f"\nSynonyms: {', '.join(metadata['synonyms'])}"
        )
        schema_lines.append(line)
    return "\n".join(schema_lines)


# =========================================================
# BUILD METRICS CONTEXT
# =========================================================

def build_metrics_context():
    metric_lines = []
    for metric, metadata in DERIVED_METRICS.items():
        line = (
            f"\nMetric: {metric}"
            f"\nDescription: {metadata['description']}"
            f"\nFormula: {metadata['formula']}"
            f"\nCategory: {metadata['category']}"
            f"\nSynonyms: {', '.join(metadata['synonyms'])}"
        )
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

    lines.append("\nMATCH PHASES (ODI):")
    for phase, data in rules["match_phases"]["ODI"].items():
        lines.append(
            f"  {phase}: overs {data['start_over']}-{data['end_over']} -- {data['description']}"
        )

    lines.append("\nBATTING BENCHMARKS — apply the correct format based on match_type in the data:")
    lines.append("\n  T20 Strike Rate:")
    for tier, data in rules["batting_benchmarks"]["T20"]["strike_rate"].items():
        lines.append(f"    {tier}: {data}")
    lines.append("\n  ODI Strike Rate:")
    for tier, data in rules["batting_benchmarks"]["ODI"]["strike_rate"].items():
        lines.append(f"    {tier}: {data}")
    lines.append("\n  Test Batting Average:")
    for tier, data in rules["batting_benchmarks"]["TEST"]["batting_average"].items():
        lines.append(f"    {tier}: {data}")

    lines.append("\n  T20 Powerplay SR:")
    for tier, data in rules["batting_benchmarks"]["T20"]["powerplay_sr"].items():
        lines.append(f"    {tier}: {data}")
    lines.append("\n  ODI Powerplay SR:")
    for tier, data in rules["batting_benchmarks"]["ODI"]["powerplay_sr"].items():
        lines.append(f"    {tier}: {data}")

    lines.append("\n  T20 Death SR:")
    for tier, data in rules["batting_benchmarks"]["T20"]["death_sr"].items():
        lines.append(f"    {tier}: {data}")
    lines.append("\n  ODI Death SR:")
    for tier, data in rules["batting_benchmarks"]["ODI"]["death_sr"].items():
        lines.append(f"    {tier}: {data}")

    lines.append("\nBOWLING BENCHMARKS — apply the correct format based on match_type in the data:")
    lines.append("\n  T20 Economy Rate:")
    for tier, data in rules["bowling_benchmarks"]["T20"]["economy_rate"].items():
        lines.append(f"    {tier}: {data}")
    lines.append("\n  ODI Economy Rate:")
    for tier, data in rules["bowling_benchmarks"]["ODI"]["economy_rate"].items():
        lines.append(f"    {tier}: {data}")
    lines.append("\n  Test Economy Rate:")
    for tier, data in rules["bowling_benchmarks"]["TEST"]["economy_rate"].items():
        lines.append(f"    {tier}: {data}")

    lines.append("\n  T20 Powerplay Economy:")
    for tier, data in rules["bowling_benchmarks"]["T20"]["powerplay_economy"].items():
        lines.append(f"    {tier}: {data}")

    lines.append("\n  T20 Death Economy:")
    for tier, data in rules["bowling_benchmarks"]["T20"]["death_economy"].items():
        lines.append(f"    {tier}: {data}")

    lines.append("\n  T20 Dot Ball %:")
    for tier, data in rules["bowling_benchmarks"]["T20"]["dot_ball_percentage"].items():
        lines.append(f"    {tier}: {data}")
    lines.append("\n  ODI Dot Ball %:")
    for tier, data in rules["bowling_benchmarks"]["ODI"]["dot_ball_percentage"].items():
        lines.append(f"    {tier}: {data}")
    lines.append("\n  Test Dot Ball %:")
    for tier, data in rules["bowling_benchmarks"]["TEST"]["dot_ball_percentage"].items():
        lines.append(f"    {tier}: {data}")

    lines.append("\nFORMAT BENCHMARK SELECTION RULE FOR SQL PLANNER:")
    lines.append("  T20 / T20I  -> use T20 benchmarks")
    lines.append("  ODI         -> use ODI benchmarks")
    lines.append("  Test        -> use TEST benchmarks")
    lines.append("  Mixed       -> include all formats and benchmark each separately")
    lines.append("  NEVER apply T20 benchmarks when the data covers ODI or Test matches")

    lines.append("\nDISMISSAL TYPES:")
    for dtype, meta in rules["dismissal_types"].items():
        lines.append(f"  {dtype}: {meta['tactical_note']}")
        lines.append(f"    pattern_insight: {meta['pattern_insight']}")

    lines.append("\nLEGAL BALL:")
    lines.append(f"  {rules['legal_ball_rules']['analysis_note']}")
    lines.append(f"  discipline_insight: {rules['legal_ball_rules']['discipline_insight']}")

    lines.append("\nKEEPER POSITION:")
    lines.append(f"  keeper_up=TRUE: {rules['keeper_position_rules']['keeper_up']['impact']}")
    lines.append(f"  keeper_up=FALSE: {rules['keeper_position_rules']['keeper_back']['impact']}")

    lines.append("\nBOWLING ANGLE:")
    lines.append(f"  around_the_wicket: {rules['bowling_angle_rules']['around_the_wicket']['right_arm_to_right_batter']}")
    lines.append(f"  over_the_wicket:   {rules['bowling_angle_rules']['over_the_wicket']['meaning']}")

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

    lines.append("\nCONSISTENCY METRICS (T20):")
    for band, data in rules["consistency_metrics"]["innings_score_bands"].items():
        lines.append(f"  {band}: {data['range']} -> {data['label']}")
    for k, v in rules["consistency_metrics"]["consistency_thresholds"].items():
        lines.append(f"  {k}: {v}")

    lines.append("\nBATTING POSITION ROLES:")
    for role, data in rules["batting_position_roles"].items():
        lines.append(f"  {role} (pos {data['positions']}): {data['role']}")

    lines.append("\nPRESSURE / CLUTCH CONTEXT:")
    for item in rules["pressure_context"]["clutch_batting_situations"]:
        lines.append(f"  batting pressure: {item}")
    for item in rules["pressure_context"]["clutch_bowling_situations"]:
        lines.append(f"  bowling pressure: {item}")

    lines.append("\nOVER-BY-OVER EXPECTATIONS (T20):")
    for over, data in rules["over_context"]["T20_expected_runs_per_over"].items():
        lines.append(f"  {over}: {data['expected']} runs -- {data['note']}")

    lines.append("\nMILESTONE SQL PATTERNS:")
    for milestone, pattern in rules["milestone_context"]["sql_patterns"].items():
        lines.append(f"  {milestone}: {pattern}")

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
    lines.append("  ALL-ROUND:")
    for tier, label in rules["performance_labels"]["allround"].items():
        lines.append(f"    {tier}: {label}")

    lines.append("\nBATTING BENCHMARKS (ALL FORMATS — apply the one matching the data's match_type):")
    for fmt in ["T20", "ODI", "TEST"]:
        if fmt in rules["batting_benchmarks"]:
            lines.append(f"\n  [{fmt}]")
            for metric, tiers in rules["batting_benchmarks"][fmt].items():
                lines.append(f"  {metric}:")
                for tier, data in tiers.items():
                    lines.append(f"    {tier}: {data}")

    lines.append("\nBOWLING BENCHMARKS (ALL FORMATS — apply the one matching the data's match_type):")
    for fmt in ["T20", "ODI", "TEST"]:
        if fmt in rules["bowling_benchmarks"]:
            lines.append(f"\n  [{fmt}]")
            for metric, tiers in rules["bowling_benchmarks"][fmt].items():
                lines.append(f"  {metric}:")
                for tier, data in tiers.items():
                    lines.append(f"    {tier}: {data}")

    lines.append("\nFORMAT BENCHMARK SELECTION RULE:")
    lines.append("  CRITICAL: Always check match_type in the data to choose the right benchmark set.")
    lines.append("  If match_type = T20 or T20I   -> use T20 benchmarks above")
    lines.append("  If match_type = ODI or ODII   -> use ODI benchmarks above")
    lines.append("  If match_type = Test          -> use TEST benchmarks above")
    lines.append("  If mixed formats in data      -> state each format separately in the analysis")
    lines.append("  NEVER apply T20 benchmarks to ODI or Test data — they are not interchangeable")

    lines.append("\nINSIGHT RULES (follow all of these):")
    for rule in rules["insight_rules"]:
        lines.append(f"  - {rule}")

    lines.append("\nMATCH SITUATION CONTEXT:")
    for situation, data in rules["match_situations"].items():
        lines.append(f"  {situation}: {data['insight']}")

    lines.append("\nDISMISSAL TACTICAL CONTEXT:")
    for dtype, meta in rules["dismissal_types"].items():
        lines.append(f"  {dtype}:")
        lines.append(f"    tactical: {meta['tactical_note']}")
        lines.append(f"    causes: {', '.join(meta.get('common_causes', []))}")
        lines.append(f"    pattern: {meta['pattern_insight']}")

    lines.append("\nSCORING ZONE CONTEXT:")
    for zone, data in rules["scoring_zone_context"].items():
        lines.append(f"  {zone}: {data['insight']}")

    lines.append("\nHEAD TO HEAD INSIGHT RULES:")
    for note in rules["head_to_head_rules"]["insight_notes"]:
        lines.append(f"  - {note}")
    for note in rules["head_to_head_rules"]["dismissal_pattern_notes"]:
        lines.append(f"  - {note}")

    lines.append("\nCONSISTENCY METRICS:")
    for band, data in rules["consistency_metrics"]["innings_score_bands"].items():
        lines.append(f"  {band}: {data['range']} -> {data['label']}")
    for k, v in rules["consistency_metrics"]["consistency_thresholds"].items():
        lines.append(f"  threshold: {k} = {v}")

    lines.append("\nRECENT FORM CONTEXT:")
    for k, v in rules["recent_form_context"]["form_vs_career"].items():
        lines.append(f"  {k}: {v}")
    for note in rules["recent_form_context"]["form_trend_insight"]:
        lines.append(f"  - {note}")

    lines.append("\nOPPONENT ANALYSIS CONTEXT:")
    bogey = rules["opponent_analysis_context"]["bogey_opponent"]
    fav   = rules["opponent_analysis_context"]["favourite_opponent"]
    lines.append(f"  bogey_opponent indicator (batting): {bogey['indicator_batting']}")
    lines.append(f"  bogey_opponent indicator (bowling): {bogey['indicator_bowling']}")
    lines.append(f"  favourite_opponent indicator (batting): {fav['indicator_batting']}")
    lines.append(f"  quality note: {rules['opponent_analysis_context']['opposition_quality_note']}")

    lines.append("\nMATCH SITUATION PRESSURE CONTEXT:")
    for item in rules["pressure_context"]["clutch_batting_situations"]:
        lines.append(f"  clutch batting: {item}")
    lines.append(f"  pressure indicator (batting): {rules['pressure_context']['pressure_indicators']['batting']}")
    lines.append(f"  pressure indicator (bowling): {rules['pressure_context']['pressure_indicators']['bowling']}")

    lines.append("\nOVER CONTEXT (T20 expected RPO — apply only when match_type is T20/T20I):")
    for over, data in rules["over_context"]["T20_expected_runs_per_over"].items():
        lines.append(f"  {over}: expected {data['expected']} -- {data['note']}")
    lines.append("  NOTE: These RPO expectations apply to T20 only. For ODI, par RPO is ~5-6 in middle overs")
    lines.append("  and 8-9 in the last 10 overs. For Test, sessions are measured in runs/session not RPO.")

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
    lines.append("  number 3 / no.3        -> batting_position = 3")
    lines.append("  number 4 / no.4        -> batting_position = 4")

    lines.append("\nPHASES (T20):")
    lines.append("  powerplay / pp / pp1   -> over_number BETWEEN 1 AND 6")
    lines.append("  middle overs           -> over_number BETWEEN 7 AND 15")
    lines.append("  death / slog overs     -> over_number BETWEEN 16 AND 20")
    lines.append("  first over             -> over_number = 1")
    lines.append("  last / final over      -> over_number = 20")

    lines.append("\nPHASES (ODI):")
    lines.append("  odi powerplay          -> over_number BETWEEN 1 AND 10")
    lines.append("  odi middle overs       -> over_number BETWEEN 11 AND 40")
    lines.append("  odi death overs        -> over_number BETWEEN 41 AND 50")

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
    lines.append("  wide                   -> legal_ball = FALSE")
    lines.append("  no ball                -> legal_ball = FALSE")
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

    lines.append("\nPERFORMANCE FORMULAS (PostgreSQL only):")
    lines.append("  economy rate        -> ROUND((SUM(runs_total)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*6, 2)")
    lines.append("  batting strike rate -> ROUND((SUM(runs_batter)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*100, 2)")
    lines.append("  batting average     -> ROUND(SUM(runs_batter)::numeric / NULLIF(COUNT(*) FILTER (WHERE wicket IS NOT NULL),0), 2)")
    lines.append("  bowling average     -> ROUND(SUM(runs_total)::numeric / NULLIF(COUNT(*) FILTER (WHERE wicket IS NOT NULL),0), 2)")
    lines.append("  dot ball %          -> ROUND((COUNT(*) FILTER (WHERE runs_total=0 AND legal_ball=TRUE)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*100, 2)")
    lines.append("  boundary %          -> ROUND((COUNT(*) FILTER (WHERE runs_batter IN (4,6))::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*100, 2)")
    lines.append("  six %               -> ROUND((COUNT(*) FILTER (WHERE runs_batter=6)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*100, 2)")
    lines.append("  scoring rate        -> ROUND((COUNT(*) FILTER (WHERE runs_batter>0 AND legal_ball=TRUE)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*100, 2)")
    lines.append("  wickets per match   -> ROUND(COUNT(*) FILTER (WHERE wicket IS NOT NULL)::numeric / NULLIF(COUNT(DISTINCT match_id),0), 2)")
    lines.append("  bowling SR          -> ROUND(COUNT(*) FILTER (WHERE legal_ball=TRUE)::numeric / NULLIF(COUNT(*) FILTER (WHERE wicket IS NOT NULL),0), 2)")

    lines.append("\nPHASE-CONDITIONAL AGGREGATIONS:")
    lines.append("  SR in powerplay  -> ROUND((SUM(runs_batter) FILTER (WHERE over_number BETWEEN 1 AND 6)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE AND over_number BETWEEN 1 AND 6),0))*100,2)")
    lines.append("  SR in death      -> ROUND((SUM(runs_batter) FILTER (WHERE over_number BETWEEN 16 AND 20)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE AND over_number BETWEEN 16 AND 20),0))*100,2)")
    lines.append("  eco in powerplay -> ROUND((SUM(runs_total) FILTER (WHERE over_number BETWEEN 1 AND 6)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE AND over_number BETWEEN 1 AND 6),0))*6,2)")
    lines.append("  eco in death     -> ROUND((SUM(runs_total) FILTER (WHERE over_number BETWEEN 16 AND 20)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE AND over_number BETWEEN 16 AND 20),0))*6,2)")

    return "\n".join(lines)


# =========================================================
# DATE / TIME CONTEXT
# =========================================================

def build_date_context():
    lines = []

    lines.append("DATE & TIME COLUMN RULES — READ EVERY LINE:")
    lines.append("")
    lines.append("PRIMARY DATE COLUMN:")
    lines.append("  match_date   -> PostgreSQL DATE type (stored as 'YYYY-MM-DD')")
    lines.append("  ALL year/season/month/day/recent/latest filters use match_date")
    lines.append("  NEVER attempt to cast or convert match_date — it is already a DATE")
    lines.append("")
    lines.append("YEAR EXTRACTION (PostgreSQL ONLY):")
    lines.append("  EXTRACT(YEAR FROM match_date)::int AS year   -- ALWAYS cast to int, ALWAYS alias as 'year'")
    lines.append("  DATE_PART('year', match_date)                -- identical result")
    lines.append("  TO_CHAR(match_date, 'YYYY')                  -- returns TEXT '2023', use for display only")
    lines.append("  NEVER use: YEAR() / match_date.year / DATEPART(year,...)")
    lines.append("")
    lines.append("MONTH / DAY EXTRACTION:")
    lines.append("  EXTRACT(MONTH FROM match_date)::int          -- integer 1-12")
    lines.append("  EXTRACT(DAY FROM match_date)::int            -- integer 1-31")
    lines.append("  TO_CHAR(match_date, 'Mon')                   -- 'Jan'..'Dec'")
    lines.append("  TO_CHAR(match_date, 'YYYY-MM')               -- '2023-04' for monthly grouping")
    lines.append("  NEVER use: MONTH() / DAY() -- MySQL syntax only")
    lines.append("")
    lines.append("RELATIVE DATE WINDOWS (PostgreSQL intervals):")
    lines.append("  'last N years'   -> WHERE match_date >= CURRENT_DATE - INTERVAL 'N years'")
    lines.append("  'last N months'  -> WHERE match_date >= CURRENT_DATE - INTERVAL 'N months'")
    lines.append("  'last N days'    -> WHERE match_date >= CURRENT_DATE - INTERVAL 'N days'")
    lines.append("  'this year'      -> WHERE EXTRACT(YEAR FROM match_date) = EXTRACT(YEAR FROM CURRENT_DATE)")
    lines.append("  'last year'      -> WHERE EXTRACT(YEAR FROM match_date) = EXTRACT(YEAR FROM CURRENT_DATE) - 1")
    lines.append("  'since YYYY'     -> WHERE match_date >= 'YYYY-01-01'::date")
    lines.append("  'before YYYY'    -> WHERE match_date < 'YYYY-01-01'::date")
    lines.append("  'in YYYY'        -> WHERE EXTRACT(YEAR FROM match_date) = YYYY")
    lines.append("  'recent/latest'  -> ORDER BY match_date DESC LIMIT 10 (or 1)")
    lines.append("")
    lines.append("LAST N MATCHES — use CTE (NOT date arithmetic — matches are not daily):")
    lines.append("""
  WITH last_matches AS (
    SELECT DISTINCT match_id, match_date
    FROM public.nv_play
    WHERE batter ILIKE '%name%'
    ORDER BY match_date DESC
    LIMIT 10
  )
  SELECT
    p.match_date,
    p.match_id,
    SUM(p.runs_batter)                                                             AS runs,
    COUNT(*) FILTER (WHERE p.legal_ball = TRUE)                                    AS balls,
    ROUND((SUM(p.runs_batter)::numeric / NULLIF(COUNT(*) FILTER (WHERE p.legal_ball=TRUE),0))*100,2) AS strike_rate,
    MAX(p.wicket)                                                                   AS dismissal
  FROM public.nv_play p
  JOIN last_matches lm ON p.match_id = lm.match_id
  WHERE p.batter ILIKE '%name%'
  GROUP BY p.match_date, p.match_id
  ORDER BY p.match_date DESC
""")
    lines.append("CORRECT INTERVAL SYNTAX:")
    lines.append("  CURRENT_DATE - INTERVAL '2 years'    -- correct")
    lines.append("  CURRENT_DATE - INTERVAL '6 months'   -- correct")
    lines.append("  NOW() - INTERVAL '1 year'            -- correct (returns TIMESTAMP)")
    lines.append("  DATEADD(year, -2, CURRENT_DATE)      -- WRONG: SQL Server")
    lines.append("  DATE_ADD(d, INTERVAL -2 YEAR)        -- WRONG: MySQL")
    lines.append("")
    lines.append("YEARLY TREND PATTERN:")
    lines.append("""
  SELECT
    EXTRACT(YEAR FROM match_date)::int                                              AS year,
    COUNT(DISTINCT match_id)                                                        AS matches,
    SUM(runs_batter)                                                                AS runs,
    COUNT(*) FILTER (WHERE legal_ball=TRUE)                                         AS balls,
    ROUND((SUM(runs_batter)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*100,2) AS strike_rate,
    ROUND(SUM(runs_batter)::numeric / NULLIF(COUNT(*) FILTER (WHERE wicket IS NOT NULL),0),2)    AS average
  FROM public.nv_play
  WHERE batter ILIKE '%player%'
  GROUP BY EXTRACT(YEAR FROM match_date)
  ORDER BY year ASC
""")
    lines.append("MONTHLY BREAKDOWN PATTERN:")
    lines.append("""
  SELECT
    TO_CHAR(match_date, 'YYYY-MM')                                                  AS month,
    COUNT(DISTINCT match_id)                                                        AS matches,
    SUM(runs_batter)                                                                AS runs,
    ROUND((SUM(runs_batter)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*100,2) AS strike_rate
  FROM public.nv_play
  WHERE batter ILIKE '%player%'
  GROUP BY TO_CHAR(match_date, 'YYYY-MM')
  ORDER BY month ASC
""")
    lines.append("NATURAL LANGUAGE -> SQL MAPPING:")
    lines.append("  'every year / year by year'   -> GROUP BY EXTRACT(YEAR FROM match_date) ORDER BY year ASC")
    lines.append("  'monthly breakdown'            -> GROUP BY TO_CHAR(match_date,'YYYY-MM') ORDER BY month ASC")
    lines.append("  'last 5 matches'               -> CTE with LIMIT 5 on distinct match_ids DESC by match_date")
    lines.append("  'this season / current year'   -> EXTRACT(YEAR FROM match_date) = EXTRACT(YEAR FROM CURRENT_DATE)")
    lines.append("  'summer 2023'                  -> match_date BETWEEN '2023-04-01' AND '2023-09-30'")
    lines.append("")
    lines.append("ALWAYS: alias EXTRACT result as 'year' / 'month' / 'day'")
    lines.append("ALWAYS: ORDER BY time column ASC for trend queries")
    lines.append("NEVER: YEAR() MONTH() DAY() DATEADD() DATE_ADD() DATEPART()")

    return "\n".join(lines)


# =========================================================
# BUILD FORMAT CONTEXT
# =========================================================

def build_format_context():
    lines = []

    lines.append("CRICKET FORMAT RULES:")
    lines.append("")
    lines.append("FORMAT DETECTION from question:")
    lines.append("  T20 / T20I / IPL / BBL / PSL / CPL     -> match_type ILIKE '%T20%'")
    lines.append("  ODI / one-day / 50-over / List A        -> match_type ILIKE '%ODI%'")
    lines.append("  Test / Test match / red-ball            -> match_type ILIKE '%Test%'")
    lines.append("  If no format specified                  -> DO NOT filter match_type")
    lines.append("")
    lines.append("OVER RANGES BY FORMAT:")
    lines.append("  T20  : total 1-20  | powerplay 1-6   | middle 7-15  | death 16-20")
    lines.append("  ODI  : total 1-50  | powerplay 1-10  | middle 11-40 | death 41-50")
    lines.append("  Test : no limit    | new ball 1-20   | middle 21-60 | old ball 61+")
    lines.append("")
    lines.append("FORMAT-SPECIFIC BENCHMARKS:")
    lines.append("  T20  batting SR elite >160 | good 140-160 | average 125-140")
    lines.append("  ODI  batting SR elite >110 | good  95-110 | average  85-95")
    lines.append("  T20  bowling eco elite <6.5 | good 6.5-8.5 | poor >9.5")
    lines.append("  ODI  bowling eco elite <4.5 | good 4.5-5.5 | poor >6.5")
    lines.append("")
    lines.append("TOURNAMENT / COMPETITION FILTERING:")
    lines.append("  IPL           -> competition ILIKE '%IPL%'")
    lines.append("  T20 World Cup -> competition ILIKE '%World Cup%' AND match_type ILIKE '%T20%'")
    lines.append("  ODI World Cup -> competition ILIKE '%World Cup%' AND match_type ILIKE '%ODI%'")
    lines.append("  Champions Trophy -> competition ILIKE '%Champions Trophy%'")
    lines.append("  Big Bash       -> competition ILIKE '%Big Bash%'")

    return "\n".join(lines)


# =========================================================
# BUILD ADVANCED QUERY PATTERNS CONTEXT
# =========================================================

def build_advanced_patterns_context():
    lines = []
    lines.append("ADVANCED QUERY PATTERNS:")
    lines.append("")

    lines.append("1. CAREER BASELINE (include for EVERY player query):")
    lines.append("""
  SELECT
    batter,
    COUNT(DISTINCT match_id)                                                        AS matches,
    COUNT(DISTINCT match_id || '-' || innings_number)                               AS innings,
    SUM(runs_batter)                                                                AS career_runs,
    COUNT(*) FILTER (WHERE legal_ball=TRUE)                                         AS career_balls,
    ROUND((SUM(runs_batter)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*100,2)  AS career_sr,
    ROUND(SUM(runs_batter)::numeric / NULLIF(COUNT(*) FILTER (WHERE wicket IS NOT NULL),0),2)     AS career_avg,
    COUNT(*) FILTER (WHERE runs_batter=4)                                           AS fours,
    COUNT(*) FILTER (WHERE runs_batter=6)                                           AS sixes,
    ROUND((COUNT(*) FILTER (WHERE runs_batter IN (4,6))::numeric
           / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*100,2)             AS boundary_pct,
    ROUND((COUNT(*) FILTER (WHERE runs_total=0 AND legal_ball=TRUE)::numeric
           / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*100,2)             AS dot_pct
  FROM public.nv_play
  WHERE batter ILIKE '%name%'
  GROUP BY batter
""")

    lines.append("2. FULL PHASE SPLIT (one query, conditional aggregation):")
    lines.append("""
  SELECT
    batter,
    SUM(runs_batter) FILTER (WHERE over_number BETWEEN 1  AND 6)                   AS pp_runs,
    COUNT(*) FILTER (WHERE legal_ball=TRUE AND over_number BETWEEN 1  AND 6)        AS pp_balls,
    ROUND((SUM(runs_batter) FILTER (WHERE over_number BETWEEN 1  AND 6)::numeric
           / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE AND over_number BETWEEN 1  AND 6),0))*100,2) AS pp_sr,
    SUM(runs_batter) FILTER (WHERE over_number BETWEEN 7  AND 15)                  AS mid_runs,
    COUNT(*) FILTER (WHERE legal_ball=TRUE AND over_number BETWEEN 7  AND 15)       AS mid_balls,
    ROUND((SUM(runs_batter) FILTER (WHERE over_number BETWEEN 7  AND 15)::numeric
           / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE AND over_number BETWEEN 7  AND 15),0))*100,2) AS mid_sr,
    SUM(runs_batter) FILTER (WHERE over_number BETWEEN 16 AND 20)                  AS death_runs,
    COUNT(*) FILTER (WHERE legal_ball=TRUE AND over_number BETWEEN 16 AND 20)       AS death_balls,
    ROUND((SUM(runs_batter) FILTER (WHERE over_number BETWEEN 16 AND 20)::numeric
           / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE AND over_number BETWEEN 16 AND 20),0))*100,2) AS death_sr
  FROM public.nv_play
  WHERE batter ILIKE '%name%'
  GROUP BY batter
""")

    lines.append("3. RECENT FORM — last N matches (CTE):")
    lines.append("""
  WITH last_matches AS (
    SELECT DISTINCT match_id, match_date
    FROM public.nv_play
    WHERE batter ILIKE '%name%'
    ORDER BY match_date DESC
    LIMIT 10
  )
  SELECT
    p.match_date,
    p.match_id,
    SUM(p.runs_batter)                                                              AS runs,
    COUNT(*) FILTER (WHERE p.legal_ball=TRUE)                                       AS balls,
    ROUND((SUM(p.runs_batter)::numeric / NULLIF(COUNT(*) FILTER (WHERE p.legal_ball=TRUE),0))*100,2) AS sr,
    MAX(p.wicket)                                                                   AS dismissal
  FROM public.nv_play p
  JOIN last_matches lm ON p.match_id = lm.match_id
  WHERE p.batter ILIKE '%name%'
  GROUP BY p.match_date, p.match_id
  ORDER BY p.match_date DESC
""")

    lines.append("4. CONSISTENCY / INNINGS SCORE DISTRIBUTION:")
    lines.append("""
  SELECT
    CASE
      WHEN innings_runs = 0                    THEN 'Duck (0)'
      WHEN innings_runs BETWEEN 1  AND 9       THEN '1-9'
      WHEN innings_runs BETWEEN 10 AND 19      THEN '10-19'
      WHEN innings_runs BETWEEN 20 AND 29      THEN '20-29'
      WHEN innings_runs BETWEEN 30 AND 49      THEN '30-49'
      WHEN innings_runs BETWEEN 50 AND 74      THEN '50-74'
      WHEN innings_runs BETWEEN 75 AND 99      THEN '75-99'
      ELSE '100+'
    END AS score_band,
    COUNT(*) AS innings_count,
    ROUND(COUNT(*)::numeric / NULLIF(SUM(COUNT(*)) OVER(),0)*100,2) AS pct
  FROM (
    SELECT match_id, innings_number, batter, SUM(runs_batter) AS innings_runs
    FROM public.nv_play
    WHERE batter ILIKE '%name%'
    GROUP BY match_id, innings_number, batter
  ) t
  GROUP BY score_band
  ORDER BY MIN(innings_runs)
""")

    lines.append("5. OPPONENT-SPECIFIC PERFORMANCE (batter):")
    lines.append("""
  SELECT
    bowling_team                                                                    AS opponent,
    COUNT(DISTINCT match_id)                                                        AS matches,
    SUM(runs_batter)                                                                AS runs,
    COUNT(*) FILTER (WHERE legal_ball=TRUE)                                         AS balls,
    ROUND((SUM(runs_batter)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*100,2)  AS strike_rate,
    ROUND(SUM(runs_batter)::numeric / NULLIF(COUNT(*) FILTER (WHERE wicket IS NOT NULL),0),2)     AS average,
    COUNT(*) FILTER (WHERE wicket IS NOT NULL)                                      AS dismissals
  FROM public.nv_play
  WHERE batter ILIKE '%name%'
  GROUP BY bowling_team
  ORDER BY runs DESC
""")

    lines.append("6. SIDE-BY-SIDE PLAYER COMPARISON:")
    lines.append("""
  SELECT
    batter                                                                          AS player,
    COUNT(DISTINCT match_id)                                                        AS matches,
    SUM(runs_batter)                                                                AS runs,
    COUNT(*) FILTER (WHERE legal_ball=TRUE)                                         AS balls,
    ROUND((SUM(runs_batter)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*100,2)  AS strike_rate,
    ROUND(SUM(runs_batter)::numeric / NULLIF(COUNT(*) FILTER (WHERE wicket IS NOT NULL),0),2)     AS average,
    COUNT(*) FILTER (WHERE runs_batter=4)                                           AS fours,
    COUNT(*) FILTER (WHERE runs_batter=6)                                           AS sixes,
    ROUND((COUNT(*) FILTER (WHERE runs_total=0 AND legal_ball=TRUE)::numeric
           / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*100,2)             AS dot_pct,
    ROUND((COUNT(*) FILTER (WHERE runs_batter IN (4,6))::numeric
           / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*100,2)             AS boundary_pct
  FROM public.nv_play
  WHERE batter ILIKE '%p1%' OR batter ILIKE '%p2%'
  GROUP BY batter
""")

    lines.append("7. CHASE vs SETTING SPLIT:")
    lines.append("""
  SELECT
    CASE WHEN innings_number = 1 THEN 'Setting' ELSE 'Chasing' END                 AS innings_type,
    COUNT(DISTINCT match_id)                                                        AS matches,
    SUM(runs_batter)                                                                AS runs,
    ROUND((SUM(runs_batter)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*100,2) AS strike_rate,
    ROUND(SUM(runs_batter)::numeric / NULLIF(COUNT(*) FILTER (WHERE wicket IS NOT NULL),0),2)    AS average
  FROM public.nv_play
  WHERE batter ILIKE '%name%'
  GROUP BY innings_number
""")

    lines.append("8. OVER-BY-OVER RUN RATE AND WICKETS:")
    lines.append("""
  SELECT
    over_number,
    ROUND(AVG(over_runs),2)    AS avg_runs_per_over,
    SUM(over_wickets)          AS total_wickets,
    COUNT(*)                   AS sample_overs
  FROM (
    SELECT match_id, innings_number, over_number,
           SUM(runs_total)                                AS over_runs,
           COUNT(*) FILTER (WHERE wicket IS NOT NULL)     AS over_wickets
    FROM public.nv_play
    GROUP BY match_id, innings_number, over_number
  ) t
  GROUP BY over_number
  ORDER BY over_number
""")

    lines.append("9. DISMISSAL BREAKDOWN WITH %:")
    lines.append("""
  SELECT
    wicket                                                                           AS dismissal_type,
    COUNT(*)                                                                         AS count,
    ROUND(COUNT(*)::numeric / NULLIF(SUM(COUNT(*)) OVER(),0)*100,2)                 AS pct
  FROM public.nv_play
  WHERE batter ILIKE '%name%' AND wicket IS NOT NULL
  GROUP BY wicket
  ORDER BY count DESC
""")

    lines.append("10. HEAD-TO-HEAD MATCHUP:")
    lines.append("""
  SELECT
    batter, bowler,
    COUNT(*) FILTER (WHERE legal_ball=TRUE)                                          AS balls,
    SUM(runs_batter)                                                                 AS runs,
    COUNT(*) FILTER (WHERE wicket IS NOT NULL)                                       AS dismissals,
    ROUND((SUM(runs_batter)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*100,2) AS batter_sr,
    COUNT(*) FILTER (WHERE runs_total=0 AND legal_ball=TRUE)                         AS dots,
    COUNT(*) FILTER (WHERE runs_batter IN (4,6))                                     AS boundaries
  FROM public.nv_play
  WHERE batter ILIKE '%batter_name%' AND bowler ILIKE '%bowler_name%'
  GROUP BY batter, bowler
""")

    lines.append("11. TOP WICKET-TAKERS (team):")
    lines.append("""
  SELECT
    bowler,
    COUNT(DISTINCT match_id)                                                        AS matches,
    COUNT(*) FILTER (WHERE wicket IS NOT NULL)                                      AS wickets,
    COUNT(*) FILTER (WHERE legal_ball=TRUE)                                         AS balls,
    SUM(runs_total)                                                                 AS runs_conceded,
    ROUND((SUM(runs_total)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*6,2)    AS economy,
    ROUND(SUM(runs_total)::numeric / NULLIF(COUNT(*) FILTER (WHERE wicket IS NOT NULL),0),2)     AS bowling_avg,
    ROUND(COUNT(*) FILTER (WHERE legal_ball=TRUE)::numeric
          / NULLIF(COUNT(*) FILTER (WHERE wicket IS NOT NULL),0),2)                AS bowling_sr
  FROM public.nv_play
  WHERE bowling_team ILIKE '%team%'
  GROUP BY bowler
  ORDER BY wickets DESC
  LIMIT 15
""")

    lines.append("12. BOWLER vs BATTING POSITION:")
    lines.append("""
  SELECT
    batting_position,
    COUNT(*) FILTER (WHERE legal_ball=TRUE)                                         AS balls,
    SUM(runs_total)                                                                 AS runs_conceded,
    COUNT(*) FILTER (WHERE wicket IS NOT NULL)                                      AS wickets,
    ROUND((SUM(runs_total)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*6,2)    AS economy,
    ROUND(SUM(runs_total)::numeric / NULLIF(COUNT(*) FILTER (WHERE wicket IS NOT NULL),0),2)     AS bowling_avg
  FROM public.nv_play
  WHERE bowler ILIKE '%name%'
  GROUP BY batting_position
  ORDER BY batting_position
""")

    lines.append("13. FANTASY CRICKET — VALUE PLAYERS:")
    lines.append("""
  -- Top 20 all-format value players in last 30 days
  SELECT
    batter                                                                          AS player,
    COUNT(DISTINCT match_id)                                                        AS matches,
    SUM(runs_batter)                                                                AS runs,
    COUNT(*) FILTER (WHERE runs_batter = 4)                                         AS fours,
    COUNT(*) FILTER (WHERE runs_batter = 6)                                         AS sixes,
    ROUND((SUM(runs_batter)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*100,2) AS strike_rate,
    ROUND(SUM(runs_batter)::numeric / NULLIF(COUNT(DISTINCT match_id),0), 2)        AS avg_runs_per_match,
    -- approximate fantasy points: runs + 4s*1 + 6s*2
    (SUM(runs_batter) + COUNT(*) FILTER (WHERE runs_batter=4) + COUNT(*) FILTER (WHERE runs_batter=6)*2) AS approx_fantasy_pts
  FROM public.nv_play
  WHERE match_date >= CURRENT_DATE - INTERVAL '30 days'
  GROUP BY batter
  HAVING COUNT(DISTINCT match_id) >= 3
  ORDER BY approx_fantasy_pts DESC
  LIMIT 20
""")

    lines.append("14. VENUE / GROUND SPLIT:")
    lines.append("""
  SELECT
    venue,
    COUNT(DISTINCT match_id)                                                        AS matches,
    SUM(runs_batter)                                                                AS runs,
    ROUND((SUM(runs_batter)::numeric / NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*100,2) AS strike_rate,
    ROUND(SUM(runs_batter)::numeric / NULLIF(COUNT(*) FILTER (WHERE wicket IS NOT NULL),0),2)    AS average
  FROM public.nv_play
  WHERE batter ILIKE '%name%'
  GROUP BY venue
  ORDER BY runs DESC
""")

    lines.append("15. MILESTONE COUNTING:")
    lines.append("""
  SELECT
    EXTRACT(YEAR FROM match_date)::int AS year,
    COUNT(*) FILTER (WHERE innings_runs >= 100)  AS centuries,
    COUNT(*) FILTER (WHERE innings_runs >= 50 AND innings_runs < 100) AS fifties,
    COUNT(*) FILTER (WHERE innings_runs = 0)     AS ducks,
    COUNT(*)                                     AS innings
  FROM (
    SELECT match_id, innings_number, batter,
           match_date,
           SUM(runs_batter) AS innings_runs
    FROM public.nv_play
    WHERE batter ILIKE '%name%'
    GROUP BY match_id, innings_number, batter, match_date
  ) t
  GROUP BY EXTRACT(YEAR FROM match_date)
  ORDER BY year
""")

    return "\n".join(lines)


# =========================================================
# BUILD BASELINE QUERIES HINT
# =========================================================

def build_baseline_queries_hint():
    lines = []

    lines.append("PROACTIVE BASELINE QUERY STRATEGY — MANDATORY:")
    lines.append("")
    lines.append("For ANY specific PLAYER question, ALWAYS generate:")
    lines.append("  Q1 career_baseline     : Full career aggregates (matches, runs, SR, avg, 4s, 6s, dot%, boundary%)")
    lines.append("  Q2 phase_split         : Powerplay/Middle/Death SR, runs, balls in one conditional aggregation query")
    lines.append("  Q3 yearly_trend        : Year-by-year performance for trend context")
    lines.append("  Q4 question_specific   : The exact dimension asked about (recent form / opponent / dismissal / h2h)")
    lines.append("")
    lines.append("For ANY TEAM question, ALWAYS generate:")
    lines.append("  Q1 team_batting        : Batting totals, run rate, wickets lost, boundary%, dot%")
    lines.append("  Q2 team_bowling        : Economy, wickets, dot%, top bowlers")
    lines.append("  Q3 phase_split         : Powerplay/Middle/Death for both batting and bowling")
    lines.append("  Q4 question_specific   : Exact dimension asked about")
    lines.append("")
    lines.append("For RECENT FORM questions:")
    lines.append("  Q1 career_baseline     : Career stats as comparison anchor")
    lines.append("  Q2 last_10_matches     : Per-match scores via CTE")
    lines.append("  Q3 phase_split_recent  : Phase split within last 10 matches only")
    lines.append("")
    lines.append("For HEAD-TO-HEAD questions:")
    lines.append("  Q1 h2h_summary         : Overall h2h (balls, runs, dismissals, SR)")
    lines.append("  Q2 batter_career       : Batter career baseline (context for h2h SR)")
    lines.append("  Q3 bowler_career       : Bowler career baseline (context for h2h economy)")
    lines.append("  Q4 h2h_phase_split     : H2h split by powerplay/middle/death")
    lines.append("")
    lines.append("For LEADERBOARD questions:")
    lines.append("  Q1 main_leaderboard    : Ranked list with HAVING thresholds, top 15-20")
    lines.append("  Q2 phase_leaders       : Same metric split by phase")
    lines.append("")
    lines.append("For FANTASY questions:")
    lines.append("  Q1 recent_form         : Last 30 days batting + bowling")
    lines.append("  Q2 venue_split         : Performance at specific venue if given")
    lines.append("  Q3 h2h_matchup         : Key matchup data for captain/vice-captain picks")
    lines.append("")
    lines.append("RATIONALE: Career baseline is the comparison anchor that makes form/phase/opponent")
    lines.append("data meaningful. Without it, the insight layer cannot assess deviation from normal.")
    lines.append("Always include MORE context than strictly requested — insight layer will synthesise.")

    return "\n".join(lines)


# =========================================================
# BUILD SYSTEMIC CRICKET KNOWLEDGE
# =========================================================

def build_systemic_cricket_knowledge():
    lines = []

    lines.append("SYSTEMIC CRICKET KNOWLEDGE (blend this with statistical findings):")
    lines.append("")

    lines.append("BATTING ARCHETYPES:")
    lines.append("  AGGRESSOR: SR >145 (T20) / >100 (ODI), high boundary%, high 6:4 ratio. Targets off-side and straight.")
    lines.append("  ANCHOR: Moderate SR — T20: 120-135 / ODI: 80-90 / Test: avg 40+. Low dot%, builds platform.")
    lines.append("  FINISHER: Low/average career average (dies frequently), but exceptional death SR — T20: >175, ODI: >120.")
    lines.append("  ACCUMULATOR: High average, moderate SR, runs in singles/twos, very low duck rate across all formats.")
    lines.append("  PINCH-HITTER: High SR for a short burst, lower average, bats middle/lower order in white-ball cricket.")
    lines.append("")

    lines.append("BOWLING ARCHETYPES:")
    lines.append("  WICKET-TAKER: Bowling SR <15 (T20) / <30 (ODI) / <50 (Test), economy slightly higher, constant pressure.")
    lines.append("  MISER: Economy <7 (T20) / <5 (ODI) / <2.5 (Test) — dot-ball machine, containment specialist.")
    lines.append("  DEATH SPECIALIST: Higher overall economy acceptable if death economy <9.0 (T20) or <7.0 (ODI); executes yorkers.")
    lines.append("  POWERPLAY ATTACKER: New-ball swing/seam, attacks top order, powerplay economy <7.5 (T20) / <6.0 (ODI).")
    lines.append("  SPINNER / CONTROL: Middle-overs specialist — economy <7.5 (T20), <5.0 (ODI), <2.5 (Test); induces false shots.")
    lines.append("")

    lines.append("WHY DISMISSAL PATTERNS HAPPEN:")
    lines.append("  High caught%: Bowler forcing aerial shots — top edge from bouncer, leading edge off seam,")
    lines.append("    or lofted shot miscued off a slower ball. Batter's aerial game being exploited.")
    lines.append("  High LBW%: Late inswing targeting the pad, cutters, or spinner's arm ball hitting pad.")
    lines.append("    Batter playing away from the line or across their front pad.")
    lines.append("  High bowled%: Yorker through the gap, late inswing, or ball keeping low through the gate.")
    lines.append("    Batter playing away from the line or high back-lift creating an exposed base.")
    lines.append("  High stumped%: Bowler has sharp variation — googly, wrong-un, pace-off, or big turn.")
    lines.append("    Batter over-committing on drive and beaten outside the off stump.")
    lines.append("  High dot% (batter): Pressure from tight bowling, defensive mindset, or bowler targeting")
    lines.append("    a technical weakness (short of length, off stump channel, angle).")
    lines.append("")

    lines.append("FORMAT-AWARE BENCHMARK RULE (CRITICAL):")
    lines.append("  ALWAYS derive the format from the data (match_type column) before applying benchmarks.")
    lines.append("  NEVER default to T20 benchmarks when the format is unknown or mixed.")
    lines.append("  T20  : SR elite >160 | economy elite <6.5 | death eco elite <9.0 | powerplay overs 1-6")
    lines.append("  ODI  : SR elite >110 | economy elite <4.5 | death eco elite <7.0 | powerplay overs 1-10")
    lines.append("  Test : avg elite >55 | economy elite <2.0 | new ball phase overs 1-20")
    lines.append("  Mixed: state both formats separately — do NOT blend benchmarks across formats")
    lines.append("")

    lines.append("POWERPLAY TACTICAL CONTEXT (format-specific):")
    lines.append("  T20 powerplay (overs 1-6): SR >150 from openers is match-defining. 60+ runs is a strong")
    lines.append("  platform. Conceding 3 wickets in powerplay costs a batting team ~25-35 runs vs par.")
    lines.append("  ODI powerplay (overs 1-10): 55+ runs with 2 wickets or fewer is a dominant start.")
    lines.append("  SR >90 in ODI powerplay is above par; >110 is elite. Economy <7.0 for bowlers is good.")
    lines.append("  Test new ball (overs 1-20): Survival is the primary batting goal. Economy <2.5 for")
    lines.append("  bowlers is excellent. A wicket in the first 10 overs is disproportionately valuable.")
    lines.append("  The powerplay is the highest-leverage phase in ALL formats — but the metrics differ.")
    lines.append("")

    lines.append("DEATH OVERS CONTEXT (format-specific):")
    lines.append("  T20 death (overs 16-20): Economy below 9.0 is elite. SR >175 for batters is elite.")
    lines.append("  Batters failing here (SR <130) cost teams 15-25 extra runs per innings.")
    lines.append("  ODI death (overs 41-50): Economy below 7.0 is elite. SR >110 for batters is good.")
    lines.append("  The best death bowlers across formats mix yorkers, slower balls, and bouncers,")
    lines.append("  reading the batter's trigger movements and back-foot position.")
    lines.append("")

    lines.append("PHASE TRANSITION INSIGHT (format-aware):")
    lines.append("  A batter excelling in powerplay but poor in death = technique player struggling against")
    lines.append("  variations and yorkers once bowlers have established their rhythm.")
    lines.append("  A batter poor in powerplay but explosive at death = finisher-type, not an opener.")
    lines.append("  T20 middle overs (7-15): the quiet engine of high totals — wicket preservation here")
    lines.append("  is as valuable as boundary hitting. ODI middle overs (11-40): run-rate maintenance")
    lines.append("  and wicket conservation are both critical — the phase where all-rounders are pivotal.")
    lines.append("  Test middle session: spinners take over, partnership building is the goal.")
    lines.append("")

    lines.append("CONSISTENCY vs MATCH-WINNING (format-aware):")
    lines.append("  T20: A player with high SR but lower average can be MORE valuable than a consistent")
    lines.append("  accumulator. Match-winning innings (50+) are disproportionately valuable.")
    lines.append("  ODI: Balance of average AND SR matters equally. A 35-average, 95-SR batter is elite.")
    lines.append("  Test: Average is king. A player averaging 45+ is a genuine asset across any era.")
    lines.append("  Both average AND strike rate must always be assessed together across all formats.")
    lines.append("")

    lines.append("HEAD-TO-HEAD PSYCHOLOGY:")
    lines.append("  When a bowler dismisses the same batter 3+ times, a psychological edge develops.")
    lines.append("  Captains exploit this in knockout matches across all formats.")
    lines.append("  T20: A batter scoring SR >180 vs a bowler owns that matchup completely.")
    lines.append("  ODI: SR >120 vs a bowler across 20+ balls is batter dominance.")
    lines.append("  Test: A batter averaging 40+ vs a bowler across 5 innings has neutralised the threat.")
    lines.append("  The matchup history shapes field settings in every format.")
    lines.append("")

    lines.append("FORM vs CLASS:")
    lines.append("  A player in poor recent form (last 5 below average) with strong career credentials")
    lines.append("  is a 'class is permanent' situation — likely to return. A consistently poor performer")
    lines.append("  across 30+ innings has a structural problem: technique, role mismatch, or selection error.")
    lines.append("  This principle applies equally to T20, ODI, and Test cricket.")
    lines.append("")

    lines.append("CHASE PSYCHOLOGY (format-aware):")
    lines.append("  Chasing is harder psychologically in all white-ball formats — required rate pressure")
    lines.append("  compounds each over. T20 chases: required rate above 10 is difficult; above 12 is crisis.")
    lines.append("  ODI chases: above 7 RPO in the final 10 overs is under pressure; above 9 is very hard.")
    lines.append("  Batters with significantly higher chasing average/SR than setting stats are genuine")
    lines.append("  clutch performers — this is a premium, rare quality in any format.")
    lines.append("")

    lines.append("OPPONENT QUALITY:")
    lines.append("  Performing against strong bowling attacks is harder regardless of format.")
    lines.append("  A high SR against a weak team may reflect opposition quality, not individual brilliance.")
    lines.append("  Always contextualise opponent quality when interpreting performance splits.")
    lines.append("")

    lines.append("FANTASY CRICKET INTELLIGENCE:")
    lines.append("  High-value captain picks: consistent performers at a venue + recent form + good matchup.")
    lines.append("  Differentials: overlooked middle-order batters or death bowlers with good venue records.")
    lines.append("  Avoid: players in poor form even if high-profile. Venue batting/bowling average is critical.")
    lines.append("  Bowling picks: target bowlers with high dot% + format-appropriate economy in current conditions.")

    return "\n".join(lines)


# =========================================================
# CONVERSATIONAL CONTEXT BUILDER
# Injects prior Q&A pairs so the LLM can resolve
# indirect references ("same player", "how about ODIs",
# "compare with Kohli") correctly.
# =========================================================

def build_session_context(session_context: list) -> str:
    if not session_context:
        return ""
    lines = ["\nPREVIOUS CONVERSATION CONTEXT (use to resolve indirect references):"]
    for i, turn in enumerate(session_context[-4:], 1):   # last 4 turns only
        q = turn.get("question", "")
        a = turn.get("summary", "")[:300]                # truncated answer summary
        lines.append(f"  Turn {i}: Q: {q}")
        if a:
            lines.append(f"           A (summary): {a}")
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
        import traceback
        return {
            "status":     "error",
            "error_type": str(type(e)),
            "message":    str(e),
            "traceback":  traceback.format_exc()
        }


@app.get("/debug")
def debug():
    return {
        "DB_HOST": os.getenv("DB_HOST"),
        "DB_NAME": os.getenv("DB_NAME"),
        "DB_USER": os.getenv("DB_USER")
    }


# =========================================================
# VERSION
# Hit GET /version to confirm the correct build is live.
# Bump APP_VERSION at the top of this file on every deploy.
# =========================================================

@app.get("/version")
def version():
    import datetime
    return {
        "version":     APP_VERSION,
        "app":         "Cricket_Scorer_AI",
        "model":       "gemini-2.5-flash",
        "status":      "running",
        "server_time": datetime.datetime.utcnow().isoformat() + "Z"
    }


# =========================================================
# GEMINI TEST
# =========================================================

@app.get("/gemini-test")
def gemini_test():
    return {"response": llm("Say hello from Cricket_Scorer_AI")}


# =========================================================
# CLEAN SQL  — ENHANCED
# Catches all common LLM anti-patterns for PostgreSQL
# =========================================================

def clean_sql(sql_query):
    sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
    sql_query = " ".join(sql_query.split())

    # ---- Wicket boolean fixes ----
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
        sql_query = sql_query.replace(wrong, right)

    # ---- MySQL / MSSQL date functions ----
    sql_query = re.sub(r'\bYEAR\s*\(\s*(\w+)\s*\)',  r'EXTRACT(YEAR  FROM \1)', sql_query, flags=re.IGNORECASE)
    sql_query = re.sub(r'\bMONTH\s*\(\s*(\w+)\s*\)', r'EXTRACT(MONTH FROM \1)', sql_query, flags=re.IGNORECASE)
    sql_query = re.sub(r'\bDAY\s*\(\s*(\w+)\s*\)',   r'EXTRACT(DAY   FROM \1)', sql_query, flags=re.IGNORECASE)

    sql_query = re.sub(r'\bDATEPART\s*\(\s*year\s*,\s*(\w+)\s*\)',  r'EXTRACT(YEAR  FROM \1)', sql_query, flags=re.IGNORECASE)
    sql_query = re.sub(r'\bDATEPART\s*\(\s*month\s*,\s*(\w+)\s*\)', r'EXTRACT(MONTH FROM \1)', sql_query, flags=re.IGNORECASE)
    sql_query = re.sub(r'\bDATEPART\s*\(\s*day\s*,\s*(\w+)\s*\)',   r'EXTRACT(DAY   FROM \1)', sql_query, flags=re.IGNORECASE)

    # ---- NULL-safe function fixes ----
    sql_query = re.sub(r'\bIFNULL\s*\(', 'COALESCE(', sql_query, flags=re.IGNORECASE)
    sql_query = re.sub(r'\bNVL\s*\(',    'COALESCE(', sql_query, flags=re.IGNORECASE)
    sql_query = re.sub(r'\bISNULL\s*\(', 'COALESCE(', sql_query, flags=re.IGNORECASE)

    # ---- MySQL DATE_ADD / MSSQL DATEADD ----
    sql_query = re.sub(
        r'\bDATE_ADD\s*\(\s*(\w+)\s*,\s*INTERVAL\s+(-?\d+)\s+(\w+)\s*\)',
        r"\1 + INTERVAL '\2 \3'",
        sql_query, flags=re.IGNORECASE
    )
    sql_query = re.sub(
        r'\bDATEADD\s*\(\s*(\w+)\s*,\s*(-?\d+)\s*,\s*(\w+)\s*\)',
        r"\3 + INTERVAL '\2 \1'",
        sql_query, flags=re.IGNORECASE
    )

    # ---- SQL Server SELECT TOP N -> strip (add LIMIT at end) ----
    sql_query = re.sub(r'\bSELECT\s+TOP\s+(\d+)\s+', r'SELECT ', sql_query, flags=re.IGNORECASE)

    # ---- Backtick identifiers (MySQL) -> double-quote ----
    sql_query = re.sub(r'`(\w+)`', r'"\1"', sql_query)

    # ---- Ensure ::int cast on EXTRACT aliases ----
    sql_query = re.sub(
        r'EXTRACT\s*\(\s*(YEAR|MONTH|DAY)\s+FROM\s+(\w+)\s*\)\s+AS\s+(\w+)',
        r'EXTRACT(\1 FROM \2)::int AS \3',
        sql_query, flags=re.IGNORECASE
    )

    # ---- INSTR -> POSITION ----
    sql_query = re.sub(r'\bINSTR\s*\(', 'POSITION(', sql_query, flags=re.IGNORECASE)

    return sql_query.strip()


# =========================================================
# VALIDATE SQL
# =========================================================

def validate_sql(sql_query):
    allowed_keywords = ["SELECT", "WITH"]
    first_word = sql_query.strip().split()[0].upper()
    if first_word not in allowed_keywords:
        raise Exception("Only SELECT queries are allowed.")

    blocked_keywords = ["DELETE", "DROP", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "CREATE"]
    upper_sql = sql_query.upper()
    for keyword in blocked_keywords:
        if keyword in upper_sql:
            raise Exception(f"{keyword} operation not allowed.")


# =========================================================
# GENERATE QUERY PLAN  — ENHANCED
# Accepts intent dict + session context for higher accuracy
# =========================================================

def generate_query_plan(question, relax_thresholds=False,
                        intent: dict = None, session_context: list = None):

    schema_context    = build_schema_context(question)
    metrics_context   = build_metrics_context()
    rules_context     = build_rules_context()
    cricket_terms     = build_cricket_terms_context()
    date_context      = build_date_context()
    format_context    = build_format_context()
    advanced_patterns = build_advanced_patterns_context()
    baseline_hint     = build_baseline_queries_hint()
    session_ctx       = build_session_context(session_context or [])

    # Build intent hint from classification
    intent_hint = ""
    if intent:
        flags = [k for k, v in intent.items() if v is True]
        intent_hint = f"\nDETECTED INTENT FLAGS: {', '.join(flags)}\n"
        if intent.get("candidate_names"):
            intent_hint += f"CANDIDATE ENTITY NAMES: {', '.join(intent['candidate_names'])}\n"

    threshold_note = ""
    if relax_thresholds:
        threshold_note = """
RETRY MODE — PREVIOUS ATTEMPT RETURNED ZERO ROWS:
- DO NOT add HAVING clauses or minimum thresholds in any query.
- Return ALL rows regardless of sample size.
- Broaden ILIKE patterns: use shorter fragments.
- If original queries filtered by year, try without year filter.
"""

    prompt = f"""
You are Cricket_Scorer_AI — an elite cricket data engineer with encyclopaedic knowledge
of T20, ODI, Test, and all global domestic formats.

YOUR ONLY JOB: Translate the user's question into a precise, production-grade multi-query
PostgreSQL plan. The plan MUST include proactive baseline queries that contextualise the answer,
even if not explicitly asked for. Think like a coaching analyst building a comprehensive dossier,
not just answering the narrow question.

{threshold_note}

{intent_hint}
{session_ctx}

DATABASE:
  Table : public.nv_play
  Grain : ONE ROW = ONE DELIVERY (ball) in a cricket match
  Engine: PostgreSQL 14+

SCHEMA:
{schema_context}

DERIVED METRICS:
{metrics_context}

CRICKET RULES & BENCHMARKS:
{rules_context}

CRICKET TERMINOLOGY -> SQL:
{cricket_terms}

DATE & TIME RULES:
{date_context}

FORMAT & COMPETITION CONTEXT:
{format_context}

ADVANCED SQL PATTERNS:
{advanced_patterns}

PROACTIVE BASELINE STRATEGY:
{baseline_hint}

CRITICAL COLUMN RULES:
  wicket is TEXT: dismissed -> wicket IS NOT NULL | not out -> wicket IS NULL | specific -> wicket = 'Caught'
  NEVER: wicket = TRUE / FALSE / 1 / 0
  Booleans TRUE/FALSE: legal_ball, free_hit, around_the_wicket, keeper_up
  Runs ALWAYS aggregate: one row = one ball. Strike rate = (SUM(runs_batter)/NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*100
  Names: ILIKE '%name%' always. Schema columns only. Pure PostgreSQL.

QUERY DESIGN RULES:
  1. SELECT or WITH...SELECT only.
  2. Every computed column: meaningful alias.
  3. ROUND(value::numeric, 2) on ALL decimals.
  4. NULLIF(..., 0) on EVERY denominator.
  5. LIMIT 15-20 for open leaderboards; no LIMIT for named entities.
  6. ORDER BY most informative column. Time: ORDER BY year ASC.
  7. COUNT(DISTINCT match_id) AS matches for match counts.
  8. FILTER (WHERE ...) for phase splits in one query.
  9. CTEs for recent form, ranked matches, percentages.
  10. EXTRACT(YEAR FROM match_date)::int AS year — always.

HAVING THRESHOLD RULE:
  APPLY only for open leaderboards (no specific entity named):
    Batting leaderboard: HAVING COUNT(*) FILTER (WHERE legal_ball=TRUE) >= 30
    Bowling leaderboard: HAVING COUNT(*) FILTER (WHERE legal_ball=TRUE) >= 24
  NEVER apply HAVING when a specific player/team/year/position is named.

MULTI-QUERY STRATEGY:
  Simple player question    -> Q1 career_baseline + Q2 phase_split + Q3 question_specific
  Form question             -> Q1 career_baseline + Q2 last_10_matches + Q3 phase_split_recent
  Team question             -> Q1 team_batting + Q2 team_bowling + Q3 phase_split
  Comparison                -> Q1 side_by_side + Q2 year_by_year_both + Q3 phase_split_both
  H2H                       -> Q1 h2h_summary + Q2 batter_career + Q3 bowler_career + Q4 h2h_phase
  Leaderboard               -> Q1 main_leaderboard + Q2 phase_leaders
  Consistency/milestones    -> Q1 career_baseline + Q2 innings_distribution + Q3 dismissal_breakdown
  Fantasy                   -> Q1 recent_form_batting + Q2 recent_form_bowling + Q3 venue_split
  Predictive/analysis       -> Q1 career_baseline + Q2 opponent_split + Q3 phase_split + Q4 recent_form

ANALYSIS TYPE -> QUERY STRATEGY:
  CAREER BATTING    -> career_baseline + phase_split + dismissal_breakdown + yearly_trend
  CAREER BOWLING    -> career_baseline + phase_split + opponent_split + dismissal_methods
  RECENT FORM       -> career_baseline + last_10_matches + phase_split_recent
  YEARLY TREND      -> career_baseline + yearly_trend
  TEAM BATTING      -> team_batting_summary + phase_split + top_contributors
  TEAM BOWLING      -> team_bowling_summary + phase_split + top_wicket_takers
  HEAD TO HEAD      -> h2h_summary + batter_career + bowler_career + h2h_by_phase
  COMPARISON        -> side_by_side + year_by_year_both + phase_split_both
  PHASE ANALYSIS    -> phase_split_detailed + phase_comparison
  LEADERBOARD       -> main_leaderboard + phase_leaders
  DISMISSAL         -> dismissal_breakdown + dismissal_by_phase
  CONSISTENCY       -> innings_distribution + career_baseline + milestone_counts
  OPPONENT          -> opponent_split + career_baseline
  CHASE vs SETTING  -> innings_split + career_baseline
  MILESTONES        -> milestone_counts + career_baseline + yearly_milestones
  OVER ANALYSIS     -> over_by_over + phase_context
  FANTASY           -> recent_form_batting + recent_form_bowling + venue_split
  PREDICTIVE        -> career_baseline + form + matchup + phase_analysis

OUTPUT FORMAT — STRICT JSON ONLY, NO PREAMBLE, NO MARKDOWN:

{{
  "analysis_type": "single | multi",
  "intent": "1-2 precise sentences describing exactly what is asked including all dimensions",
  "has_time_dimension": true | false,
  "has_phase_dimension": true | false,
  "has_comparison": true | false,
  "has_recent_form": true | false,
  "has_fantasy_dimension": true | false,
  "has_predictive_dimension": true | false,
  "threshold_applied": true | false,
  "queries": [
    {{
      "name": "descriptive_snake_case_name",
      "purpose": "exactly what this query computes and why it is needed",
      "sql": "SELECT ..."
    }}
  ]
}}

USER QUESTION:
{question}
"""

    raw = llm(prompt).replace("```json", "").replace("```", "").strip()
    json_match = re.search(r'\{[\s\S]*\}', raw)
    if json_match:
        raw = json_match.group(0)

    try:
        return json.loads(raw)
    except Exception:
        return {
            "analysis_type":          "single",
            "has_time_dimension":      False,
            "has_phase_dimension":     False,
            "has_comparison":          False,
            "has_recent_form":         False,
            "has_fantasy_dimension":   False,
            "has_predictive_dimension":False,
            "threshold_applied":       False,
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
    format_context = build_format_context()

    prompt = f"""
You are a PostgreSQL expert for cricket ball-by-ball data.
TABLE: public.nv_play (each row = one delivery)

SCHEMA: {schema_context}
TERMINOLOGY: {cricket_terms}
DATE RULES: {date_context}
FORMAT RULES: {format_context}

Return ONLY raw SQL, no markdown, no backticks, no explanation.
Rules: SELECT only | PostgreSQL syntax | Schema columns only | wicket IS NOT NULL for dismissals
| legal_ball/free_hit/around_the_wicket/keeper_up are BOOLEAN | ROUND(x::numeric,2) | NULLIF(x,0)
| ILIKE '%name%' | NO HAVING | EXTRACT(YEAR FROM match_date)::int AS year | NO YEAR()/MONTH()/DATEADD()
| SR = (SUM(runs_batter)/NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*100
| Economy = (SUM(runs_total)/NULLIF(COUNT(*) FILTER (WHERE legal_ball=TRUE),0))*6

QUESTION: {question}
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

def execute_query_plan_with_retry(query_plan, question,
                                   intent: dict = None,
                                   session_context: list = None):
    all_results, all_sql = execute_query_plan(query_plan)

    if results_are_empty(all_results):
        # Attempt 1: Strip HAVING clauses
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

        if results_are_empty(retry_results):
            # Attempt 2: Full LLM retry with broad patterns
            fresh_plan = generate_query_plan(
                question,
                relax_thresholds=True,
                intent=intent,
                session_context=session_context
            )
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
# GENERATE CRICKET INSIGHT  — ENHANCED
# Full Gemini-grade reasoning with intent-aware depth
# =========================================================

def generate_cricket_insight(question, query_results,
                              small_sample=False,
                              intent: dict = None,
                              session_context: list = None):

    compact_data          = json.dumps(query_results, default=str)[:18000]
    insight_rules_context = build_insight_rules_context()
    format_context        = build_format_context()
    systemic_knowledge    = build_systemic_cricket_knowledge()
    session_ctx           = build_session_context(session_context or [])

    small_sample_note = ""
    if small_sample:
        small_sample_note = """
SMALL SAMPLE SIZE: Thresholds relaxed to retrieve this data. Mention naturally within
the analysis (e.g. "across limited appearances", "small sample caveat"). Do NOT refuse
to analyse. Still classify against benchmarks and deliver a full verdict.
"""

    # Use pre-classified intent if provided, else do keyword check
    if intent is None:
        intent = classify_intent(question)

    is_time_query     = intent.get("is_trend", False)
    is_phase_query    = intent.get("is_phase", False)
    is_comparison     = intent.get("is_comparison", False)
    is_form_query     = intent.get("is_form", False)
    is_opponent_query = intent.get("is_opponent", False)
    is_consistency    = intent.get("is_consistency", False)
    is_h2h            = intent.get("is_h2h", False)
    is_pressure       = intent.get("is_chase", False) or intent.get("is_pressure", False)
    is_allrounder     = intent.get("is_allrounder", False)
    is_leaderboard    = intent.get("is_leaderboard", False)
    is_fantasy        = intent.get("is_fantasy", False)
    is_predictive     = intent.get("is_predictive", False)
    is_milestone      = intent.get("is_milestone", False)
    is_over_by_over   = intent.get("is_over_by_over", False)
    is_venue          = intent.get("is_venue", False)

    dimension_notes = ""

    if is_time_query:
        dimension_notes += """
TIME-SERIES: Structure chronologically earliest to latest. Classify overall TRAJECTORY.
Call out BEST year and WORST year with exact figures. Describe year-on-year deltas.
Identify INFLECTION POINTS. In Verdict: upward / declining / plateauing arc?
"""
    if is_phase_query:
        dimension_notes += """
PHASE ANALYSIS: Dedicate a section to each phase. Classify each with tier label.
State STRONGEST phase, WEAKEST phase, and TACTICAL IMPLICATION for each.
Link phase weakness to dismissal patterns. Compare vs over-by-over benchmarks.
"""
    if is_comparison:
        dimension_notes += """
COMPARISON: Present BOTH entities side-by-side from the start. Lead with comparison table.
State CLEAR WINNER on each metric — no hedging. Overall winner with justification.
"""
    if is_form_query:
        dimension_notes += """
RECENT FORM: Separate CAREER STATS (anchor) from RECENT FORM clearly.
State whether form is hot (+20% above career), cold (-20% below), or normal (within 10%).
Identify hot streaks or lean patches with match evidence. Verdict: current momentum?
"""
    if is_opponent_query:
        dimension_notes += """
OPPONENT ANALYSIS: Rank opponents best to worst. Identify BOGEY OPPONENT and FAVOURITE.
Explain WHY — bowling attack type, conditions, historical context. Note small samples.
"""
    if is_consistency:
        dimension_notes += """
CONSISTENCY: Lead with innings distribution table. Calculate duck rate, 30+ rate, 50+ rate.
Classify as BOOM-OR-BUST or CONSISTENT ACCUMULATOR. Compare vs format-appropriate consistency thresholds.
Do NOT assume T20 — use whichever format the data reflects (T20, ODI, Test, or mixed).
"""
    if is_h2h:
        dimension_notes += """
HEAD-TO-HEAD: State WHO CONTROLS THE MATCHUP. Apply h2h thresholds (SR<100 = bowler controls).
Identify DISMISSAL PATTERN. Compare h2h SR to career SR. TACTICAL RECOMMENDATION for captain.
"""
    if is_pressure:
        dimension_notes += """
PRESSURE / CHASE: Separate SETTING vs CHASING stats. Classify as Clutch Performer or not.
Reference required rate context. State if higher chasing performance = genuine clutch asset.
"""
    if is_allrounder:
        dimension_notes += """
ALL-ROUNDER: Present batting AND bowling with equal weight. Apply all-rounder tier labels.
Check against format-appropriate all-rounder thresholds (T20: 200 runs + 10 wickets / ODI: 500 runs + 20 wickets / Test: 1000 runs + 50 wickets).
Apply the correct threshold based on the match_type in the data — do NOT default to T20 if matches span multiple formats.
State if genuine dual value exists.
"""
    if is_leaderboard:
        dimension_notes += """
LEADERBOARD: Rank top performers clearly. Highlight surprising entries, gaps between ranks.
Note what separates #1 from #2. Identify the 'most underrated' player in the list.
"""
    if is_fantasy:
        dimension_notes += """
FANTASY CRICKET: Frame every insight around selection impact.
Captain recommendation: highest floor + ceiling. Vice-captain: second-best risk/reward.
Differentials: overlooked players with good matchup data. Bold recommendation required.
State expected fantasy points range where data allows.
"""
    if is_predictive:
        dimension_notes += """
PREDICTIVE ANALYSIS: Ground predictions in career patterns, recent form, opponent matchup.
State probability qualitatively (likely / possible / unlikely) and justify with data.
Identify key risk factors. Avoid overclaiming — be calibrated.
"""
    if is_milestone:
        dimension_notes += """
MILESTONE TRACKING: Show year-by-year milestone counts (centuries, fifties, ducks).
Highlight best milestone season, career totals, and conversion rates (50->100).
"""
    if is_over_by_over:
        dimension_notes += """
OVER-BY-OVER: Link run rate expectations to the correct format benchmarks per over.
Identify the format from the data (T20: 20 overs, ODI: 50 overs, Test: unlimited) before applying expected RPO.
Identify overs where performance significantly deviates from expected RPO for that format.
"""
    if is_venue:
        dimension_notes += """
VENUE ANALYSIS: Rank venues by performance. Identify best and worst grounds.
Note if home/away split exists. Link venue conditions to performance pattern.
"""

    prompt = f"""
You are Cricket_Scorer_AI — the world's most advanced cricket analytics engine, combining:
- Statistical rigour of a professional data scientist
- Tactical depth of an elite coaching analyst (Andy Flower, Ravi Shastri depth)
- Storytelling quality of the finest cricket commentators (Harsha Bhogle, Nasser Hussain)
- Deep knowledge of player archetypes, tactical frameworks, and cricket history

You have database results AND a cricket knowledge base. Transform raw numbers into genuine insight.
BLEND systemic cricket knowledge with data where it explains WHY patterns exist.
NEVER fabricate statistics. NEVER mention SQL, database, queries, or data structures.

{small_sample_note}
{session_ctx}

USER QUESTION: {question}

DATABASE RESULTS:
{compact_data}

CRICKET RULES, BENCHMARKS & INSIGHT GUIDELINES:
{insight_rules_context}

FORMAT CONTEXT:
{format_context}

SYSTEMIC CRICKET KNOWLEDGE (blend with data):
{systemic_knowledge}

DIMENSION-SPECIFIC INSTRUCTIONS:
{dimension_notes}

REPORT STRUCTURE (follow exactly):

---

## 🏏 [Compelling specific headline — entity name + context + time period if relevant]

### 📊 Key Numbers
Markdown table. ALL data rows, no truncation.
- Career/player: Matches | Innings | Runs | Balls | SR | Avg | 4s | 6s | Dot% | Boundary%
- Time series:   Year | Matches | Runs/Wickets | SR/Economy | Average
- Phase split:   Phase | Balls | Runs | SR/Economy | Wickets | Dot%
- Comparison:    Metric | Entity A | Entity B (side-by-side)
- Leaderboard:   Rank | Name | Matches | primary metric | secondary metrics
- Fantasy:       Player | Form SR | Avg per match | Fours | Sixes | Approx Points | Selection Verdict

### 🔍 Deep Analysis

**Overall Picture**
BEFORE writing anything: scan the data for the match_type column values.
  - If match_type contains "T20" or "T20I" -> use T20 benchmarks
  - If match_type contains "ODI"            -> use ODI benchmarks
  - If match_type contains "Test"           -> use Test benchmarks
  - If match_type is missing or mixed       -> state "Format not confirmed from data" and apply conservative benchmarks; DO NOT default to T20
State the detected format as the first sentence of this section.
Then write 1-2 paragraphs with exact figures and tier labels (🔴 Elite / 🟠 Excellent / 🟡 Good / 🟢 Average / ⚪ Below Par).
Every tier label must reference the format-correct threshold. Example: "A strike rate of 68.75 falls ⚪ Below Par against the T20 benchmark of 110 minimum."
Connect stats to cricket context for that specific format. What does this mean for team strategy?

**Strengths**
2-3 evidence-backed strengths. Cricket language: corridor of uncertainty, hard length,
wrist position, release variation, powerplay intent. Every claim: a specific number.
Reference format-appropriate benchmarks when stating whether a stat is strong.

**Weaknesses / Vulnerabilities**
2-3 evidence-backed weaknesses. What would an opposition analyst exploit?
What does the captain set the field for? Exact stats.
Reference format-appropriate benchmarks when stating whether a stat is weak.

{"**Year-by-Year Trend** — chronological narrative, year-on-year deltas, best/worst year inflection points." if is_time_query else ""}
{"**Phase Breakdown** — Phases are format-dependent (T20: PP/Middle/Death | ODI: PP/Middle/Death | Test: New Ball/Middle/Old Ball). Identify the format first, then apply correct phase ranges. Classify each phase with tier label, state strongest/weakest phase, and tactical implication." if is_phase_query else ""}
{"**Head-to-Head Breakdown** — who controls the matchup, dismissal patterns, captain recommendation." if is_h2h else ""}
{"**Comparison** — metric-by-metric, clear winner on each, overall winner with justification. Ensure both entities are compared against the same format benchmarks." if is_comparison else ""}
{"**Opponent Split** — best and worst opponents, bogey team/bowler identified, tactical pattern." if is_opponent_query else ""}
{"**Consistency Profile** — score band distribution, duck rate, 30+ rate, boom-or-bust classifier. Apply format-appropriate consistency thresholds." if is_consistency else ""}
{"**Chase vs Setting** — side-by-side setting/chasing stats, clutch performer assessment." if is_pressure else ""}
{"**Fantasy Recommendation** — captain pick, vice-captain, differentials, avoid list with reasoning." if is_fantasy else ""}
{"**Predictive Outlook** — data-backed likelihood statements, key risk factors, selection recommendation." if is_predictive else ""}

**Tactical Intelligence**
What the opposition captain should plan for. What this team/player's coach should address.
Phase-specific bowling plans, field settings, batting approach suggestions — all format-specific.
Ground every suggestion in specific numbers from the data.

### 📈 Standout Moments / Records
4-6 bullets mixing impressive achievements AND concerning patterns.
Format: **[Bold the key stat]** — [1-2 sentence explanation of significance]
{"Include: best year, worst year, biggest year-on-year change." if is_time_query else ""}
{"Include: fantasy captain rationale, differential pick justification." if is_fantasy else ""}

### 💡 Verdict
5-7 sentences. Expert-panel quality. Open with tier classification and the format(s) it applies to.
{"Trajectory: upward / declining / plateauing and why." if is_time_query else ""}
{"Definitively state who is better and why." if is_comparison else ""}
{"Is this a clutch performer or does pressure expose them?" if is_pressure else ""}
{"Bold fantasy selection recommendation in final sentence." if is_fantasy else ""}
{"State predictive confidence level and primary risk." if is_predictive else ""}
Single most important finding. Forward-looking recommendation.

---

### ℹ️ Additional Context

⚠️ THIS SECTION IS MANDATORY. YOU MUST WRITE IT. DO NOT SKIP IT. DO NOT END THE RESPONSE AFTER THE VERDICT.
Every sub-heading below must appear verbatim. Fill in each one using the data provided.
If a field cannot be determined from the data, write "Not available from current data" — do NOT omit the heading.

**📋 Format & Sample**
State the cricket format this analysis covers. Derive it from the match_type values in the data.
If match_type is not in the data, state "Format not specified in data — benchmarks applied conservatively."
State the exact sample size: total balls faced or bowled, total innings, total matches.
State whether the sample is sufficient (High: 200+ balls / 15+ innings), moderate (Medium: 80-199 balls / 8-14 innings), or limited (Low: <80 balls / <8 innings) and what that means for reliability.

**📐 Benchmarks Used**
List the exact benchmark thresholds used to produce every tier label in this report.
Format each line as: [Metric] — [Format]: [tier thresholds]
Example: Strike Rate — T20: Poor <110 | Average 110-124 | Good 125-139 | Excellent 140-159 | Elite >160
Example: Economy Rate — ODI: Elite <4.0 | Excellent 4.0-4.79 | Good 4.8-5.49 | Average 5.5-5.99 | Poor >6.0
Include one benchmark line for every metric that appears in the Key Numbers table.

**📅 Data Coverage**
State the earliest and latest match_date visible in the data.
State which competitions or tournaments are covered if identifiable from the data.
Flag any obvious gaps — e.g. "Only 2 seasons visible; career may extend beyond this window."
If dates are not in the data, write: "Date range not available in current result set."

**🔗 Related Analyses**
Name exactly 2 follow-up analyses that would add the most useful context given what was asked.
Frame each as a direct question the user could ask next.
Example: "Phase breakdown: How does this player's strike rate split across powerplay, middle overs, and death overs?"
Example: "Recent form: Has performance improved or declined across the last 10 matches?"
Choose the 2 most analytically relevant follow-ups — not generic ones.

**🎯 Confidence Assessment**
State confidence as one of: 🟢 High | 🟡 Medium | 🔴 Low
Justify the rating with the sample size and pattern consistency observed.
State one specific thing that would upgrade the confidence level.
Example: "🔴 Low — 80 balls is insufficient for reliable T20 SR conclusions. Confidence upgrades to Medium with 200+ balls faced."

---

HARD OUTPUT RULES — VIOLATIONS ARE NOT ACCEPTABLE:
1. The response MUST contain all 5 sections of ### ℹ️ Additional Context — every heading must appear
2. NEVER end the response after ### 💡 Verdict — the Additional Context section always follows
3. NEVER assume T20 — derive format from match_type in the data; if unknown, state "format unknown"
4. Apply format-correct benchmarks: T20 SR elite >160 | ODI SR elite >110 | Test avg elite >55
5. Every tier label (Elite/Excellent/Good/Average/Below Par) must cite the benchmark that produced it
6. Exact numbers for every claim — no approximations, no hedging
7. Cricket terminology only — never mention SQL, database, queries, tables, or columns
8. 800-1500 words total across all sections
"""

    return llm(prompt)


# =========================================================
# GENERATE CHART CONFIG  — ENHANCED
# Supports fantasy, milestone, over-by-over, venue types
# =========================================================

def generate_chart_config(question, query_results, intent: dict = None):

    compact_data = json.dumps(query_results, default=str)[:10000]

    if intent is None:
        intent = classify_intent(question)

    is_time      = intent.get("is_trend", False)
    is_phase     = intent.get("is_phase", False)
    is_compare   = intent.get("is_comparison", False)
    is_over      = intent.get("is_over_by_over", False)
    is_dismissal = any(w in question.lower() for w in ["dismissal","how out","bowled","caught","lbw","wicket type"])
    is_dist      = intent.get("is_consistency", False)
    is_fantasy   = intent.get("is_fantasy", False)
    is_milestone = intent.get("is_milestone", False)
    is_venue     = intent.get("is_venue", False)

    hints = ""
    if is_time:
        hints += "TIME: year column present -> 'line' (1 metric) or 'bar' (2+ metrics). x_key='year'\n"
    if is_phase:
        hints += "PHASE: Powerplay/Middle/Death categories -> 'bar'. x_key='phase'\n"
    if is_compare:
        hints += "COMPARE: 2-5 entities, 2+ metrics -> 'bar'. 2-4 entities, 4-6 metrics -> 'radar'\n"
    if is_over:
        hints += "OVER-BY-OVER: over_number on x-axis -> 'line'. x_key='over_number'\n"
    if is_dismissal:
        hints += "DISMISSAL: types as categories -> 'pie' (pct) or 'bar_colored' (count)\n"
    if is_dist:
        hints += "DISTRIBUTION: score bands -> 'bar_colored'. x_key='score_band'\n"
    if is_fantasy:
        hints += "FANTASY: approx_fantasy_pts per player -> 'bar_colored'. x_key='player'\n"
    if is_milestone:
        hints += "MILESTONE: year + centuries + fifties stacked -> 'bar'. x_key='year'\n"
    if is_venue:
        hints += "VENUE: venue as x_key, runs or SR as y -> 'bar_colored'. x_key='venue'\n"

    prompt = f"""
Cricket data visualisation expert. Select the chart type that best communicates the PRIMARY INSIGHT.
Return null if no chart adds meaningful value.

HINTS FOR THIS QUERY:
{hints}

CHART TYPES:
  bar         : Multi-metric comparison (2+ y_keys). Max 12 rows.
  bar_colored : Single-metric leaderboard, each bar different colour. Max 15 rows.
  line        : Time/ordered trend (year, over_number). Max 25 rows.
  area        : Cumulative volume trend. Max 25 rows.
  pie         : Distribution / share of whole. 1 y_key. 3-8 categories.
  radar       : Multi-dimension profile. 2-4 entities, 4-6 metrics. Max 4 rows.

SELECTION PRIORITY:
1. year/time + 1 metric                     -> line
2. year/time + 2+ metrics                   -> bar
3. year/time + cumulative volume            -> area
4. over_number ordered                      -> line
5. dismissal/score band distribution        -> pie (pct) or bar_colored (count)
6. phase breakdown (PP/Mid/Death)           -> bar
7. compare N entities on 1 metric           -> bar_colored
8. compare N entities on 2+ metrics         -> bar
9. profile 2-4 players on 4-6 metrics       -> radar
10. fantasy points ranking                  -> bar_colored
11. milestone year-by-year                  -> bar
12. venue breakdown                         -> bar_colored
13. only 1 data row                         -> null
14. no numeric columns                      -> null

DATA RULES:
- x_key: string, category, or integer (year/over_number)
- y_keys: NUMERIC only. Exclude text columns, IDs, non-numeric identifiers.
- Do NOT include columns that would visually dwarf others (total_balls alongside strike_rate)
- Use the first/most relevant query result when multiple exist
- title: specific, include entity name. subtitle: optional one-liner.

RETURN STRICT JSON (no preamble, no markdown):
{{
  "chart_type": "bar|bar_colored|line|area|pie|radar",
  "title": "Specific title with entity name",
  "subtitle": "Optional context line",
  "x_key": "column_name",
  "y_keys": ["metric1", "metric2"],
  "data": [{{"x_value": ..., "metric1": ..., "metric2": ...}}]
}}

Return exactly: null    if no chart is appropriate.

USER QUESTION: {question}
DATABASE RESULTS: {compact_data}
"""

    raw = llm(prompt).replace("```json", "").replace("```", "").strip()

    if raw.lower().strip() == "null":
        return None

    json_match = re.search(r'\{[\s\S]*\}', raw)
    if json_match:
        raw = json_match.group(0)

    try:
        return json.loads(raw)
    except Exception:
        return None


# =========================================================
# GENERATE STRUCTURED INTENT SUMMARY
# Returns a short human-readable interpretation of the
# question so the frontend can display "Understanding: ..."
# =========================================================

def generate_intent_summary(question: str, intent: dict) -> str:
    flags = []
    mapping = {
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
    }
    for k, label in mapping.items():
        if intent.get(k):
            flags.append(label)

    fmt = []
    if intent.get("fmt_t20"):  fmt.append("T20")
    if intent.get("fmt_odi"):  fmt.append("ODI")
    if intent.get("fmt_test"): fmt.append("Test")

    parts = []
    if flags:
        parts.append(", ".join(flags))
    if fmt:
        parts.append(f"Format: {'/'.join(fmt)}")
    if intent.get("candidate_names"):
        parts.append(f"Entities: {', '.join(intent['candidate_names'])}")

    return " | ".join(parts) if parts else "General cricket query"


# =========================================================
# GENERATE SQL ROUTE
# =========================================================

@app.post("/generate-sql")
def generate_sql(req: QueryRequest):
    try:
        intent = classify_intent(req.question)
        return {
            "question":      req.question,
            "intent_flags":  {k: v for k, v in intent.items() if v is True},
            "intent_summary":generate_intent_summary(req.question, intent),
            "query_plan":    generate_query_plan(
                                req.question,
                                intent=intent,
                                session_context=req.session_context
                             )
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# =========================================================
# MAIN ASK ROUTE  — ENHANCED
# Full pipeline: classify → plan → execute → retry →
#                chart → insight → respond
# =========================================================

@app.post("/ask")
def ask_question(req: QueryRequest):
    try:

        # ── Step 0: Classify intent (once, reused everywhere) ──
        intent = classify_intent(req.question)
        intent_summary = generate_intent_summary(req.question, intent)

        # ── Step 1: Generate SQL query plan ──────────────────
        query_plan = generate_query_plan(
            req.question,
            intent=intent,
            session_context=req.session_context
        )

        # ── Step 2: Execute with auto-retry on empty results ──
        query_results, sql_queries, thresholds_relaxed = execute_query_plan_with_retry(
            query_plan,
            req.question,
            intent=intent,
            session_context=req.session_context
        )

        # ── Step 3: Format plain-text tables ─────────────────
        tables = [
            {
                "query_name": item["query_name"],
                "purpose":    item["purpose"],
                "table":      format_results(item["results"])
            }
            for item in query_results
        ]

        # ── Step 4: Generate chart config ────────────────────
        chart_config = generate_chart_config(
            req.question,
            query_results,
            intent=intent
        )

        # ── Step 5: Generate cricket insight ─────────────────
        insight = generate_cricket_insight(
            req.question,
            query_results,
            small_sample=thresholds_relaxed,
            intent=intent,
            session_context=req.session_context
        )

        # ── Step 6: Build response summary for session memory ─
        insight_summary = insight[:400] if insight else ""

        # ── Step 7: Return full response ─────────────────────
        return {
            "question":              req.question,
            "intent_summary":        intent_summary,
            "intent_flags": {
                "analysis_type":       query_plan.get("analysis_type",       "single"),
                "has_time_dimension":  query_plan.get("has_time_dimension",  False),
                "has_phase_dimension": query_plan.get("has_phase_dimension", False),
                "has_comparison":      query_plan.get("has_comparison",      False),
                "has_recent_form":     query_plan.get("has_recent_form",     False),
                "has_fantasy":         query_plan.get("has_fantasy_dimension", False),
                "has_predictive":      query_plan.get("has_predictive_dimension", False),
            },
            "thresholds_relaxed":    thresholds_relaxed,
            "sql_queries":           sql_queries,
            "results":               query_results,
            "tables":                tables,
            "chart_config":          chart_config,
            "insight":               insight,
            # Session memory helper — frontend should persist this
            # and pass it back as session_context in the next request
            "session_turn": {
                "question": req.question,
                "summary":  insight_summary
            }
        }

    except Exception as e:
        import traceback
        return {
            "status":    "error",
            "message":   str(e),
            "traceback": traceback.format_exc()
        }


# =========================================================
# INTERPRET ROUTE
# Pure NL interpretation without DB execution — for
# explaining what the system understands about a question
# =========================================================

@app.post("/interpret")
def interpret_question(req: QueryRequest):
    """
    Returns a structured interpretation of the user's question:
    intent flags, entities, format, analysis type, and a
    plain-English explanation — without running any SQL.
    """
    try:
        intent  = classify_intent(req.question)
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
            "question":        req.question,
            "intent_summary":  summary,
            "intent_flags":    {k: v for k, v in intent.items() if v is True},
            "explanation":     explanation
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}