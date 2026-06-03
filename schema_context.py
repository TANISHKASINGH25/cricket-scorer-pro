"""
NV_PLAY_DATA_DICTIONARY = {

    # =========================================================
    # MATCH INFORMATION
    # =========================================================

    "Competition": {
        "description": "Tournament or competition name",
        "datatype": "text",
        "category": "match_info",
        "aggregation": None,
        "synonyms": [
            "tournament",
            "series",
            "league",
            "competition"
        ]
    },

    "Match": {
        "description": "Match identifier or match name",
        "datatype": "text",
        "category": "match_info",
        "aggregation": None,
        "synonyms": [
            "game",
            "fixture",
            "match"
        ]
    },

    "Date": {
        "description": "Match date",
        "datatype": "date",
        "category": "match_info",
        "aggregation": None,
        "synonyms": [
            "match date",
            "game date"
        ]
    },

    "Start Time": {
        "description": "Scheduled start time of match",
        "datatype": "timestamp",
        "category": "match_info",
        "aggregation": None,
        "synonyms": [
            "start",
            "match start",
            "time"
        ]
    },

    # =========================================================
    # BALL / OVER INFORMATION
    # =========================================================

    "Innings": {
        "description": "Current innings number",
        "datatype": "integer",
        "category": "innings",
        "aggregation": None,
        "synonyms": [
            "innings",
            "inning"
        ]
    },

    "Over": {
        "description": "Over number in innings",
        "datatype": "integer",
        "category": "ball_tracking",
        "aggregation": None,
        "synonyms": [
            "over",
            "overs"
        ]
    },

    "Ball": {
        "description": "Ball number within over",
        "datatype": "integer",
        "category": "ball_tracking",
        "aggregation": None,
        "synonyms": [
            "delivery",
            "ball"
        ]
    },

    "Runs": {
        "description": "Runs scored from bat",
        "datatype": "integer",
        "category": "batting",
        "aggregation": "SUM",
        "synonyms": [
            "score",
            "batting runs",
            "runs scored",
            "scoring"
        ]
    },

    "Extra Runs": {
        "description": "Extra runs awarded on delivery",
        "datatype": "integer",
        "category": "extras",
        "aggregation": "SUM",
        "synonyms": [
            "extras",
            "wides",
            "no balls",
            "extra runs"
        ]
    },

    "Batter": {
        "description": "Name of striker batter",
        "datatype": "text",
        "category": "batting",
        "aggregation": None,
        "synonyms": [
            "batsman",
            "batter",
            "striker"
        ]
    },

    "Bowler": {
        "description": "Name of current bowler",
        "datatype": "text",
        "category": "bowling",
        "aggregation": None,
        "synonyms": [
            "bowler",
            "pitcher"
        ]
    },

    "Wicket": {
        "description": "Whether wicket fell on delivery",
        "datatype": "boolean",
        "category": "dismissal",
        "aggregation": "COUNT",
        "synonyms": [
            "dismissal",
            "out",
            "wicket"
        ]
    },

    "Power Play": {
        "description": "Whether delivery occurred during powerplay",
        "datatype": "boolean",
        "category": "match_phase",
        "aggregation": None,
        "synonyms": [
            "powerplay",
            "field restriction"
        ]
    },

    "Run Rate After": {
        "description": "Run rate after delivery",
        "datatype": "float",
        "category": "match_state",
        "aggregation": "AVG",
        "synonyms": [
            "run rate",
            "rr"
        ]
    },

    "Req Run Rate After": {
        "description": "Required run rate after delivery",
        "datatype": "float",
        "category": "match_state",
        "aggregation": "AVG",
        "synonyms": [
            "required run rate",
            "rrr"
        ]
    },

    "Shot": {
        "description": "Shot played by batter",
        "datatype": "text",
        "category": "batting",
        "aggregation": None,
        "synonyms": [
            "stroke",
            "shot",
            "batting shot"
        ]
    },

    "Line": {
        "description": "Line of delivery",
        "datatype": "text",
        "category": "bowling",
        "aggregation": None,
        "synonyms": [
            "bowling line",
            "line"
        ]
    },

    "Length": {
        "description": "Length of delivery",
        "datatype": "text",
        "category": "bowling",
        "aggregation": None,
        "synonyms": [
            "bowling length",
            "yorker",
            "short ball",
            "good length"
        ]
    },

    "SwingAngle": {
        "description": "Swing angle of ball",
        "datatype": "float",
        "category": "ball_physics",
        "aggregation": "AVG",
        "synonyms": [
            "swing",
            "movement",
            "late swing"
        ]
    },

    "Speed": {
        "description": "Ball release speed",
        "datatype": "float",
        "category": "ball_physics",
        "aggregation": "AVG",
        "synonyms": [
            "pace",
            "ball speed",
            "bowling speed"
        ]
    }

}

"""

NV_PLAY_DICTIONARY = {

    # =====================================================
    # MATCH INFORMATION
    # =====================================================

    "competition": {
        "description": "Tournament or competition name",
        "datatype": "TEXT",
        "category": "match_info",
        "aggregation": "groupable",
        "synonyms": ["tournament", "league", "series"]
    },

    "match": {
        "description": "Match name or fixture",
        "datatype": "TEXT",
        "category": "match_info",
        "aggregation": "groupable",
        "synonyms": ["fixture", "game"]
    },

    "date": {
        "description": "Match date",
        "datatype": "TEXT",
        "category": "match_info",
        "aggregation": "groupable",
        "synonyms": ["match date"]
    },

    "venue": {
        "description": "Cricket ground or stadium",
        "datatype": "TEXT",
        "category": "match_info",
        "aggregation": "groupable",
        "synonyms": ["stadium", "ground"]
    },

    "innings": {
        "description": "Innings number",
        "datatype": "INTEGER",
        "category": "match_info",
        "aggregation": "groupable",
        "synonyms": ["inning"]
    },

    "over": {
        "description": "Over number",
        "datatype": "INTEGER",
        "category": "match_info",
        "aggregation": "groupable",
        "synonyms": ["overs"]
    },

    "ball": {
        "description": "Ball number within over",
        "datatype": "INTEGER",
        "category": "match_info",
        "aggregation": "groupable",
        "synonyms": ["delivery"]
    },

    "innings_ball": {
        "description": "Ball number in innings",
        "datatype": "INTEGER",
        "category": "match_info",
        "aggregation": "groupable",
        "synonyms": ["delivery number"]
    },

    # =====================================================
    # BATTING
    # =====================================================

    "batter": {
        "description": "Name of batter",
        "datatype": "TEXT",
        "category": "batting",
        "aggregation": "groupable",
        "synonyms": ["batsman", "striker"]
    },

    "batter_id": {
        "description": "Unique batter identifier",
        "datatype": "TEXT",
        "category": "batting",
        "aggregation": "groupable",
        "synonyms": ["player id"]
    },

    "non_striker": {
        "description": "Non striker batter",
        "datatype": "TEXT",
        "category": "batting",
        "aggregation": "groupable",
        "synonyms": ["runner"]
    },

    "batting_position": {
        "description": "Batting order position",
        "datatype": "INTEGER",
        "category": "batting",
        "aggregation": "averageable",
        "synonyms": ["batting order"]
    },

    "batting_hand": {
        "description": "Batting handedness",
        "datatype": "TEXT",
        "category": "batting",
        "aggregation": "groupable",
        "synonyms": ["right hand", "left hand"]
    },

    "runs": {
        "description": "Runs scored by batter",
        "datatype": "INTEGER",
        "category": "batting",
        "aggregation": "summable",
        "synonyms": ["score", "batting runs"]
    },

    "cumulative_batter_runs": {
        "description": "Running total batter runs",
        "datatype": "INTEGER",
        "category": "batting",
        "aggregation": "max",
        "synonyms": ["career runs", "innings runs"]
    },

    "cumulative_batter_balls": {
        "description": "Running total balls faced",
        "datatype": "INTEGER",
        "category": "batting",
        "aggregation": "max",
        "synonyms": ["balls faced"]
    },

    "shot": {
        "description": "Type of batting shot",
        "datatype": "NUMERIC",
        "category": "batting",
        "aggregation": "groupable",
        "synonyms": ["stroke", "batting shot"]
    },

    "connection": {
        "description": "Quality of bat-ball connection",
        "datatype": "NUMERIC",
        "category": "batting",
        "aggregation": "averageable",
        "synonyms": ["timing", "middle"]
    },

    # =====================================================
    # BOWLING
    # =====================================================

    "bowler": {
        "description": "Name of bowler",
        "datatype": "TEXT",
        "category": "bowling",
        "aggregation": "groupable",
        "synonyms": ["bowler name"]
    },

    "bowler_id": {
        "description": "Unique bowler identifier",
        "datatype": "TEXT",
        "category": "bowling",
        "aggregation": "groupable",
        "synonyms": ["bowler code"]
    },

    "bowler_type": {
        "description": "Bowling style/type",
        "datatype": "TEXT",
        "category": "bowling",
        "aggregation": "groupable",
        "synonyms": ["pace", "spin"]
    },

    "delivery": {
        "description": "Type of delivery",
        "datatype": "NUMERIC",
        "category": "bowling",
        "aggregation": "groupable",
        "synonyms": ["ball type"]
    },

    "speed": {
        "description": "Ball release speed",
        "datatype": "NUMERIC",
        "category": "bowling",
        "aggregation": "averageable",
        "synonyms": ["pace", "velocity"]
    },

    "bouncespeed": {
        "description": "Ball speed after bounce",
        "datatype": "NUMERIC",
        "category": "bowling",
        "aggregation": "averageable",
        "synonyms": ["bounce speed"]
    },

    "swingangle": {
        "description": "Amount of swing movement",
        "datatype": "NUMERIC",
        "category": "bowling",
        "aggregation": "averageable",
        "synonyms": ["swing"]
    },

    "deviation": {
        "description": "Deviation after pitching",
        "datatype": "NUMERIC",
        "category": "bowling",
        "aggregation": "averageable",
        "synonyms": ["movement"]
    },

    "releaseX": {
        "description": "Bowler release X coordinate",
        "datatype": "NUMERIC",
        "category": "tracking",
        "aggregation": "averageable",
        "synonyms": ["release position"]
    },

    "releaseY": {
        "description": "Bowler release Y coordinate",
        "datatype": "NUMERIC",
        "category": "tracking",
        "aggregation": "averageable",
        "synonyms": ["release point"]
    },

    "releaseZ": {
        "description": "Bowler release height",
        "datatype": "NUMERIC",
        "category": "tracking",
        "aggregation": "averageable",
        "synonyms": ["release height"]
    },

    # =====================================================
    # DISMISSALS
    # =====================================================

    "wicket": {
    "description": "Type of dismissal. NULL means no wicket.",
    "datatype": "TEXT",
    "category": "dismissal",
    "aggregation": "count",
    "synonyms": [
        "dismissal",
        "out",
        "wicket type"
    ]
    },

    "dismissed_batter": {
        "description": "Dismissed batter name",
        "datatype": "TEXT",
        "category": "dismissal",
        "aggregation": "groupable",
        "synonyms": ["out batter"]
    },

    "appeal_type": {
        "description": "Type of appeal",
        "datatype": "TEXT",
        "category": "dismissal",
        "aggregation": "groupable",
        "synonyms": ["appeal"]
    },

    # =====================================================
    # TEAM STATS
    # =====================================================

    "batting_team": {
        "description": "Batting side",
        "datatype": "TEXT",
        "category": "team",
        "aggregation": "groupable",
        "synonyms": ["team batting"]
    },

    "bowling_team": {
        "description": "Bowling side",
        "datatype": "TEXT",
        "category": "team",
        "aggregation": "groupable",
        "synonyms": ["team bowling"]
    },

    "team_runs": {
        "description": "Team total runs",
        "datatype": "INTEGER",
        "category": "team",
        "aggregation": "max",
        "synonyms": ["score"]
    },

    "team_wickets": {
        "description": "Team wickets fallen",
        "datatype": "INTEGER",
        "category": "team",
        "aggregation": "max",
        "synonyms": ["wickets"]
    },

    "run_rate_at_start": {
        "description": "Run rate before ball",
        "datatype": "NUMERIC",
        "category": "team",
        "aggregation": "averageable",
        "synonyms": ["current run rate"]
    },

    "run_rate_after": {
        "description": "Run rate after ball",
        "datatype": "NUMERIC",
        "category": "team",
        "aggregation": "averageable",
        "synonyms": ["updated run rate"]
    },

    "req_run_rate_at_start": {
        "description": "Required run rate before delivery",
        "datatype": "NUMERIC",
        "category": "team",
        "aggregation": "averageable",
        "synonyms": ["required rate"]
    },

    "req_run_rate_after": {
        "description": "Required run rate after delivery",
        "datatype": "NUMERIC",
        "category": "team",
        "aggregation": "averageable",
        "synonyms": ["required run rate"]
    },

    # =====================================================
    # FIELDING
    # =====================================================

    "fielder1": {
        "description": "Primary fielder involved",
        "datatype": "TEXT",
        "category": "fielding",
        "aggregation": "groupable",
        "synonyms": ["fielder"]
    },

    "fielder1_position": {
        "description": "Primary fielder position",
        "datatype": "TEXT",
        "category": "fielding",
        "aggregation": "groupable",
        "synonyms": ["field position"]
    },

    "keeper_up": {
        "description": "Whether wicketkeeper stood up",
        "datatype": "BOOLEAN",
        "category": "fielding",
        "aggregation": "countable",
        "synonyms": ["keeper standing up"]
    },

    "around_the_wicket": {
        "description": "Bowler around the wicket",
        "datatype": "BOOLEAN",
        "category": "bowling",
        "aggregation": "countable",
        "synonyms": ["around wicket"]
    },

    # =====================================================
    # BALL TRACKING
    # =====================================================

    "pitchX": {
        "description": "Pitch impact X coordinate",
        "datatype": "NUMERIC",
        "category": "tracking",
        "aggregation": "averageable",
        "synonyms": ["pitch line"]
    },

    "pitchY": {
        "description": "Pitch impact Y coordinate",
        "datatype": "NUMERIC",
        "category": "tracking",
        "aggregation": "averageable",
        "synonyms": ["pitch length"]
    },

    "impactX": {
        "description": "Bat impact X coordinate",
        "datatype": "NUMERIC",
        "category": "tracking",
        "aggregation": "averageable",
        "synonyms": ["bat impact"]
    },

    "impactY": {
        "description": "Bat impact Y coordinate",
        "datatype": "NUMERIC",
        "category": "tracking",
        "aggregation": "averageable",
        "synonyms": ["impact position"]
    },

    "landingX": {
        "description": "Ball landing X coordinate",
        "datatype": "NUMERIC",
        "category": "tracking",
        "aggregation": "averageable",
        "synonyms": ["landing line"]
    },

    "landingY": {
        "description": "Ball landing Y coordinate",
        "datatype": "NUMERIC",
        "category": "tracking",
        "aggregation": "averageable",
        "synonyms": ["landing length"]
    },

    # =====================================================
    # MATCH RESULT
    # =====================================================

    "result": {
        "description": "Match result",
        "datatype": "TEXT",
        "category": "result",
        "aggregation": "groupable",
        "synonyms": ["outcome"]
    },

    "winning_team": {
        "description": "Winning team",
        "datatype": "TEXT",
        "category": "result",
        "aggregation": "groupable",
        "synonyms": ["winner"]
    },

    "losing_team": {
        "description": "Losing team",
        "datatype": "TEXT",
        "category": "result",
        "aggregation": "groupable",
        "synonyms": ["loser"]
    },

    # =====================================================
    # BALL DETAILS
    # =====================================================

    "legal_ball": {
        "description": "Whether delivery was legal",
        "datatype": "BOOLEAN",
        "category": "ball",
        "aggregation": "countable",
        "synonyms": ["valid delivery"]
    },

    "free_hit": {
        "description": "Whether delivery was free hit",
        "datatype": "BOOLEAN",
        "category": "ball",
        "aggregation": "countable",
        "synonyms": ["free delivery"]
    },

    "ball_type": {
        "description": "Type of cricket ball",
        "datatype": "TEXT",
        "category": "ball",
        "aggregation": "groupable",
        "synonyms": ["ball variation"]
    },

    "ball_colour": {
        "description": "Color of cricket ball",
        "datatype": "TEXT",
        "category": "ball",
        "aggregation": "groupable",
        "synonyms": ["ball color"]
    }

}