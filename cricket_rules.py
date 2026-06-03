# =====================================================
# CRICKET RULES DICTIONARY
# Complete context for Cricket_Scorer_AI
# Used to enrich SQL generation and insight quality
# =====================================================

CRICKET_RULES = {

    # =================================================
    # PLAYER ROLE PRIORITIES
    # =================================================

    "metric_priorities": {

        "strike_rate": [
            "batter",
            "all_rounder",
            "bowler"
        ],

        "batting_average": [
            "batter",
            "all_rounder",
            "bowler"
        ],

        "boundary_percentage": [
            "batter",
            "all_rounder",
            "bowler"
        ],

        "economy_rate": [
            "bowler",
            "all_rounder",
            "batter"
        ],

        "bowling_average": [
            "bowler",
            "all_rounder",
            "batter"
        ],

        "dot_ball_percentage": [
            "bowler",
            "all_rounder",
            "batter"
        ],

        "wickets": [
            "bowler",
            "all_rounder",
            "batter"
        ],

        "partnerships": [
            "batter",
            "all_rounder"
        ],

        "fielding_impact": [
            "fielder",
            "all_rounder",
            "bowler",
            "batter"
        ]
    },

    # =================================================
    # MINIMUM SAMPLE SIZE
    # Prevents stats being drawn from too few balls
    # =================================================

    "minimum_sample_size": {

        "batting": {
            "balls_faced": 30,
            "innings": 3
        },

        "bowling": {
            "balls_bowled": 24,
            "innings": 3
        },

        "phase_analysis": {
            "balls": 20
        },

        "head_to_head": {
            "balls": 12
        },

        "partnership": {
            "balls_together": 20
        },

        "venue_analysis": {
            "matches": 3
        },

        "opposition_analysis": {
            "matches": 2
        }
    },

    # =================================================
    # MATCH FORMATS
    # =================================================

    "match_formats": {

        "T10": {
            "max_overs": 10,
            "innings_per_match": 2,
            "typical_score_range": "60-120"
        },

        "T20": {
            "max_overs": 20,
            "innings_per_match": 2,
            "typical_score_range": "140-200"
        },

        "ODI": {
            "max_overs": 50,
            "innings_per_match": 2,
            "typical_score_range": "250-320"
        },

        "TEST": {
            "max_overs": None,
            "innings_per_match": 4,
            "typical_score_range": "200-500"
        },

        "FIRST_CLASS": {
            "max_overs": None,
            "innings_per_match": 4,
            "typical_score_range": "200-500"
        },

        "THE_HUNDRED": {
            "max_balls": 100,
            "innings_per_match": 2,
            "typical_score_range": "120-160"
        }
    },

    # =================================================
    # MATCH PHASES
    # Over ranges for each format
    # =================================================

    "match_phases": {

        "T10": {
            "powerplay": {
                "start_over": 1,
                "end_over": 2,
                "description": "Fielding restrictions — 2 fielders outside circle"
            },
            "middle_overs": {
                "start_over": 3,
                "end_over": 7,
                "description": "Building phase"
            },
            "death_overs": {
                "start_over": 8,
                "end_over": 10,
                "description": "Final slog — maximum aggression"
            }
        },

        "T20": {
            "powerplay": {
                "start_over": 1,
                "end_over": 6,
                "description": "Fielding restrictions — 2 fielders outside circle. Key phase for early wickets or runs"
            },
            "middle_overs": {
                "start_over": 7,
                "end_over": 15,
                "description": "Consolidation and acceleration. Spinners dominate"
            },
            "death_overs": {
                "start_over": 16,
                "end_over": 20,
                "description": "Maximum aggression. Yorkers, bouncers, slog sweeps"
            }
        },

        "ODI": {
            "powerplay_1": {
                "start_over": 1,
                "end_over": 10,
                "description": "Mandatory powerplay — 2 fielders outside circle"
            },
            "powerplay_2": {
                "start_over": 11,
                "end_over": 40,
                "description": "Middle overs — 4 fielders outside circle"
            },
            "death_overs": {
                "start_over": 41,
                "end_over": 50,
                "description": "Final powerplay — 4 fielders outside circle"
            }
        },

        "TEST": {
            "new_ball_phase": {
                "start_over": 1,
                "end_over": 20,
                "description": "New ball swings and seams. Most dangerous for batters"
            },
            "middle_session": {
                "start_over": 21,
                "end_over": 60,
                "description": "Ball loses shine. Spinners come in. Consolidation"
            },
            "old_ball_phase": {
                "start_over": 61,
                "end_over": 80,
                "description": "Reverse swing potential. Batting becomes easier"
            },
            "second_new_ball_phase": {
                "start_over": 81,
                "end_over": 120,
                "description": "Second new ball after 80 overs. Danger phase again"
            }
        }
    },

    # =================================================
    # BATTING BENCHMARKS
    # Used to classify player performance quality
    # =================================================

    "batting_benchmarks": {

        "T20": {
            "strike_rate": {
                "poor":      {"max": 110, "label": "Slow — struggles to keep up with match tempo"},
                "average":   {"min": 110, "max": 124, "label": "Acceptable but below par"},
                "good":      {"min": 125, "max": 139, "label": "Solid T20 batter"},
                "excellent": {"min": 140, "max": 159, "label": "High-class T20 batter"},
                "elite":     {"min": 160, "label": "Match-winning, explosive batter"}
            },
            "batting_average": {
                "poor":      {"max": 15,  "label": "Inconsistent"},
                "average":   {"min": 15,  "max": 24, "label": "Moderate"},
                "good":      {"min": 25,  "max": 34, "label": "Reliable"},
                "excellent": {"min": 35,  "max": 44, "label": "Very consistent"},
                "elite":     {"min": 45,  "label": "World-class consistency"}
            },
            "boundary_percentage": {
                "low":    {"max": 40,  "label": "Too much running, struggles to find boundaries"},
                "medium": {"min": 40,  "max": 54, "label": "Balanced"},
                "high":   {"min": 55,  "max": 64, "label": "Strong boundary hitter"},
                "elite":  {"min": 65,  "label": "Dominant boundary scorer"}
            }
        },

        "ODI": {
            "strike_rate": {
                "poor":      {"max": 75,  "label": "Too slow for modern ODI cricket"},
                "average":   {"min": 75,  "max": 84,  "label": "Acceptable in anchor role"},
                "good":      {"min": 85,  "max": 94,  "label": "Good ODI batter"},
                "excellent": {"min": 95,  "max": 109, "label": "High-quality ODI batter"},
                "elite":     {"min": 110, "label": "Explosive ODI batter"}
            },
            "batting_average": {
                "poor":      {"max": 25,  "label": "Below par"},
                "average":   {"min": 25,  "max": 34,  "label": "Decent"},
                "good":      {"min": 35,  "max": 44,  "label": "Good"},
                "excellent": {"min": 45,  "max": 54,  "label": "Very good"},
                "elite":     {"min": 55,  "label": "World-class"}
            }
        },

        "TEST": {
            "strike_rate": {
                "poor":      {"max": 35, "label": "Very slow, risks losing sessions"},
                "average":   {"min": 35, "max": 44, "label": "Conservative Test batter"},
                "good":      {"min": 45, "max": 54, "label": "Balanced attacker"},
                "excellent": {"min": 55, "max": 64, "label": "Positive Test batter"},
                "elite":     {"min": 65, "label": "Dominant, aggressive Test batter"}
            },
            "batting_average": {
                "poor":      {"max": 25, "label": "Struggles at Test level"},
                "average":   {"min": 25, "max": 34, "label": "Solid Test player"},
                "good":      {"min": 35, "max": 44, "label": "Quality Test batter"},
                "excellent": {"min": 45, "max": 54, "label": "Very reliable"},
                "elite":     {"min": 55, "label": "All-time great territory"}
            }
        }
    },

    # =================================================
    # BOWLING BENCHMARKS
    # Used to classify bowler performance quality
    # =================================================

    "bowling_benchmarks": {

        "T20": {
            "economy_rate": {
                "elite":     {"max": 6.5,  "label": "Exceptional — match-winning spell"},
                "excellent": {"min": 6.5,  "max": 7.49, "label": "Very hard to score off"},
                "good":      {"min": 7.5,  "max": 8.49, "label": "Effective T20 bowler"},
                "average":   {"min": 8.5,  "max": 9.49, "label": "Manageable"},
                "poor":      {"min": 9.5,  "label": "Expensive — batters dominate"}
            },
            "bowling_average": {
                "elite":     {"max": 18,  "label": "Taking wickets very cheaply"},
                "excellent": {"min": 18,  "max": 22,  "label": "High-quality wicket taker"},
                "good":      {"min": 23,  "max": 27,  "label": "Decent wicket taker"},
                "average":   {"min": 28,  "max": 32,  "label": "Average"},
                "poor":      {"min": 33,  "label": "Wickets too expensive"}
            },
            "dot_ball_percentage": {
                "poor":      {"max": 30,  "label": "Not creating enough pressure"},
                "average":   {"min": 30,  "max": 39,  "label": "Moderate pressure"},
                "good":      {"min": 40,  "max": 49,  "label": "Good pressure bowler"},
                "excellent": {"min": 50,  "max": 59,  "label": "Very high pressure"},
                "elite":     {"min": 60,  "label": "Dominates batters with dot balls"}
            }
        },

        "ODI": {
            "economy_rate": {
                "elite":     {"max": 4.0,  "label": "Exceptional control"},
                "excellent": {"min": 4.0,  "max": 4.79, "label": "Very economical"},
                "good":      {"min": 4.8,  "max": 5.49, "label": "Solid ODI bowler"},
                "average":   {"min": 5.5,  "max": 5.99, "label": "Acceptable"},
                "poor":      {"min": 6.0,  "label": "Expensive in ODI context"}
            },
            "bowling_average": {
                "elite":     {"max": 22,  "label": "World-class"},
                "excellent": {"min": 22,  "max": 27,  "label": "Very good"},
                "good":      {"min": 28,  "max": 32,  "label": "Solid"},
                "average":   {"min": 33,  "max": 37,  "label": "Average"},
                "poor":      {"min": 38,  "label": "Below par"}
            }
        },

        "TEST": {
            "economy_rate": {
                "elite":     {"max": 2.0,  "label": "Miserly — wins sessions single-handedly"},
                "excellent": {"min": 2.0,  "max": 2.49, "label": "Excellent control"},
                "good":      {"min": 2.5,  "max": 2.99, "label": "Good"},
                "average":   {"min": 3.0,  "max": 3.49, "label": "Acceptable in Tests"},
                "poor":      {"min": 3.5,  "label": "Too easy to score off in Tests"}
            },
            "bowling_average": {
                "elite":     {"max": 20,  "label": "All-time great territory"},
                "excellent": {"min": 20,  "max": 25,  "label": "World-class Test bowler"},
                "good":      {"min": 26,  "max": 30,  "label": "Quality Test bowler"},
                "average":   {"min": 31,  "max": 35,  "label": "Decent"},
                "poor":      {"min": 36,  "label": "Wickets too expensive for Tests"}
            }
        }
    },

    # =================================================
    # DISMISSAL TYPES
    # Context for analysis and tactical insights
    # =================================================

    "dismissal_types": {

        "Caught": {
            "bowler_credit": True,
            "fielder_credit": True,
            "tactical_note": "Indicates aerial hitting or edges — bowler creating false shots"
        },

        "Bowled": {
            "bowler_credit": True,
            "fielder_credit": False,
            "tactical_note": "Clean bowled — bowler beat the batter completely. Indicates great accuracy or movement"
        },

        "LBW": {
            "bowler_credit": True,
            "fielder_credit": False,
            "tactical_note": "Ball would hit stumps — bowler hitting the pads. Indicates good line"
        },

        "Run Out": {
            "bowler_credit": False,
            "fielder_credit": True,
            "tactical_note": "Fielding pressure or poor running between wickets"
        },

        "Stumped": {
            "bowler_credit": True,
            "fielder_credit": True,
            "tactical_note": "Batter played for turn that wasn't there — spinner or pace off delivery"
        },

        "Hit Wicket": {
            "bowler_credit": True,
            "fielder_credit": False,
            "tactical_note": "Batter dislodged own stumps — rare. Usually on pull shot or ducking"
        },

        "Handled Ball": {
            "bowler_credit": False,
            "fielder_credit": False,
            "tactical_note": "Obstructing the field or handling the ball — extremely rare"
        },

        "Retired Out": {
            "bowler_credit": False,
            "fielder_credit": False,
            "tactical_note": "Batter chose to retire — sometimes tactical in shorter formats"
        }
    },

    # =================================================
    # BOWLING TYPE CONTEXT
    # Used for tactical insight generation
    # =================================================

    "bowling_types": {

        "pace": {
            "sub_types": ["Fast", "Fast-Medium", "Medium-Fast", "Medium"],
            "key_metrics": ["economy_rate", "dot_ball_percentage", "bowling_average"],
            "tactical_weapons": ["yorker", "bouncer", "outswing", "inswing", "reverse swing"],
            "vulnerable_phase": "powerplay and death overs"
        },

        "spin": {
            "sub_types": ["Off-spin", "Leg-spin", "Left-arm orthodox", "Left-arm chinaman"],
            "key_metrics": ["economy_rate", "wickets", "dot_ball_percentage"],
            "tactical_weapons": ["turn", "flight", "googly", "doosra", "arm ball"],
            "vulnerable_phase": "middle overs"
        }
    },

    # =================================================
    # FIELDING POSITIONS
    # Context for caught dismissal analysis
    # =================================================

    "fielding_positions": {

        "catching_positions": [
            "slip",
            "gully",
            "point",
            "cover",
            "mid-off",
            "mid-on",
            "square leg",
            "fine leg",
            "third man",
            "long-on",
            "long-off",
            "deep square leg",
            "deep mid-wicket"
        ],

        "pressure_positions": [
            "slip",
            "gully",
            "short leg",
            "silly mid-on",
            "silly mid-off"
        ]
    },

    # =================================================
    # INNINGS CONTEXT RULES
    # Always include these in analysis
    # =================================================

    "innings_rules": {

        "always_include_innings_played": True,
        "compare_innings_before_conclusions": True,
        "mention_runs_per_innings": True,
        "mention_wickets_per_innings": True,
        "mention_balls_per_innings": True,
        "note_not_out_innings": True,
        "separate_batting_bowling_innings": True
    },

    # =================================================
    # TEAM ANALYSIS RULES
    # =================================================

    "team_analysis": {

        "always_include": [
            "matches_played",
            "total_innings",
            "average_score",
            "run_rate",
            "highest_score",
            "lowest_score",
            "total_wickets_lost",
            "powerplay_average",
            "death_overs_average"
        ],

        "batting_team_context": [
            "Compare powerplay runs vs opposition average",
            "Identify strongest batting phase",
            "Note boundary hitting rate",
            "Note dot ball percentage conceded"
        ],

        "bowling_team_context": [
            "Compare economy to format benchmark",
            "Identify most economical phase",
            "Note wicket-taking frequency",
            "Note death overs economy"
        ]
    },

    # =================================================
    # VENUE / PITCH ANALYSIS
    # =================================================

    "venue_analysis": {

        "always_include": [
            "matches_played",
            "average_first_innings_score",
            "average_second_innings_score",
            "average_winning_score",
            "chasing_win_percentage"
        ],

        "pitch_types": {

            "batting_paradise": {
                "indicator": "avg_first_innings > 180 (T20)",
                "insight": "High-scoring ground — batters dominate. Spinners struggle"
            },

            "bowlers_pitch": {
                "indicator": "avg_first_innings < 140 (T20)",
                "insight": "Low-scoring — bowlers dominate. Pace and bounce key"
            },

            "spin_friendly": {
                "indicator": "spin_wicket_percentage > 50%",
                "insight": "Spinners dominate — turn and bounce off the surface"
            },

            "pace_friendly": {
                "indicator": "pace_wicket_percentage > 60%",
                "insight": "Seam and swing — new ball crucial"
            }
        }
    },

    # =================================================
    # PARTNERSHIP ANALYSIS
    # =================================================

    "partnership_rules": {

        "key_partnerships": [
            "Opening partnership (1st wicket)",
            "Middle order anchor (3rd-4th wicket)",
            "Late order slog (8th-10th wicket)"
        ],

        "always_include": [
            "runs_added",
            "balls_faced_together",
            "run_rate_during_partnership",
            "boundaries_hit_together"
        ],

        "context_notes": [
            "Opening partnerships set the tempo for entire innings",
            "Long partnerships indicate batter temperament and compatibility",
            "Late order partnerships can change match outcome significantly"
        ]
    },

    # =================================================
    # HEAD TO HEAD RULES
    # =================================================

    "head_to_head_rules": {

        "minimum_balls": 12,

        "always_include": [
            "balls_faced",
            "runs_scored",
            "times_dismissed",
            "dot_balls_received",
            "boundaries_hit",
            "strike_rate_in_matchup"
        ],

        "insight_notes": [
            "A batter dismissed 3+ times by same bowler indicates a clear weakness",
            "SR below 100 against a bowler in T20 means the bowler has the upper hand",
            "SR above 180 against a bowler means the batter dominates that matchup"
        ]
    },

    # =================================================
    # COMPARISON RULES
    # =================================================

    "comparison_rules": [

        "Always compare innings played before comparing totals",
        "Always compare balls faced before comparing strike rates",
        "Always compare balls bowled before comparing economy rates",
        "Use per innings metrics for fair comparison",
        "Use per match metrics for team comparisons",
        "Always mention sample size",
        "Do not rank players on fewer than minimum sample size",
        "Rank players by efficiency metrics, not just volume",
        "Mention format context when comparing across formats",
        "Note if one player batted in favorable conditions vs tough conditions"
    ],

    # =================================================
    # INSIGHT GENERATION RULES
    # These guide the commentary quality
    # =================================================

    "insight_rules": [

        "Always mention innings played and sample size",
        "Always classify performance against benchmark (poor/average/good/excellent/elite)",
        "Mention rank among peers when data allows",
        "Compare against format benchmark, not just dataset average",
        "Mention strengths with specific stat evidence",
        "Mention weaknesses with specific stat evidence",
        "Always include phase-wise performance if data has it",
        "Mention opposition quality if head-to-head data available",
        "Mention venue trends when data spans multiple grounds",
        "Identify patterns — e.g. flat track bully vs tough conditions performer",
        "Mention tactical implications for opposing captain",
        "Use cricket commentary language naturally",
        "Avoid generic statements — every claim must be backed by a number",
        "End with a definitive verdict — not a vague summary"
    ],

    # =================================================
    # SPECIAL MATCH SITUATIONS
    # =================================================

    "match_situations": {

        "powerplay_batting": {
            "expectation_T20": "40-60 runs with max 2 wickets",
            "insight": "Getting 2 wickets in powerplay gives bowling team significant advantage"
        },

        "death_batting": {
            "expectation_T20": "50-60 runs in last 5 overs",
            "insight": "Finishing ability separates good from great T20 teams"
        },

        "chase_pressure": {
            "insight": "Chasing teams need early wickets in powerplay to restrict scoring"
        },

        "spin_in_middle": {
            "insight": "Spinners bowling in overs 7-15 (T20) are key to containing run rate"
        },

        "super_over": {
            "insight": "Best striker and best death bowler are the defining selections"
        }
    },

    # =================================================
    # ALL ROUNDER CLASSIFICATION
    # =================================================

    "all_rounder_thresholds": {

        "T20": {
            "batting_minimum_runs": 200,
            "bowling_minimum_wickets": 10,
            "batting_sr_minimum": 120,
            "bowling_economy_maximum": 9.0
        },

        "ODI": {
            "batting_minimum_runs": 500,
            "bowling_minimum_wickets": 20,
            "batting_average_minimum": 25,
            "bowling_economy_maximum": 5.5
        },

        "TEST": {
            "batting_minimum_runs": 1000,
            "bowling_minimum_wickets": 50,
            "batting_average_minimum": 30,
            "bowling_average_maximum": 35
        }
    },

    # =================================================
    # FREE HIT RULES
    # =================================================

    "free_hit_rules": {

        "triggered_by": "No-ball (over-stepping)",
        "batter_advantage": "Cannot be dismissed except run out",
        "expected_outcome": "Batter should score 6, 4, or minimum 2",
        "insight": "Free hit dot balls or wickets (via run out) represent missed opportunities",
        "analysis_note": "Track free_hit = TRUE balls for batting aggression analysis"
    },

    # =================================================
    # KEEPER POSITION IMPACT
    # =================================================

    "keeper_position_rules": {

        "keeper_up": {
            "meaning": "Wicketkeeper standing up to stumps",
            "impact": "Limits batter's ability to leave crease — increases stumping risk",
            "common_for": "Spin bowling, medium pace on slow pitches"
        },

        "keeper_back": {
            "meaning": "Wicketkeeper standing back",
            "impact": "More ground to cover — batter can run out faster",
            "common_for": "Fast bowling"
        },

        "analysis_note": "Compare batter SR and dismissal rate when keeper_up vs keeper_back"
    },

    # =================================================
    # AROUND / OVER THE WICKET RULES
    # =================================================

    "bowling_angle_rules": {

        "around_the_wicket": {
            "meaning": "Bowler delivering from the non-dominant side",
            "right_arm_to_right_batter": "Angles into the body — LBW threat",
            "right_arm_to_left_batter": "Angles away — edge to slip risk",
            "insight": "Analyse economy and wicket rate around vs over wicket"
        },

        "over_the_wicket": {
            "meaning": "Standard delivery angle",
            "insight": "Compare dismissal type distribution for each angle"
        }
    },

    # =================================================
    # LEGAL BALL RULES
    # =================================================

    "legal_ball_rules": {

        "legal_ball_true": "Counts toward over. Use for SR, economy, dot% calculations",
        "legal_ball_false": "Wide or no-ball. Gives batting team free runs",
        "analysis_note": "Always filter legal_ball = TRUE for strike rate and economy calculations",
        "extra_types": ["wide", "no_ball", "bye", "leg_bye"],
        "insight": "High no-ball / wide count indicates discipline problems in bowling attack"
    },

    # =================================================
    # SCORING ZONES (BATTING SHOT CLASSIFICATION)
    # =================================================

    "scoring_zone_context": {

        "boundary_4": {
            "meaning": "Ball reaches boundary — 4 runs",
            "insight": "High 4s count = gap-finding ability and placement"
        },

        "boundary_6": {
            "meaning": "Ball clears boundary — 6 runs",
            "insight": "High 6s count = power hitting and aerial game"
        },

        "dot_ball": {
            "meaning": "0 runs scored off delivery",
            "insight": "High dot ball % for batter = under pressure or conservative. High dot% for bowler = excellent"
        },

        "singles_and_twos": {
            "meaning": "Running between wickets",
            "insight": "High proportion of 1s and 2s suggests strike rotation ability but may indicate boundary-hitting difficulty"
        }
    },

    # =================================================
    # CONTEXT LABELS FOR INSIGHT OUTPUT
    # Labels used by AI to describe performance tier
    # =================================================

    "performance_labels": {

        "batting": {
            "elite":     "🔴 Elite — world-class performer",
            "excellent": "🟠 Excellent — high-quality player",
            "good":      "🟡 Good — solid contributor",
            "average":   "🟢 Average — room for improvement",
            "poor":      "⚪ Below Par — significant weaknesses"
        },

        "bowling": {
            "elite":     "🔴 Elite — match-winner",
            "excellent": "🟠 Excellent — very difficult to score off",
            "good":      "🟡 Good — reliable bowler",
            "average":   "🟢 Average — can be targeted",
            "poor":      "⚪ Below Par — expensive and ineffective"
        }
    }
}