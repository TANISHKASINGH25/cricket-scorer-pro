# =====================================================
# CRICKET RULES DICTIONARY
# Complete context for Cricket_Scorer_AI
# Enhanced for complex query accuracy and tactical depth
# =====================================================

CRICKET_RULES = {

    # =================================================
    # PLAYER ROLE PRIORITIES
    # =================================================

    "metric_priorities": {

        "strike_rate": ["batter", "all_rounder", "bowler"],
        "batting_average": ["batter", "all_rounder", "bowler"],
        "boundary_percentage": ["batter", "all_rounder", "bowler"],
        "economy_rate": ["bowler", "all_rounder", "batter"],
        "bowling_average": ["bowler", "all_rounder", "batter"],
        "dot_ball_percentage": ["bowler", "all_rounder", "batter"],
        "wickets": ["bowler", "all_rounder", "batter"],
        "partnerships": ["batter", "all_rounder"],
        "fielding_impact": ["fielder", "all_rounder", "bowler", "batter"]
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
        },

        "recent_form": {
            "matches": 5,
            "note": "Use last 5-10 matches for form analysis; flag if fewer available"
        },

        "consistency_analysis": {
            "innings": 10,
            "note": "Need at least 10 innings to meaningfully assess consistency"
        },

        "over_analysis": {
            "overs_bowled": 10,
            "note": "Need at least 10 overs bowled in that over-number to spot patterns"
        }
    },

    # =================================================
    # MATCH FORMATS
    # =================================================

    "match_formats": {

        "T10": {
            "max_overs": 10,
            "innings_per_match": 2,
            "typical_score_range": "60-120",
            "powerplay_overs": "1-2",
            "death_overs": "8-10"
        },

        "T20": {
            "max_overs": 20,
            "innings_per_match": 2,
            "typical_score_range": "140-200",
            "powerplay_overs": "1-6",
            "middle_overs": "7-15",
            "death_overs": "16-20",
            "par_score_note": "150 is considered par; 170+ is a strong total; 200+ is exceptional"
        },

        "ODI": {
            "max_overs": 50,
            "innings_per_match": 2,
            "typical_score_range": "250-320",
            "powerplay_overs": "1-10",
            "middle_overs": "11-40",
            "death_overs": "41-50",
            "par_score_note": "270 is par; 300+ is competitive; 330+ is dominant"
        },

        "TEST": {
            "max_overs": None,
            "innings_per_match": 4,
            "typical_score_range": "200-500",
            "new_ball_overs": "1-20",
            "middle_session": "21-60",
            "old_ball": "61-80",
            "second_new_ball": "81+",
            "par_score_note": "350+ gives a team a strong first-innings platform"
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
    # Over ranges with tactical detail per format
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
                "description": "Fielding restrictions — only 2 fielders outside circle. High-risk phase: both teams want control. Typical target: 45-55 runs, max 2 wickets lost.",
                "batting_target_runs": "45-55",
                "batting_target_wickets_lost": 2,
                "bowling_target_wickets": 2,
                "bowling_target_economy": 7.5,
                "key_roles": "Openers, new-ball bowlers"
            },
            "middle_overs": {
                "start_over": 7,
                "end_over": 15,
                "description": "Consolidation and acceleration. Spinners dominate. Set batters should push SR above 120. Bowlers aim for dots and breakthroughs.",
                "batting_target_sr": 120,
                "bowling_target_economy": 8.0,
                "key_roles": "Spinners, anchor batters, power hitters entering"
            },
            "death_overs": {
                "start_over": 16,
                "end_over": 20,
                "description": "Maximum aggression. Yorkers, bouncers, slog sweeps. Teams target 55-65 runs in last 5 overs.",
                "batting_target_runs_last5": "55-65",
                "bowling_target_economy": 9.0,
                "key_roles": "Finishers (batting positions 5-7), death specialists"
            }
        },

        "ODI": {
            "powerplay_1": {
                "start_over": 1,
                "end_over": 10,
                "description": "Mandatory powerplay — 2 fielders outside circle. Teams target 50-60 runs.",
                "batting_target_runs": "50-60",
                "batting_target_wickets_lost": 2
            },
            "middle_overs": {
                "start_over": 11,
                "end_over": 40,
                "description": "Middle overs — 4 fielders outside circle. Rotate strike, build platform. Key for all-rounders.",
                "batting_target_sr": 85,
                "bowling_target_economy": 5.5
            },
            "death_overs": {
                "start_over": 41,
                "end_over": 50,
                "description": "Final push — teams target 80-100 runs in last 10 overs. Yorkers and variations crucial.",
                "batting_target_runs_last10": "80-100",
                "bowling_target_economy": 7.0
            }
        },

        "TEST": {
            "new_ball_phase": {
                "start_over": 1,
                "end_over": 20,
                "description": "New ball swings and seams. Most dangerous for batters. Survival is key.",
                "key_roles": "Openers, new-ball bowlers"
            },
            "middle_session": {
                "start_over": 21,
                "end_over": 60,
                "description": "Ball loses shine. Spinners come in. Consolidation and building big partnerships.",
                "key_roles": "Middle-order batters, spinners"
            },
            "old_ball_phase": {
                "start_over": 61,
                "end_over": 80,
                "description": "Reverse swing potential. Batting becomes relatively easier. Build total.",
                "key_roles": "Lower-middle order, reverse swing specialists"
            },
            "second_new_ball_phase": {
                "start_over": 81,
                "end_over": 120,
                "description": "Second new ball after 80 overs. Danger phase resumes. Tail at risk.",
                "key_roles": "New-ball bowlers, tail batters"
            }
        }
    },

    # =================================================
    # BATTING BENCHMARKS
    # Per format, per metric, with tier labels
    # =================================================

    "batting_benchmarks": {

        "T20": {
            "strike_rate": {
                "poor":      {"max": 110,               "label": "Slow — struggles to keep up with match tempo"},
                "average":   {"min": 110, "max": 124,   "label": "Acceptable but below par for T20"},
                "good":      {"min": 125, "max": 139,   "label": "Solid T20 batter"},
                "excellent": {"min": 140, "max": 159,   "label": "High-class T20 batter"},
                "elite":     {"min": 160,               "label": "Match-winning, explosive batter"}
            },
            "batting_average": {
                "poor":      {"max": 15,                "label": "Inconsistent — frequent failures"},
                "average":   {"min": 15,  "max": 24,   "label": "Moderate — useful but unreliable"},
                "good":      {"min": 25,  "max": 34,   "label": "Reliable contributor"},
                "excellent": {"min": 35,  "max": 44,   "label": "Very consistent"},
                "elite":     {"min": 45,               "label": "World-class consistency"}
            },
            "boundary_percentage": {
                "low":    {"max": 39,                   "label": "Too much running; struggles to find boundaries"},
                "medium": {"min": 40,  "max": 54,      "label": "Balanced — mixed boundaries and rotation"},
                "high":   {"min": 55,  "max": 64,      "label": "Strong boundary hitter"},
                "elite":  {"min": 65,                   "label": "Dominant boundary scorer"}
            },
            "dot_ball_percentage_batting": {
                "elite":     {"max": 25,                "label": "Exceptional — barely wastes a ball"},
                "excellent": {"min": 25, "max": 34,    "label": "Very active — keeps scoreboard ticking"},
                "good":      {"min": 35, "max": 44,    "label": "Acceptable rotation"},
                "average":   {"min": 45, "max": 54,    "label": "Too many dots — building pressure on self"},
                "poor":      {"min": 55,               "label": "Dot-heavy — under serious pressure"}
            },
            "powerplay_sr": {
                "poor":      {"max": 110,               "label": "Too slow in powerplay — wasting field restrictions"},
                "average":   {"min": 110, "max": 129,  "label": "Below par powerplay aggression"},
                "good":      {"min": 130, "max": 149,  "label": "Good powerplay batter"},
                "excellent": {"min": 150, "max": 169,  "label": "Very aggressive powerplay opener"},
                "elite":     {"min": 170,               "label": "Dominates with field restrictions"}
            },
            "death_sr": {
                "poor":      {"max": 120,               "label": "Cannot finish — team loses momentum"},
                "average":   {"min": 120, "max": 149,  "label": "Decent finisher"},
                "good":      {"min": 150, "max": 174,  "label": "Reliable death batter"},
                "excellent": {"min": 175, "max": 199,  "label": "Explosive finisher"},
                "elite":     {"min": 200,               "label": "World-class — match-winning in death overs"}
            }
        },

        "ODI": {
            "strike_rate": {
                "poor":      {"max": 75,               "label": "Too slow for modern ODI cricket"},
                "average":   {"min": 75,  "max": 84,  "label": "Acceptable in anchor role"},
                "good":      {"min": 85,  "max": 94,  "label": "Good ODI batter"},
                "excellent": {"min": 95,  "max": 109, "label": "High-quality ODI batter"},
                "elite":     {"min": 110,              "label": "Explosive ODI batter"}
            },
            "batting_average": {
                "poor":      {"max": 25,               "label": "Below par"},
                "average":   {"min": 25,  "max": 34,  "label": "Decent"},
                "good":      {"min": 35,  "max": 44,  "label": "Good"},
                "excellent": {"min": 45,  "max": 54,  "label": "Very good"},
                "elite":     {"min": 55,               "label": "World-class"}
            }
        },

        "TEST": {
            "strike_rate": {
                "poor":      {"max": 35,               "label": "Very slow — risks losing sessions"},
                "average":   {"min": 35, "max": 44,   "label": "Conservative Test batter"},
                "good":      {"min": 45, "max": 54,   "label": "Balanced attacker"},
                "excellent": {"min": 55, "max": 64,   "label": "Positive, proactive Test batter"},
                "elite":     {"min": 65,               "label": "Dominant, aggressive Test batter"}
            },
            "batting_average": {
                "poor":      {"max": 25,               "label": "Struggles at Test level"},
                "average":   {"min": 25, "max": 34,   "label": "Solid Test player"},
                "good":      {"min": 35, "max": 44,   "label": "Quality Test batter"},
                "excellent": {"min": 45, "max": 54,   "label": "Very reliable — backbone of lineup"},
                "elite":     {"min": 55,               "label": "All-time great territory"}
            }
        }
    },

    # =================================================
    # BOWLING BENCHMARKS
    # Per format, per metric, with tier labels
    # =================================================

    "bowling_benchmarks": {

        "T20": {
            "economy_rate": {
                "elite":     {"max": 6.5,              "label": "Exceptional — match-winning economy"},
                "excellent": {"min": 6.5,  "max": 7.49, "label": "Very hard to score off"},
                "good":      {"min": 7.5,  "max": 8.49, "label": "Effective T20 bowler"},
                "average":   {"min": 8.5,  "max": 9.49, "label": "Manageable — can be improved"},
                "poor":      {"min": 9.5,              "label": "Expensive — batters dominate"}
            },
            "bowling_average": {
                "elite":     {"max": 18,               "label": "Taking wickets very cheaply"},
                "excellent": {"min": 18,  "max": 22,  "label": "High-quality wicket taker"},
                "good":      {"min": 23,  "max": 27,  "label": "Decent wicket taker"},
                "average":   {"min": 28,  "max": 32,  "label": "Average"},
                "poor":      {"min": 33,               "label": "Wickets too expensive"}
            },
            "dot_ball_percentage": {
                "poor":      {"max": 30,               "label": "Not creating enough pressure"},
                "average":   {"min": 30,  "max": 39,  "label": "Moderate pressure"},
                "good":      {"min": 40,  "max": 49,  "label": "Good pressure bowler"},
                "excellent": {"min": 50,  "max": 59,  "label": "Very high dot-ball pressure"},
                "elite":     {"min": 60,               "label": "Dominates — batters can't score"}
            },
            "bowling_strike_rate": {
                "elite":     {"max": 12,               "label": "Taking a wicket every 2 overs — elite"},
                "excellent": {"min": 12,  "max": 15,  "label": "Frequent wicket taker"},
                "good":      {"min": 16,  "max": 19,  "label": "Wickets at good frequency"},
                "average":   {"min": 20,  "max": 24,  "label": "Wickets at average frequency"},
                "poor":      {"min": 25,               "label": "Infrequent wicket taker"}
            },
            "powerplay_economy": {
                "elite":     {"max": 6.0,              "label": "Exceptional new-ball control"},
                "excellent": {"min": 6.0,  "max": 7.0, "label": "Very good opening bowler"},
                "good":      {"min": 7.0,  "max": 8.0, "label": "Acceptable powerplay"},
                "average":   {"min": 8.0,  "max": 9.0, "label": "Slightly expensive in PP"},
                "poor":      {"min": 9.0,              "label": "Gets hit in powerplay"}
            },
            "death_economy": {
                "elite":     {"max": 7.5,              "label": "World-class death bowler"},
                "excellent": {"min": 7.5,  "max": 8.99, "label": "Very good death bowler"},
                "good":      {"min": 9.0,  "max": 10.49, "label": "Decent death options"},
                "average":   {"min": 10.5, "max": 11.99, "label": "Expensive at death"},
                "poor":      {"min": 12.0,             "label": "Targeted — cannot be trusted in death"}
            }
        },

        "ODI": {
            "economy_rate": {
                "elite":     {"max": 4.0,              "label": "Exceptional control"},
                "excellent": {"min": 4.0,  "max": 4.79, "label": "Very economical"},
                "good":      {"min": 4.8,  "max": 5.49, "label": "Solid ODI bowler"},
                "average":   {"min": 5.5,  "max": 5.99, "label": "Acceptable"},
                "poor":      {"min": 6.0,              "label": "Expensive in ODI context"}
            },
            "bowling_average": {
                "elite":     {"max": 22,               "label": "World-class"},
                "excellent": {"min": 22,  "max": 27,  "label": "Very good"},
                "good":      {"min": 28,  "max": 32,  "label": "Solid"},
                "average":   {"min": 33,  "max": 37,  "label": "Average"},
                "poor":      {"min": 38,               "label": "Below par"}
            }
        },

        "TEST": {
            "economy_rate": {
                "elite":     {"max": 2.0,              "label": "Miserly — wins sessions single-handedly"},
                "excellent": {"min": 2.0,  "max": 2.49, "label": "Excellent control"},
                "good":      {"min": 2.5,  "max": 2.99, "label": "Good"},
                "average":   {"min": 3.0,  "max": 3.49, "label": "Acceptable in Tests"},
                "poor":      {"min": 3.5,              "label": "Too easy to score off in Tests"}
            },
            "bowling_average": {
                "elite":     {"max": 20,               "label": "All-time great territory"},
                "excellent": {"min": 20,  "max": 25,  "label": "World-class Test bowler"},
                "good":      {"min": 26,  "max": 30,  "label": "Quality Test bowler"},
                "average":   {"min": 31,  "max": 35,  "label": "Decent"},
                "poor":      {"min": 36,               "label": "Wickets too expensive for Tests"}
            }
        }
    },

    # =================================================
    # DISMISSAL TYPES
    # Tactical context and pattern insights
    # =================================================

    "dismissal_types": {

        "Caught": {
            "bowler_credit": True,
            "fielder_credit": True,
            "tactical_note": "Indicates aerial hitting, top-edge, or edged drive. Bowler creating false shots or drawing the drive",
            "common_causes": ["short ball inducing pull/hook", "full delivery drawing drive", "pace off tempting loft", "spin with sharp turn"],
            "pattern_insight": "High caught% against a specific bowler suggests that bowler exploits an aerial weakness"
        },

        "Bowled": {
            "bowler_credit": True,
            "fielder_credit": False,
            "tactical_note": "Beaten completely — yorker, sharp inswing, or ball that stays low",
            "common_causes": ["yorker beating the bat", "inswing late movement", "ball keeping low", "batter playing across the line"],
            "pattern_insight": "High bowled% suggests bowler has excellent yorker or the batter plays away from the line"
        },

        "LBW": {
            "bowler_credit": True,
            "fielder_credit": False,
            "tactical_note": "Ball hitting pad in line with stumps — good length hitting the pads",
            "common_causes": ["full length on middle stump", "late inswing", "arm ball from spinner", "batter missing straight delivery"],
            "pattern_insight": "High LBW% against a bowler suggests that bowler targets the pads effectively — good line bowler"
        },

        "Run Out": {
            "bowler_credit": False,
            "fielder_credit": True,
            "tactical_note": "Fielding pressure or miscommunication between batters",
            "common_causes": ["direct hit", "sharp fielding at short leg/point", "poor calling between batters", "pressure from fielder"],
            "pattern_insight": "High run out% in death overs suggests fielding team creating excellent pressure through ground fielding"
        },

        "Stumped": {
            "bowler_credit": True,
            "fielder_credit": True,
            "tactical_note": "Batter played for turn/movement that wasn't there — went down the track",
            "common_causes": ["spinner with wide turn", "googly or wrong one", "pace-off delivery", "batter overcommitting to drive"],
            "pattern_insight": "High stumped% against a bowler shows the bowler is beating the batter with flight or pace variation"
        },

        "Hit Wicket": {
            "bowler_credit": True,
            "fielder_credit": False,
            "tactical_note": "Batter dislodged own stumps — rare. Usually on pull shot, ducking bouncer, or aggressive drive",
            "common_causes": ["pulling a short ball and losing balance", "backing away to leg and hitting off stump"],
            "pattern_insight": "Very rare — flag as a curiosity in analysis"
        },

        "Handled Ball": {
            "bowler_credit": False,
            "fielder_credit": False,
            "tactical_note": "Obstructing the field or handling the ball — extremely rare",
            "common_causes": ["deliberately deflecting ball away from stumps"],
            "pattern_insight": "Mention as unusual circumstance"
        },

        "Retired Out": {
            "bowler_credit": False,
            "fielder_credit": False,
            "tactical_note": "Batter chose to retire — sometimes tactical in shorter formats to bring in a pinch-hitter",
            "pattern_insight": "Treat as not out for average calculations unless specifically counting retirements"
        }
    },

    # =================================================
    # BOWLING TYPE CONTEXT
    # =================================================

    "bowling_types": {

        "pace": {
            "sub_types": ["Fast", "Fast-Medium", "Medium-Fast", "Medium"],
            "key_metrics": ["economy_rate", "dot_ball_percentage", "bowling_average"],
            "tactical_weapons": ["yorker", "bouncer", "outswing", "inswing", "reverse swing", "slower ball", "knuckleball"],
            "vulnerable_phase": "powerplay and death overs",
            "insight": "Pace bowlers are most dangerous with the new ball in powerplay and with reverse swing in death overs"
        },

        "spin": {
            "sub_types": ["Off-spin", "Leg-spin", "Left-arm orthodox", "Left-arm chinaman"],
            "key_metrics": ["economy_rate", "wickets", "dot_ball_percentage"],
            "tactical_weapons": ["turn", "flight", "googly", "doosra", "arm ball", "carrom ball", "topspinner"],
            "vulnerable_phase": "middle overs",
            "insight": "Spinners are most effective in middle overs on used pitches; danger in powerplay if pitch assists"
        }
    },

    # =================================================
    # BATTING POSITION ROLES  (NEW)
    # Tactical context for positional analysis
    # =================================================

    "batting_position_roles": {

        "opener_1": {
            "positions": [1],
            "role": "Anchor or aggressor — faces new ball, sets tone of innings",
            "expected_sr_T20": "130-160",
            "expected_avg_T20": "25+",
            "tactical_note": "Opener 1 often the most technically correct — faces most balls in powerplay"
        },

        "opener_2": {
            "positions": [2],
            "role": "Power hitter or accumulator alongside opener 1",
            "expected_sr_T20": "130-170",
            "tactical_note": "Opening pair chemistry critical — one anchor, one aggressor is common strategy"
        },

        "top_order_3": {
            "positions": [3],
            "role": "Best batter in team. Comes in early, bats longest. Bridge between openers and middle order.",
            "expected_sr_T20": "135-155",
            "expected_avg_T20": "30+",
            "tactical_note": "No. 3 is pivotal — must handle both powerplay and death if needed"
        },

        "middle_order": {
            "positions": [4, 5, 6],
            "role": "Stabiliser (4), accelerator (5), finisher (6). Must be adaptable to any match situation.",
            "expected_sr_T20": "140-165",
            "tactical_note": "Middle order performance often decides match outcome — wickets here are the most costly"
        },

        "finisher": {
            "positions": [5, 6, 7],
            "role": "Designed to bat at the death — maximise runs in final overs. Must hit big from ball one.",
            "expected_sr_T20": "150-200 in death overs",
            "tactical_note": "A good finisher can add 20-30 runs to an innings total alone"
        },

        "lower_order": {
            "positions": [7, 8, 9, 10, 11],
            "role": "Support role — chip in with useful runs. Protect specialist batters' strike.",
            "tactical_note": "Lower order partnerships can be match-changing if top order under-performs"
        }
    },

    # =================================================
    # FIELDING POSITIONS
    # =================================================

    "fielding_positions": {

        "catching_positions": [
            "slip", "gully", "point", "cover", "mid-off", "mid-on",
            "square leg", "fine leg", "third man", "long-on", "long-off",
            "deep square leg", "deep mid-wicket"
        ],

        "pressure_positions": [
            "slip", "gully", "short leg", "silly mid-on", "silly mid-off"
        ],

        "powerplay_positions": {
            "inside_circle": "All fielders except 2 must be inside the 30-yard circle in T20 powerplay",
            "tactical_note": "Gaps at cover and mid-wicket are common powerplay targets for batters"
        },

        "death_fielding": {
            "typical_setup": "Two men on the boundary — long-on and long-off. Fine leg and deep square leg.",
            "tactical_note": "Death fielding errors (drops, misfields) can cost 10-15 runs in T20"
        }
    },

    # =================================================
    # INNINGS CONTEXT RULES
    # =================================================

    "innings_rules": {

        "always_include_innings_played": True,
        "compare_innings_before_conclusions": True,
        "mention_runs_per_innings": True,
        "mention_wickets_per_innings": True,
        "mention_balls_per_innings": True,
        "note_not_out_innings": True,
        "separate_batting_bowling_innings": True,

        "not_out_note": "Not-out innings inflate batting averages — always note how many not-outs in a sample",
        "innings_progression": "Compare early innings (positions 1-3) vs middle (4-6) vs tail (7-11) contribution to understand team balance"
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
            "Compare powerplay runs vs format benchmark (T20: 45-55 runs)",
            "Identify strongest batting phase by run rate",
            "Note boundary hitting rate — high boundary% = aggressive team",
            "Note dot ball percentage — high dot% = team under pressure",
            "Death overs scoring rate critical for final total"
        ],

        "bowling_team_context": [
            "Compare economy to format benchmark",
            "Identify most economical phase — where team creates most pressure",
            "Note wicket-taking frequency per over",
            "Death overs economy is the hardest to control — weight accordingly",
            "Powerplay wickets change match tempo dramatically"
        ],

        "win_loss_context": {
            "note": "If win/loss data available, split stats by match result to find what drives winning",
            "pattern": "Teams that win tend to take 3+ wickets in powerplay (T20) and score 170+ batting first"
        }
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
                "insight": "High-scoring ground — batters dominate. Spinners struggle. Chasing is difficult."
            },

            "bowlers_pitch": {
                "indicator": "avg_first_innings < 140 (T20)",
                "insight": "Low-scoring — bowlers dominate. Pace and bounce key. Batting first gives big advantage."
            },

            "spin_friendly": {
                "indicator": "spin_wicket_percentage > 50%",
                "insight": "Spinners dominate — turn and bounce. Teams should play 3 spinners."
            },

            "pace_friendly": {
                "indicator": "pace_wicket_percentage > 60%",
                "insight": "Seam and swing — new ball crucial. Teams should play extra pacer."
            },

            "balanced": {
                "indicator": "avg_first_innings 155-175 (T20)",
                "insight": "Balanced surface — both batting and bowling rewarded. Toss and batting depth matter."
            }
        }
    },

    # =================================================
    # PARTNERSHIP ANALYSIS
    # =================================================

    "partnership_rules": {

        "key_partnerships": [
            "Opening partnership (1st wicket) — sets tone, exploits powerplay",
            "Middle order anchor partnership (3rd-4th wicket) — rebuilds or consolidates",
            "Acceleration partnership (5th-6th wicket) — score-building in middle overs",
            "Death partnership (7th-10th wicket) — tail wagging or consolidating total"
        ],

        "always_include": [
            "runs_added",
            "balls_faced_together",
            "run_rate_during_partnership",
            "boundaries_hit_together"
        ],

        "benchmark_partnerships": {
            "T20_opening": "50+ opening stand is excellent. 30+ is par.",
            "T20_match_winning": "80+ partnership in any wicket is potentially match-winning",
            "T20_tail_contribution": "20+ from lower-order partnership is a bonus"
        },

        "context_notes": [
            "Opening partnerships set the tempo for the entire innings",
            "Long partnerships indicate batter temperament and compatibility",
            "Late order partnerships can change match outcome by 15-25 runs",
            "Run rate during partnership vs overall team run rate shows impact"
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
            "A batter dismissed 3+ times by the same bowler has a clear matchup weakness",
            "T20 SR below 100 against a specific bowler means the bowler controls the matchup",
            "T20 SR above 180 against a bowler means the batter completely dominates",
            "High dot ball% (>55%) faced against a bowler indicates real difficulty scoring",
            "0 dismissals across 20+ balls suggests batter has neutralised the bowler",
            "Look for patterns: does the batter struggle early (first 5 balls) or late in spell?"
        ],

        "dismissal_pattern_notes": [
            "Same dismissal type 3+ times suggests a repeatable trap being set",
            "LBW twice suggests bowler targets pads effectively vs this batter",
            "Caught twice in same region suggests bowler exploits a shot shape weakness"
        ]
    },

    # =================================================
    # COMPARISON RULES
    # =================================================

    "comparison_rules": [

        "Always compare innings played before comparing cumulative totals",
        "Always compare balls faced before comparing strike rates",
        "Always compare balls bowled before comparing economy rates",
        "Use per-innings metrics for fair player comparison",
        "Use per-match metrics for team-level comparisons",
        "Always mention sample size — small samples need explicit caveats",
        "Do not rank players on fewer than minimum sample size unless noted",
        "Rank players by efficiency metrics (SR, average, economy) not just volumes",
        "Mention format context — T20 SR of 130 vs ODI SR of 130 are very different achievements",
        "Note if one entity played in favorable conditions vs tough conditions",
        "For career comparisons: break into eras or year ranges if performance changed significantly",
        "For team comparisons: note opposition quality if available"
    ],

    # =================================================
    # INSIGHT GENERATION RULES
    # =================================================

    "insight_rules": [

        "Always mention innings played and sample size before drawing conclusions",
        "Rank against peers when data allows — 'top 5 in the dataset', 'best economy in the period'",
        "Compare against format benchmark tier labels — not just dataset average",
        "Every strength claim must cite a specific number",
        "Every weakness claim must cite a specific number",
        "Always include phase-wise performance if data has it",
        "Mention opposition quality if head-to-head data available",
        "Mention venue or ground trends when data spans multiple grounds",
        "Identify patterns: flat-track bully vs tough-conditions performer",
        "Mention tactical implications for opposing captain — 'opposition should attack with spin in middle overs'",
        "Use cricket commentary language naturally — yorker, doosra, knuckleball, corridor of uncertainty",
        "Avoid generic statements — every claim must be backed by a number",
        "End with a definitive verdict — not a vague summary",
        "If comparing two entities, state which is better and why, do not sit on the fence",
        "For time-trend questions: identify the single best year and single worst year with exact stats",
        "For dismissal analysis: identify the primary method of dismissal and its tactical implication",
        "For phase analysis: identify the strongest and weakest phase and the tactical consequence",
        "Note if a player performs differently in first vs second innings (T20/ODI: setting vs chasing)"
    ],

    # =================================================
    # SPECIAL MATCH SITUATIONS
    # =================================================

    "match_situations": {

        "powerplay_batting": {
            "expectation_T20": "40-55 runs with maximum 2 wickets lost",
            "insight": "Getting 2 wickets in powerplay gives bowling team a 25-30 run advantage on average"
        },

        "death_batting": {
            "expectation_T20": "50-65 runs in last 5 overs (overs 16-20)",
            "insight": "Finishing ability separates elite T20 teams. A dedicated finisher can add 20+ runs alone."
        },

        "chase_pressure": {
            "insight": "Chasing teams with strong openers tend to win more often — run rate in powerplay is critical in chases",
            "T20_note": "Required run rate above 10 in T20 = difficult; above 12 = very difficult"
        },

        "spin_in_middle": {
            "insight": "Spinners bowling in overs 7-15 (T20) are key to containing run rate. Economy < 7.5 is elite here."
        },

        "collapse_recovery": {
            "insight": "Losing 3 wickets in powerplay (T20) makes recovery very difficult. Tail partnerships critical."
        },

        "low_total_defence": {
            "T20_threshold": "Defending a total under 140 in T20 requires 3+ wickets in powerplay",
            "insight": "Bowling teams defending small totals must attack — defensive bowling leads to defeat"
        },

        "super_over": {
            "insight": "Best striker (highest SR) and best death bowler are the defining selections in super over"
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
            "bowling_economy_maximum": 9.0,
            "classification_note": "True T20 all-rounder must contribute meaningfully in both disciplines — not just fill an extra bowling slot"
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

        "triggered_by": "No-ball (over-stepping the crease)",
        "batter_advantage": "Cannot be dismissed by any method except run out",
        "expected_outcome": "Batter should score 4, 6, or minimum 2 — dot ball on free hit is a missed opportunity",
        "insight": "Free hit dot balls represent a batting failure. Free hit wickets (run-out) are extremely rare and represent a fielding triumph.",
        "analysis_note": "Track free_hit = TRUE balls for batting aggression analysis. Compare free hit SR vs normal ball SR."
    },

    # =================================================
    # KEEPER POSITION IMPACT
    # =================================================

    "keeper_position_rules": {

        "keeper_up": {
            "meaning": "Wicketkeeper standing up to stumps — right behind the stumps",
            "impact": "Limits batter's ability to leave crease — increases stumping risk. Psychologically restricts the batter.",
            "common_for": "Spin bowling, medium pace on slow pitches, keeping pressure on batters"
        },

        "keeper_back": {
            "meaning": "Wicketkeeper standing well back from stumps",
            "impact": "More ground to cover for stumpings — batter can charge down the track more safely",
            "common_for": "Fast and medium-fast bowling where ball comes through quickly"
        },

        "analysis_note": "Compare batter SR and dismissal rate when keeper_up vs keeper_back to find if the keeper position impacts a specific batter"
    },

    # =================================================
    # BOWLING ANGLE RULES
    # =================================================

    "bowling_angle_rules": {

        "around_the_wicket": {
            "meaning": "Bowler delivers from the non-dominant side of the stumps",
            "right_arm_to_right_batter": "Angles into the body — strong LBW and bowled threat. Hard to play off side.",
            "right_arm_to_left_batter": "Angles away — invites drive and edge to slip/gully",
            "left_arm_to_right_batter": "Angles away — classic left-arm angle, danger of caught behind",
            "insight": "Compare economy and wicket rate around vs over wicket to find bowler's preferred angle"
        },

        "over_the_wicket": {
            "meaning": "Standard delivery angle from dominant side",
            "right_arm_to_right_batter": "Standard — can swing away or hold its line",
            "insight": "Most bowlers are more effective over the wicket — note when they switch to around as a tactical change"
        }
    },

    # =================================================
    # LEGAL BALL RULES
    # =================================================

    "legal_ball_rules": {

        "legal_ball_true": "Counts toward the over. Use for SR, economy, dot%, boundary% calculations.",
        "legal_ball_false": "Wide or no-ball. Gives batting team free runs and does not count toward over.",
        "analysis_note": "ALWAYS filter legal_ball = TRUE for strike rate and economy calculations to avoid inflated stats",
        "extra_types": ["wide", "no_ball", "bye", "leg_bye"],
        "discipline_insight": "High no-ball and wide counts signal bowling discipline problems. Calculate extras rate as: (wides + no-balls) / total balls * 100",
        "free_runs_impact": "A bowler giving 5+ extras per spell in T20 is effectively bowling an extra over for the batting side"
    },

    # =================================================
    # SCORING ZONES (SHOT CLASSIFICATION)
    # =================================================

    "scoring_zone_context": {

        "boundary_4": {
            "meaning": "Ball reaches boundary on the ground — 4 runs",
            "insight": "High 4s count = gap-finding ability, placement, and timing. Ground game."
        },

        "boundary_6": {
            "meaning": "Ball clears the boundary in the air — 6 runs",
            "insight": "High 6s count = power hitting, aerial game, big-hitting ability. Sky game."
        },

        "dot_ball": {
            "meaning": "0 runs scored off a legal delivery",
            "insight": "High dot ball % for a batter = under pressure or being defensive. High dot% for bowler = excellent control."
        },

        "singles_and_twos": {
            "meaning": "1 or 2 runs from a legal delivery — running between wickets",
            "insight": "High 1s/2s proportion = good strike rotation and running ability, but may signal difficulty finding boundaries"
        },

        "six_to_four_ratio": {
            "meaning": "Ratio of sixes to fours hit",
            "insight": "High 6:4 ratio (>0.5) = power hitter who prefers aerial route. Low ratio = ground game player."
        }
    },

    # =================================================
    # CONSISTENCY METRICS  (NEW)
    # Quantifying reliability across innings
    # =================================================

    "consistency_metrics": {

        "innings_score_bands": {
            "duck":          {"range": "0",      "label": "Duck — complete failure"},
            "low":           {"range": "1-9",    "label": "Single digit — below par"},
            "start":         {"range": "10-19",  "label": "Good start but not converted"},
            "contribution":  {"range": "20-29",  "label": "Useful contribution"},
            "solid":         {"range": "30-49",  "label": "Solid innings — real value"},
            "fifty":         {"range": "50-74",  "label": "Half-century — match impact"},
            "big_fifty":     {"range": "75-99",  "label": "Near-century — match-winning potential"},
            "century":       {"range": "100+",   "label": "Century — exceptional innings"}
        },

        "consistency_thresholds": {
            "T20_consistent":  "Scoring 20+ in >50% of innings",
            "T20_explosive":   "Scoring 50+ in >20% of innings",
            "T20_reliable":    "Duck rate below 10%",
            "high_variance":   "Large gap between average and median score indicates boom-or-bust player"
        },

        "milestone_counts": {
            "ducks":     "COUNT of innings where batter dismissed for 0",
            "fifties":   "COUNT of innings where batter scored 50-99",
            "centuries": "COUNT of innings where batter scored 100+",
            "30_plus":   "COUNT of innings where batter scored 30+ — useful consistency metric in T20"
        }
    },

    # =================================================
    # PRESSURE / CLUTCH CONTEXT  (NEW)
    # Defining high-pressure situations
    # =================================================

    "pressure_context": {

        "clutch_batting_situations": [
            "Chasing with required rate > 10 (T20)",
            "Team lost 3 wickets in powerplay — must rebuild",
            "Last 5 overs — must score at > 12 RPO",
            "Super over — ultimate pressure",
            "Must-win knockout match"
        ],

        "clutch_bowling_situations": [
            "Defending a total under 140 in T20 — must take wickets",
            "Death overs — protecting a total of 5-10 runs",
            "Free hit delivery — batter has no dismissal risk",
            "Super over bowling"
        ],

        "pressure_indicators": {
            "batting": "SR drops significantly vs career average = under pressure",
            "bowling": "Economy spikes in death overs vs overall economy = struggles under pressure"
        }
    },

    # =================================================
    # RECENT FORM CONTEXT  (NEW)
    # How to interpret and present form data
    # =================================================

    "recent_form_context": {

        "form_window": {
            "short_form":  "Last 5 matches — snapshot of current momentum",
            "medium_form": "Last 10 matches — reliable form indicator",
            "long_form":   "Last 20 matches — trends and patterns"
        },

        "form_vs_career": {
            "hot_form":  "Recent average/SR > 20% above career figure — player in form",
            "cold_form": "Recent average/SR > 20% below career figure — player struggling",
            "normal":    "Within 10% of career figure — consistent performer"
        },

        "form_trend_insight": [
            "Compare last 5 vs last 10 to spot emerging trends",
            "Consecutive failures (3+ low scores) = lean patch",
            "Two back-to-back big scores doesn't establish form — needs a run",
            "Form trends heading into knockout stages or important series matter most"
        ]
    },

    # =================================================
    # OVER-BY-OVER CONTEXT  (NEW)
    # Expected run rates and wicket probabilities per over
    # =================================================

    "over_context": {

        "T20_expected_runs_per_over": {
            "over_1":   {"expected": "5-7",   "note": "Bowlers have advantage with new ball"},
            "over_2":   {"expected": "6-8",   "note": "Batters starting to settle"},
            "over_3":   {"expected": "7-9",   "note": "Good powerplay over — batters attacking"},
            "over_4":   {"expected": "7-9",   "note": "Mid powerplay — key contest"},
            "over_5":   {"expected": "8-10",  "note": "Powerplay push begins"},
            "over_6":   {"expected": "8-11",  "note": "Last powerplay over — high aggression"},
            "over_7":   {"expected": "6-8",   "note": "First middle over — often spin introduced"},
            "over_10":  {"expected": "6-8",   "note": "Mid middle overs — consolidation"},
            "over_15":  {"expected": "7-9",   "note": "End of middle — pre-death acceleration"},
            "over_16":  {"expected": "9-12",  "note": "Death begins — aggression ramps up"},
            "over_18":  {"expected": "10-13", "note": "Peak aggression — boundaries expected"},
            "over_19":  {"expected": "11-14", "note": "Second-to-last — crucial"},
            "over_20":  {"expected": "12-16", "note": "Final over — maximum hitting"}
        },

        "wicket_danger_overs": {
            "T20": [1, 2, 6, 7, 16, 17],
            "note": "Overs 1-2 (new ball), over 6-7 (transition), overs 16-17 (death entry) are highest wicket probability"
        }
    },

    # =================================================
    # OPPONENT ANALYSIS CONTEXT  (NEW)
    # How to interpret split-by-opponent data
    # =================================================

    "opponent_analysis_context": {

        "bogey_opponent": {
            "definition": "Team/bowler a player consistently underperforms against",
            "indicator_batting": "SR or average > 25% below career figure vs that opponent",
            "indicator_bowling": "Economy > 15% above career figure vs that opponent",
            "insight": "Bogey opponents may expose a technical or tactical weakness — look at dismissal patterns"
        },

        "favourite_opponent": {
            "definition": "Team/bowler a player consistently dominates",
            "indicator_batting": "SR or average > 25% above career figure vs that opponent",
            "indicator_bowling": "Economy > 15% below career figure vs that opponent",
            "insight": "Favourite opponents may have weak links that the player exploits"
        },

        "opposition_quality_note": "Performances against top-ranked teams carry more weight than vs lower-ranked opposition"
    },

    # =================================================
    # MILESTONE TRACKING CONTEXT  (NEW)
    # For milestone-based queries
    # =================================================

    "milestone_context": {

        "batting_milestones": {
            "duck":       "Dismissed for 0 — scored no runs",
            "golden_duck": "Dismissed on first ball — worst possible outcome",
            "fifty":      "Scored 50-99 in an innings",
            "century":    "Scored 100+ in an innings",
            "double_century": "Scored 200+ in a Test innings",
            "fastest_fifty": "Least balls to reach 50 — measure of aggression",
            "fastest_century": "Least balls to reach 100"
        },

        "bowling_milestones": {
            "maiden_over":     "Conceded 0 runs in a complete over",
            "hat_trick":       "Three wickets on three consecutive balls",
            "five_wicket_haul": "5 or more wickets in a single innings",
            "ten_wicket_match": "10 or more wickets across both innings (Test only)"
        },

        "sql_patterns": {
            "duck": "runs_batter = 0 AND wicket IS NOT NULL",
            "golden_duck": "runs_batter = 0 AND wicket IS NOT NULL AND ball_number = 1",
            "fifty_plus": "SUM(runs_batter) >= 50 per innings group",
            "century": "SUM(runs_batter) >= 100 per innings group"
        }
    },

    # =================================================
    # PERFORMANCE TIER LABELS
    # Used in insight output
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
        },

        "allround": {
            "elite":     "🔴 Elite All-Rounder — match-winner with bat AND ball",
            "excellent": "🟠 Excellent All-Rounder — significant contributor both ways",
            "good":      "🟡 Good All-Rounder — chips in with bat and ball",
            "average":   "🟢 Bits-and-pieces Player — limited impact in both roles",
            "poor":      "⚪ Not an All-Rounder — one-dimensional"
        }
    },

    # =================================================
    # QUERY RESOLUTION HINTS  (NEW)
    # Guides the AI when questions are ambiguous
    # =================================================

    "query_resolution_hints": {

        "ambiguous_player_name": "Use ILIKE '%partial_name%' — player names may differ across datasets (e.g. 'V Kohli', 'Virat Kohli', 'Kohli')",
        "ambiguous_team_name": "Use ILIKE '%partial_name%' — team names vary (e.g. 'India', 'Indian', 'Team India')",
        "career_vs_recent": "If 'career' not specified, default to all available data. If 'recent' or 'form' specified, use last 10 matches.",
        "format_not_specified": "If no format mentioned, include all formats but note format breakdown in insight",
        "year_not_specified": "If no year mentioned, use all available years. Suggest yearly breakdown in chart.",
        "phase_not_specified": "For individual player queries always run phase breakdown as a secondary query",
        "opponent_not_specified": "For individual player queries, include opponent breakdown as tertiary query for deeper analysis",

        "complex_question_decomposition": [
            "Break complex questions into: (1) overall summary, (2) trend or split, (3) context or comparison",
            "Always answer the exact question asked, then add supporting context",
            "If multiple entities compared — lead with comparison table, then individual analysis",
            "If time range specified — filter strictly, do not include years outside range"
        ]
    }
}