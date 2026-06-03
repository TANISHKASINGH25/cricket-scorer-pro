DERIVED_METRICS = {

    # =========================================================
    # BATTING METRICS
    # =========================================================

    "strike_rate": {
        "description": "Batting strike rate",
        "formula": '(SUM("Runs") * 100.0 / COUNT(*))',
        "category": "batting",
        "synonyms": [
            "strike rate",
            "batting strike rate",
            "scoring rate"
        ]
    },

    "batting_average": {
        "description": "Average runs scored per dismissal",
        "formula": '''
        (
            SUM("Runs") /
            NULLIF(
                SUM(
                    CASE
                        WHEN "Wicket" = TRUE THEN 1
                        ELSE 0
                    END
                ),
                0
            )
        )
        ''',
        "category": "batting",
        "synonyms": [
            "average",
            "batting average"
        ]
    },

    "boundary_runs": {
        "description": "Runs scored in boundaries",
        "formula": '''
        SUM(
            CASE
                WHEN "Runs" IN (4, 6)
                THEN "Runs"
                ELSE 0
            END
        )
        ''',
        "category": "batting",
        "synonyms": [
            "boundary runs",
            "fours and sixes"
        ]
    },

    "boundary_percentage": {
        "description": "Percentage of runs scored in boundaries",
        "formula": '''
        (
            SUM(
                CASE
                    WHEN "Runs" IN (4, 6)
                    THEN 1
                    ELSE 0
                END
            ) * 100.0 / COUNT(*)
        )
        ''',
        "category": "batting",
        "synonyms": [
            "boundary percentage",
            "boundary frequency"
        ]
    },

    # =========================================================
    # BOWLING METRICS
    # =========================================================

    "economy_rate": {
        "description": "Runs conceded per over",
        "formula": '''
        (
            SUM("Runs") / (COUNT(*) / 6.0)
        )
        ''',
        "category": "bowling",
        "synonyms": [
            "economy",
            "economy rate",
            "bowling economy"
        ]
    },

    "bowling_strike_rate": {
        "description": "Balls bowled per wicket",
        "formula": '''
        (
            COUNT(*) /
            NULLIF(
                SUM(
                    CASE
                        WHEN "Wicket" = TRUE THEN 1
                        ELSE 0
                    END
                ),
                0
            )
        )
        ''',
        "category": "bowling",
        "synonyms": [
            "bowling strike rate",
            "balls per wicket"
        ]
    },

    "dot_ball_percentage": {
        "description": "Percentage of dot balls bowled",
        "formula": '''
        (
            SUM(
                CASE
                    WHEN "Runs" = 0
                    THEN 1
                    ELSE 0
                END
            ) * 100.0 / COUNT(*)
        )
        ''',
        "category": "bowling",
        "synonyms": [
            "dot ball percentage",
            "dot ball rate"
        ]
    },

    # =========================================================
    # MATCH STATE METRICS
    # =========================================================

    "run_rate": {
        "description": "Runs scored per over",
        "formula": '''
        (
            SUM("Runs") / (COUNT(*) / 6.0)
        )
        ''',
        "category": "match_state",
        "synonyms": [
            "run rate",
            "rr"
        ]
    },

    "required_run_rate": {
        "description": "Required runs per over",
        "formula": 'AVG("Req Run Rate After")',
        "category": "match_state",
        "synonyms": [
            "required run rate",
            "rrr"
        ]
    },

    # =========================================================
    # PHYSICS / TRACKING METRICS
    # =========================================================

    "average_speed": {
        "description": "Average bowling speed",
        "formula": 'AVG("Speed")',
        "category": "ball_physics",
        "synonyms": [
            "average speed",
            "pace"
        ]
    },

    "average_swing": {
        "description": "Average swing movement",
        "formula": 'AVG("SwingAngle")',
        "category": "ball_physics",
        "synonyms": [
            "average swing",
            "ball movement"
        ]
    },

    "average_deviation": {
        "description": "Average seam deviation",
        "formula": 'AVG("Deviation")',
        "category": "ball_physics",
        "synonyms": [
            "deviation",
            "seam movement"
        ]
    }

}