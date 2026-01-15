# backend/utils/species_metadata.py

SPECIES_METADATA = {
    "black_bear": {
        "aliases": ["black_bear"],
        "age_thresholds": {
            "juvenile": 3800,
            "adult": 8500
        },
        "wildness": "Very High",
        "human_risk": "Extreme"
    },

    "bob_cat": {
        "aliases": ["bob_cat", "bobcat"],
        "age_thresholds": {
            "juvenile": 1200,
            "adult": 3000
        },
        "wildness": "Medium",
        "human_risk": "Moderate"
    },

    "elephant": {
        "aliases": ["elephant"],
        "age_thresholds": {
            "juvenile": 12000,
            "adult": 25000
        },
        "wildness": "Very High",
        "human_risk": "Extreme"
    },

    "gray_fox": {
        "aliases": ["gray_fox"],
        "age_thresholds": {
            "juvenile": 1600,
            "adult": 3600
        },
        "wildness": "Medium",
        "human_risk": "Low"
    },

    "horse": {
        "aliases": ["horse"],
        "age_thresholds": {
            "juvenile": 5000,
            "adult": 11000
        },
        "wildness": "Medium",
        "human_risk": "Moderate"
    },

    "lion": {
        "aliases": ["lion"],
        "age_thresholds": {
            "juvenile": 4200,
            "adult": 9000
        },
        "wildness": "Very High",
        "human_risk": "Extreme"
    }
}
