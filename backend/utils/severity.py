# backend/utils/severity.py

from backend.utils.species_metadata import SPECIES_METADATA


def estimate_severity(animal_name: str):
    """
    Returns wildness and human risk severity for an animal.

    Output keys:
    - wildness
    - human_risk
    """

    if not animal_name:
        return {"wildness": "Unknown", "human_risk": "Unknown"}

    animal = animal_name.strip().lower()

    for meta in SPECIES_METADATA.values():
        if animal in meta["aliases"]:
            return {
                "wildness": meta["wildness"],
                "human_risk": meta["human_risk"]
            }

    return {"wildness": "Unknown", "human_risk": "Unknown"}
