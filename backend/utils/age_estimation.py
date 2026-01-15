# backend/utils/age_estimation.py

from backend.utils.species_metadata import SPECIES_METADATA


def estimate_age_group(animal_name: str, footprint_area: float) -> str:
    """
    Estimate animal age group based on footprint area (in pixels).

    Returns one of:
    - Juvenile
    - Sub-Adult
    - Adult
    - Unknown
    """

    if not animal_name or footprint_area <= 0:
        return "Unknown"

    animal = animal_name.strip().lower()

    for meta in SPECIES_METADATA.values():
        if animal in meta["aliases"]:
            t = meta["age_thresholds"]

            if footprint_area < t["juvenile"]:
                return "Juvenile"
            elif footprint_area < t["adult"]:
                return "Sub-Adult"
            else:
                return "Adult"

    return "Unknown"
