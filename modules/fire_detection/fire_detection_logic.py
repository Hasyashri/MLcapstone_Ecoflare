# modules/fire_detection/fire_detection_logic.py

"""
MVP 1 – Wildfire Detection using API Fusion + Voting.

This module:
- Calls the five data sources (FIRMS, CWFIS, Weather, IoT, Vegetation).
- Applies a simple voting rule to decide if a fire is detected.
- Returns a structured dictionary to be used by FastAPI / Streamlit.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List
import datetime

from .fetch_live_data import get_nasa_firms, get_cwfis, get_weather
from .iot_data import get_iot_sensor
from .vegetation_data import get_vegetation_risk


@dataclass
class SourceResult:
    name: str
    status: str          # "OK" or "ALERT"
    details: Dict[str, Any]


@dataclass
class DetectionResult:
    region: str
    time_window: str
    fire_detected: bool
    vote_count: int
    threshold: int
    sources: List[SourceResult]
    generated_at: str


def run_detection_voting(
    region: str,
    time_window: str,
    lat: float,
    lon: float,
    location_id: str,
    threshold: int = 2,
) -> Dict[str, Any]:
    """
    Core detection function.

    Fire is detected if at least `threshold` sources are in ALERT status.
    """

    # Call the 5 data sources
    raw_sources: List[Dict[str, Any]] = [
        get_nasa_firms(region, time_window),
        get_cwfis(region),
        get_weather(lat, lon),
        get_iot_sensor(location_id),
        get_vegetation_risk(lat, lon),
    ]

    sources: List[SourceResult] = [
        SourceResult(
            name=s["name"],
            status=s["status"],
            details=s["details"],
        )
        for s in raw_sources
    ]

    vote_count = sum(1 for s in sources if s.status == "ALERT")
    fire_detected = vote_count >= threshold

    result = DetectionResult(
        region=region,
        time_window=time_window,
        fire_detected=fire_detected,
        vote_count=vote_count,
        threshold=threshold,
        sources=sources,
        generated_at=datetime.datetime.utcnow().isoformat() + "Z",
    )

    output = asdict(result)
    output["sources"] = [asdict(s) for s in sources]
    return output


# Quick manual test
if __name__ == "__main__":
    demo = run_detection_voting(
        region="BC-North",
        time_window="last_60m",
        lat=54.0,
        lon=-125.0,
        location_id="sensor-001",
        threshold=2,
    )
    from pprint import pprint

    pprint(demo)
