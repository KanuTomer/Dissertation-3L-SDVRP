from pathlib import Path
import json

def load_cvrp_and_boxes(path: str) -> dict:
    """Load a merged dataset (CVRP + boxes). Accepts JSON for skeleton.
    Uses utf-8-sig to tolerate BOMs and different editors on Windows.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {p.resolve()}")
    if p.suffix.lower() == ".json":
        # use utf-8-sig to silently handle BOM if present
        with p.open("r", encoding="utf-8-sig") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON file {p.resolve()}: {e}") from e
    raise ValueError(f"Unsupported dataset format for skeleton loader: {p.suffix}")
