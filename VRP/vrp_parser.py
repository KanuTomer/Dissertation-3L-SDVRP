from __future__ import annotations

from pathlib import Path


SECTION_NODE_COORD = "NODE_COORD_SECTION"
SECTION_DEMAND = "DEMAND_SECTION"
SECTION_DEPOT = "DEPOT_SECTION"
TERMINATOR = "EOF"


class VRPParseError(ValueError):
    pass


def _split_key_value(line: str) -> tuple[str, str] | None:
    if ":" not in line:
        return None
    key, value = line.split(":", 1)
    return key.strip(), value.strip()


def parse_vrp_file(vrp_path: str | Path) -> dict:
    vrp_path = Path(vrp_path)
    if not vrp_path.exists():
        raise FileNotFoundError(f"VRP file not found: {vrp_path}")

    lines = vrp_path.read_text(encoding="utf-8-sig").splitlines()

    headers: dict[str, str] = {}
    coordinates: dict[int, tuple[float, float]] = {}
    demands: dict[int, int] = {}
    depot_ids: list[int] = []
    section: str | None = None

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue

        if line == TERMINATOR:
            break

        if line in {SECTION_NODE_COORD, SECTION_DEMAND, SECTION_DEPOT}:
            section = line
            continue

        if section is None:
            key_value = _split_key_value(line)
            if key_value is not None:
                key, value = key_value
                headers[key.upper()] = value
            continue

        parts = line.split()
        if section == SECTION_NODE_COORD:
            if len(parts) < 3:
                raise VRPParseError(
                    f"Invalid node coordinate line {line_number} in {vrp_path.name}: {raw_line}"
                )
            node_id = int(parts[0])
            x_coord = float(parts[1])
            y_coord = float(parts[2])
            coordinates[node_id] = (x_coord, y_coord)
            continue

        if section == SECTION_DEMAND:
            if len(parts) < 2:
                raise VRPParseError(
                    f"Invalid demand line {line_number} in {vrp_path.name}: {raw_line}"
                )
            node_id = int(parts[0])
            demand = int(float(parts[1]))
            demands[node_id] = demand
            continue

        if section == SECTION_DEPOT:
            depot_id = int(parts[0])
            if depot_id == -1:
                section = None
            else:
                depot_ids.append(depot_id)

    if not coordinates:
        raise VRPParseError(f"No NODE_COORD_SECTION found in {vrp_path.name}")

    if not demands:
        raise VRPParseError(f"No DEMAND_SECTION found in {vrp_path.name}")

    if not depot_ids:
        raise VRPParseError(f"No depot ID found in DEPOT_SECTION for {vrp_path.name}")

    dimension_raw = headers.get("DIMENSION")
    if dimension_raw is not None:
        try:
            expected_dimension = int(dimension_raw)
        except ValueError as exc:
            raise VRPParseError(f"Invalid DIMENSION in {vrp_path.name}: {dimension_raw}") from exc
        if expected_dimension != len(coordinates):
            raise VRPParseError(
                f"DIMENSION mismatch in {vrp_path.name}: header={expected_dimension}, parsed={len(coordinates)}"
            )

    missing_demand_ids = sorted(set(coordinates) - set(demands))
    if missing_demand_ids:
        raise VRPParseError(
            f"Missing demand values for node IDs in {vrp_path.name}: {missing_demand_ids[:10]}"
        )

    primary_depot_id = depot_ids[0]
    if primary_depot_id not in coordinates:
        raise VRPParseError(
            f"Depot ID {primary_depot_id} is not present in NODE_COORD_SECTION for {vrp_path.name}"
        )

    customers: list[dict] = []
    for node_id in sorted(coordinates):
        x_coord, y_coord = coordinates[node_id]
        customers.append(
            {
                "id": node_id,
                "customer_id": node_id,
                "x": x_coord,
                "y": y_coord,
                "demand": demands[node_id],
                "is_depot": node_id == primary_depot_id,
            }
        )

    depot_x, depot_y = coordinates[primary_depot_id]
    instance_name = headers.get("NAME", vrp_path.stem).strip() or vrp_path.stem

    return {
        "instance_name": instance_name,
        "inst_name": instance_name,
        "name": instance_name,
        "source_file": vrp_path.name,
        "dimension": len(coordinates),
        "capacity": int(float(headers.get("CAPACITY", "0"))),
        "depot_id": primary_depot_id,
        "depot": [depot_x, depot_y],
        "customers": customers,
        "headers": headers,
    }
