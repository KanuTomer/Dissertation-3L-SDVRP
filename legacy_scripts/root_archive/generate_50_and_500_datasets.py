from pathlib import Path
import re

xml_dir = Path(r"C:\Kanu\Kanu(D)\Dissertation\XML")

# ---------------------------------------------------
# Helper to parse .vrp file
# ---------------------------------------------------
def parse_vrp(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    metadata = {}
    coords = []
    demands = []
    depot = []

    section = None

    for line in lines:
        if not line:
            continue

        if line.startswith("NAME"):
            metadata["NAME"] = line.split(":")[1].strip()
        elif line.startswith("COMMENT"):
            metadata["COMMENT"] = line.split(":", 1)[1].strip()
        elif line.startswith("TYPE"):
            metadata["TYPE"] = line.split(":")[1].strip()
        elif line.startswith("DIMENSION"):
            metadata["DIMENSION"] = int(line.split(":")[1].strip())
        elif line.startswith("EDGE_WEIGHT_TYPE"):
            metadata["EDGE_WEIGHT_TYPE"] = line.split(":")[1].strip()
        elif line.startswith("CAPACITY"):
            metadata["CAPACITY"] = int(line.split(":")[1].strip())
        elif line == "NODE_COORD_SECTION":
            section = "coords"
        elif line == "DEMAND_SECTION":
            section = "demand"
        elif line == "DEPOT_SECTION":
            section = "depot"
        elif line == "EOF":
            break
        else:
            if section == "coords":
                parts = re.split(r"\s+", line)
                coords.append({
                    "id": int(parts[0]),
                    "x": int(parts[1]),
                    "y": int(parts[2]),
                })
            elif section == "demand":
                parts = re.split(r"\s+", line)
                demands.append({
                    "id": int(parts[0]),
                    "demand": int(parts[1]),
                })
            elif section == "depot":
                if line != "-1":
                    depot.append(int(line))

    return metadata, coords, demands, depot


# ---------------------------------------------------
# Create XML50 from one XML100 file
# ---------------------------------------------------
source_50 = xml_dir / "XML100_1111_01.vrp"
metadata, coords, demands, depot = parse_vrp(source_50)

coords_50 = coords[:51]
demands_50 = demands[:51]

output_50 = xml_dir / "XML50_1111_01.vrp"

with open(output_50, "w", encoding="utf-8") as f:
    f.write("NAME : XML50_1111_01\n")
    f.write("COMMENT : Generated from XML100_1111_01\n")
    f.write("TYPE : CVRP\n")
    f.write("DIMENSION : 51\n")
    f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
    f.write("CAPACITY : 2\n")
    f.write("NODE_COORD_SECTION\n")

    for row in coords_50:
        f.write(f"{row['id']} {row['x']} {row['y']}\n")

    f.write("DEMAND_SECTION\n")

    for row in demands_50:
        f.write(f"{row['id']} {row['demand']}\n")

    f.write("DEPOT_SECTION\n")
    f.write("1\n")
    f.write("-1\n")
    f.write("EOF\n")

print(f"Created {output_50}")


# ---------------------------------------------------
# Create XML500 from five XML100 files
# ---------------------------------------------------
source_files_500 = [
    xml_dir / "XML100_1111_01.vrp",
    xml_dir / "XML100_1111_02.vrp",
    xml_dir / "XML100_1111_03.vrp",
    xml_dir / "XML100_1111_04.vrp",
    xml_dir / "XML100_1111_05.vrp",
]

combined_coords = []
combined_demands = []

new_id = 1

for file_index, vrp_file in enumerate(source_files_500):
    metadata, coords, demands, depot = parse_vrp(vrp_file)

    # Keep depot only from first file
    if file_index == 0:
        depot_coord = coords[0]
        depot_demand = demands[0]

        combined_coords.append({
            "id": 1,
            "x": depot_coord["x"],
            "y": depot_coord["y"]
        })

        combined_demands.append({
            "id": 1,
            "demand": 0
        })

        new_id = 2

    # Skip depot from all files
    for coord_row, demand_row in zip(coords[1:], demands[1:]):
        combined_coords.append({
            "id": new_id,
            "x": coord_row["x"],
            "y": coord_row["y"]
        })

        combined_demands.append({
            "id": new_id,
            "demand": demand_row["demand"]
        })

        new_id += 1

output_500 = xml_dir / "XML500_1111_01.vrp"

with open(output_500, "w", encoding="utf-8") as f:
    f.write("NAME : XML500_1111_01\n")
    f.write("COMMENT : Combined from five XML100 benchmark files\n")
    f.write("TYPE : CVRP\n")
    f.write(f"DIMENSION : {len(combined_coords)}\n")
    f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
    f.write("CAPACITY : 20\n")
    f.write("NODE_COORD_SECTION\n")

    for row in combined_coords:
        f.write(f"{row['id']} {row['x']} {row['y']}\n")

    f.write("DEMAND_SECTION\n")

    for row in combined_demands:
        f.write(f"{row['id']} {row['demand']}\n")

    f.write("DEPOT_SECTION\n")
    f.write("1\n")
    f.write("-1\n")
    f.write("EOF\n")

print(f"Created {output_500}")
