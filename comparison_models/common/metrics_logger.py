import csv
import os


def save_metrics(output_csv, row_dict):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    existing_rows = []
    fieldnames = list(row_dict.keys())

    if os.path.exists(output_csv):
        with open(output_csv, "r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames:
                for name in reader.fieldnames:
                    if name not in fieldnames:
                        fieldnames.append(name)
            existing_rows = list(reader)

    for name in row_dict.keys():
        if name not in fieldnames:
            fieldnames.append(name)

    dedupe_keys = [key for key in ("dataset", "model", "seed") if row_dict.get(key) is not None]
    if dedupe_keys:
        existing_rows = [
            row for row in existing_rows
            if any(str(row.get(key, "")) != str(row_dict.get(key, "")) for key in dedupe_keys)
        ]

    existing_rows.append(row_dict)

    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)
