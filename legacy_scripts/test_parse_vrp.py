# test_parse_vrp.py
from glob import glob
from merge_cvrp_and_boxes import parse_cvrp_xml

files = glob("../XML/*.vrp")[:10]
print("Testing up to", len(files), "VRP files (first 10):")
for f in files:
    try:
        data = parse_cvrp_xml(f)
        print(f"\n{f} -> customers:", len(data['customers']), " sample:", data['customers'][:1])
    except Exception as e:
        print("\nFAILED parse:", f, e)

