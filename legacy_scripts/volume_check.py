# volume_check.py
import glob, json, csv
for m in glob.glob("../Benchmark dataset and instance generator for Real-World 3dBPP/Output/merged/*_merged.json"):
    d=json.load(open(m))
    boxes_csv = "../Benchmark dataset and instance generator for Real-World 3dBPP/Output/merged/" + d['boxes_file']
    vols = 0
    with open(boxes_csv) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            vols += float(r['length'])*float(r['width'])*float(r['height'])
    cont = d.get('container', {'L':100,'W':100,'H':100})
    cont_vol = cont['L']*cont['W']*cont['H']
    print(m, "boxes:", sum(1 for _ in open(boxes_csv)) - 1, "total_box_vol:", int(vols), "container_vol:", int(cont_vol), "ratio:", round(vols/cont_vol,3))

