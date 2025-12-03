# clean_boxes_save.py
from glob import glob
from merge_cvrp_and_boxes import read_box_file
import csv, os

src_files = glob("../Benchmark dataset and instance generator for Real-World 3dBPP/Input/3dBPP_*.txt")
outdir = "../Benchmark dataset and instance generator for Real-World 3dBPP/Input/cleaned"
os.makedirs(outdir, exist_ok=True)

for f in src_files:
    boxes = read_box_file(f)
    # filter out boxes with any non-positive dimension
    boxes = [b for b in boxes if b['l'] > 0 and b['w'] > 0 and b['h'] > 0]
    outpath = os.path.join(outdir, os.path.basename(f).replace('.txt', '.csv'))
    with open(outpath, 'w', newline='') as of:
        writer = csv.DictWriter(of, fieldnames=['box_id','length','width','height','weight'])
        writer.writeheader()
        for b in boxes:
            writer.writerow({'box_id': b['id'], 'length': b['l'], 'width': b['w'], 'height': b['h'], 'weight': b.get('weight')})
    print("Saved cleaned:", outpath, "boxes:", len(boxes))
