from run_experiments import load_instance

name, container, boxes = load_instance(
    "../Benchmark dataset and instance generator for Real-World 3dBPP/Output/merged/XML100_1111_01_merged.json"
)

print("Container:", container)
print("First 5 boxes:")
for b in boxes[:5]:
    print(b)
