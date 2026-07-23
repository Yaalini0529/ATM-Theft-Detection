from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train

root = Path(__file__).resolve().parents[1]
output = train.prepare_dataset(root / "dataset")
print(output)
print("train labels", len(list((root / "dataset" / "train" / "labels").glob("*.txt"))))
print("valid labels", len(list((root / "dataset" / "valid" / "labels").glob("*.txt"))))
print("train images", len(list((root / "dataset" / "train" / "images").glob("*.png"))))
print("valid images", len(list((root / "dataset" / "valid" / "images").glob("*.png"))))
