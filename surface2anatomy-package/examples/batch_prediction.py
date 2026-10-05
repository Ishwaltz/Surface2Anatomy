"""
Surface2Anatomy Example: High-Throughput Cohort Batch Prediction.
"""

from pathlib import Path
from surface2anatomy import SurfaceAnatomyModel

def main():
    model = SurfaceAnatomyModel.from_pretrained()

    surfaces_dir = Path("surfaces_cohort/")
    surface_files = list(surfaces_dir.glob("*.ply"))
    print(f"Processing cohort of {len(surface_files)} surfaces...")

    results = model.predict_batch(
        surface_files,
        targets=["liver", "spleen", "kidney_left", "kidney_right"]
    )

    for f, res in zip(surface_files, results):
        print(f"Subject {f.name}: Liver={res['liver'].centroid_mm}")

if __name__ == "__main__":
    main()
