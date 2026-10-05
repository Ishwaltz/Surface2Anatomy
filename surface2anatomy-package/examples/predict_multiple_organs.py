"""
Surface2Anatomy Example: Simultaneous Multi-Organ Query in a Single Forward Pass.
"""

from surface2anatomy import SurfaceAnatomyModel

def main():
    model = SurfaceAnatomyModel.from_pretrained()

    targets = ["liver", "spleen", "kidney_left", "kidney_right", "heart", "brain"]
    print(f"Querying {len(targets)} internal organs simultaneously...")

    result = model.predict_multiple("patient_surface.ply", targets=targets, units="mm")

    print(f"\nCompleted in {result.total_latency_ms:.1f} ms:")
    for target_name, pred in result.items():
        print(f"  {target_name:15s} -> Centroid: {pred.centroid_mm} mm (±{pred.ensemble_disagreement_mm:.1f} mm)")

if __name__ == "__main__":
    main()
