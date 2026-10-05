"""
Surface2Anatomy Example: Single Organ Localization (Spleen).
"""

from surface2anatomy import SurfaceAnatomyModel

def main():
    print("Loading pretrained Surface2Anatomy ensemble...")
    model = SurfaceAnatomyModel.from_pretrained()

    surface_path = "patient_surface.ply"
    print(f"Predicting spleen centroid from {surface_path}...")
    try:
        result = model.predict(surface_path, target="spleen", units="mm")
        print(f"\nSpleen Centroid (Canonical mm): {result.centroid_mm}")
        print(f"Seed Disagreement: {result['spleen'].ensemble_disagreement_mm:.2f} mm")
        print(f"Sampled Points: {result.preprocessing.sampled_point_count}")
    except FileNotFoundError:
        print("Note: Provide a valid 3D surface file (.ply, .obj, .stl, etc.)")

if __name__ == "__main__":
    main()
