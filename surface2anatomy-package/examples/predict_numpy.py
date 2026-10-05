"""
Surface2Anatomy Example: In-Memory NumPy Coordinate Inference.
"""

import numpy as np
from surface2anatomy import SurfaceAnatomyModel

def main():
    model = SurfaceAnatomyModel.from_pretrained()

    # Synthetic patient surface point cloud in millimeters (N x 3)
    points = np.random.uniform(-180.0, 180.0, (5000, 3)).astype(np.float32)
    points[:, 2] = np.random.uniform(-400.0, 400.0, 5000)

    print("Running inference directly on NumPy array...")
    result = model.predict(points, target="liver", units="mm")
    print(f"Predicted Liver Centroid: {result.centroid_mm}")

if __name__ == "__main__":
    main()
