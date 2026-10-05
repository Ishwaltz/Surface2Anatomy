# Supported Input Formats

Surface2Anatomy accepts external 3D surface geometry:
- **Point Clouds**: `.ply`, `.pcd`, `.xyz`, `.txt`, `.npy`
- **Surface Meshes**: `.ply`, `.obj`, `.stl`

### Rejecting 2D Photographs
Standard 2D RGB photographs (`.jpg`, `.png`, `.jpeg`) are explicitly rejected with:
`The current Surface2Anatomy model requires 3D surface geometry or depth-derived point clouds. A standard 2D RGB photograph is not currently a supported model input.`
