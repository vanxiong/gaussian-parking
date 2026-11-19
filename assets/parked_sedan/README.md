# Parked sedan appearance asset

Source: [Innerscene Sedan car](https://www.innerscene.com/tools/library/3d-parts/sedan-car-42109739).
Original unbranded model released under **CC0 1.0**. The source page permits use,
modification and redistribution. Source download:
https://files.innerscene.com/library/sedan-car-42109739/75c64c1b3c8b03734314/sedan-original.glb

`sedan-original.glb` is the pinned source. `sedan.usda` is generated with
`tools/convert_parked_sedan.py`, preserving triangles, vertex normals, and material
factors. It is self-contained, in meters, with X forward, Y left, Z up.
`asset.json` records hashes, bounds and mesh counts. No external textures or
runtime downloads are required. Conversion needs Python and numpy only.

This is an **appearance-only** ablation: original parked-car shapes supply the
same collision/LiDAR and occupancy references in both scene variants. The detailed
mesh is visual-only, fitted to the original car's full bounding dimensions.
The shared proxy target is an approximation of the detailed surface, not an exact
mesh-ground-truth claim. The model still consumes both camera and LiDAR inputs.
