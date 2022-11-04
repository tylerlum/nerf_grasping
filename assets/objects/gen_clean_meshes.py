import os
import trimesh

OBJ_DATA = [
    ("banana", 1.0),
    ("bleach_cleanser", 0.75),
    ("box", 0.075),
    ("mug", 1.0),
    ("power_drill", 1.0),
    ("spatula", 1.0),
    ("teddy_bear", 0.01),
]

if __name__ == "__main__":
    for (obj, scale) in OBJ_DATA:
        os.makedirs(f"meshes/{obj}", exist_ok=True)
        mesh = trimesh.load(f"raw_meshes/{obj}/textured.obj")
        mesh.apply_translation(-mesh.centroid)
        mesh.apply_scale(scale)

        mesh.export(f"meshes/{obj}/textured.obj")
