import numpy as np
import open3d as o3d

heightmap = np.load("generated_terrain/moon_terrain.npy")
rows, cols = heightmap.shape
X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
Z = heightmap

vertices = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
faces = []
for i in range(rows - 1):
    for j in range(cols - 1):
        v0 = i * cols + j
        v1 = v0 + 1
        v2 = v0 + cols
        v3 = v2 + 1
        faces.append([v0, v2, v1])
        faces.append([v1, v2, v3])

mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(vertices)
mesh.triangles = o3d.utility.Vector3iVector(faces)
mesh.compute_vertex_normals()

o3d.io.write_triangle_mesh("moon_terrain.obj", mesh)
print("✅ Exported terrain to moon_terrain.obj")