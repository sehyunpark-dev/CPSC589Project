import taichi as ti
import numpy as np
import trimesh
import pyquaternion as pyq
from pyquaternion import Quaternion

dir_path = "../models/"

class OBJLoader:
    def __init__(self,
                 file_name: str,
                 translation=np.array([0,0,0], dtype=float),
                 rotation_axis=np.array([0,0,1], dtype=float),
                 rotation_degree=0.0, # degree
                 scale=np.array([1,1,1], dtype=float)):
        ti.init(arch=ti.gpu)

        #######################################################################
        # Declaring member variables

        self.file_name = dir_path + file_name + ".obj"

        # Numpy and mesh variables
        self.mesh = None
        self.vertices_np = None
        self.edges_np = None
        self.faces_np = None
        self.num_vertices = 0
        self.num_edges = 0
        self.num_faces = 0

        # Transform
        self.translation = translation
        self.rotation_axis = rotation_axis / np.linalg.norm(rotation_axis) # normalize
        self.rotation_radian = np.radians(rotation_degree)
        self.scale = scale

        # Taichi variables
        self.ti_vertices = None
        self.ti_edges = None
        self.ti_faces = None

        #######################################################################
        # Applying functions

        self.load_obj()
        self.fill_taichi_fields()

    ###########################################################################
    # Class functions

    def load_obj(self):
        try:
            self.mesh = trimesh.load_mesh(self.file_name)
            self.vertices_np = self.mesh.vertices
            self.faces_np = self.mesh.faces
            self.edges_np = self.mesh.edges_unique

        except Exception as e:
            print(f"An error occurred while trying to load the model:, {e}")
            return False

        try:
            # Apply transformation to the vertices
            self.vertices_np = self.vertices_np * self.scale

            q = Quaternion(axis=self.rotation_axis, angle=self.rotation_radian)
            rot_mat = q.rotation_matrix
            self.vertices_np = self.vertices_np @ rot_mat.T

            self.vertices_np += self.translation

        except Exception as e:
            print(f"An error occurred while trying to apply transformation to the model: {e}")

    def fill_taichi_fields(self):
        if self.mesh is None or self.vertices_np is None or self.edges_np is None:
            print("Error: One of the mesh variables are empty.")
            return False

        try:
            self.num_vertices = self.vertices_np.shape[0]
            self.num_edges = self.edges_np.shape[0]
            self.num_faces = self.faces_np.shape[0]

            self.ti_vertices = ti.Vector.field(3, dtype=ti.f32, shape=self.num_vertices)
            self.ti_edges = ti.Vector.field(2, dtype=ti.i32, shape=self.num_edges)
            self.ti_faces = ti.Vector.field(3, dtype=ti.i32, shape=self.num_faces)

            self.ti_vertices.from_numpy(self.vertices_np.astype(np.float32))
            self.ti_edges.from_numpy(self.edges_np.astype(np.float32))
            self.ti_faces.from_numpy(self.faces_np.astype(np.float32))

        except Exception as e:
            print(f"Error filling Taichi fields: {e}")
            return False

    ###########################################################################
    # Debug functions

    def debug_info(self, print_data=False, num_samples=5):
        print("========== Debug Info ==========")
        print(f"File Name: {self.file_name}")
        if self.mesh is None:
            print("[Error] Mesh is not loaded.")
            return

        print(f"Mesh loaded: vertices={self.mesh.vertices.shape}, "
              f"faces={self.mesh.faces.shape}, edges={self.mesh.edges_unique.shape}")

        print("----- Numpy Arrays -----")
        if self.vertices_np is not None:
            print(f"vertices_np.shape = {self.vertices_np.shape}, dtype = {self.vertices_np.dtype}")
        else:
            print("[Error] vertices_np is None")

        if self.edges_np is not None:
            print(f"edges_np.shape    = {self.edges_np.shape}, dtype = {self.edges_np.dtype}")
        else:
            print("[Error] edges_np is None")

        if self.faces_np is not None:
            print(f"faces_np.shape    = {self.faces_np.shape}, dtype = {self.faces_np.dtype}")
        else:
            print("[Error] faces_np is None")

        print("----- Taichi Fields -----")
        if self.ti_vertices is not None:
            print(f"ti_vertices.shape = {self.ti_vertices.shape}, dtype = {self.ti_vertices.dtype}")
        else:
            print("[Error] ti_vertices is None")

        if self.ti_edges is not None:
            print(f"ti_edges.shape    = {self.ti_edges.shape}, dtype = {self.ti_edges.dtype}")
        else:
            print("[Error] ti_edges is None")

        if self.ti_faces is not None:
            print(f"ti_faces.shape    = {self.ti_faces.shape}, dtype = {self.ti_faces.dtype}")
        else:
            print("[Error] ti_faces is None")

        if print_data and self.vertices_np is not None and self.ti_vertices is not None:
            n_samples = min(num_samples, self.num_vertices)
            print(f"----- First {n_samples} vertices (Numpy) -----")
            print(self.vertices_np[:n_samples])
            print(f"----- First {n_samples} vertices (Taichi) -----")
            print(self.ti_vertices.to_numpy()[:n_samples])

        if print_data and self.edges_np is not None and self.ti_edges is not None:
            n_samples = min(num_samples, self.num_edges)
            print(f"----- First {n_samples} edges (Numpy) -----")
            print(self.edges_np[:n_samples])
            print(f"----- First {n_samples} edges (Taichi) -----")
            print(self.ti_edges.to_numpy()[:n_samples])

        if print_data and self.faces_np is not None and self.ti_faces is not None:
            n_samples = min(num_samples, self.num_faces)
            print(f"----- First {n_samples} faces (Numpy) -----")
            print(self.faces_np[:n_samples])
            print(f"----- First {n_samples} faces (Taichi) -----")
            print(self.ti_faces.to_numpy()[:n_samples])

# loader = OBJLoader("plane_8", rotation_degree=45.0)
# loader.debug_info(print_data=True, num_samples=3)