import taichi as ti
import numpy as np
import trimesh
import os
from pyquaternion import Quaternion

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "../../models")

@ti.data_oriented
class OBJLoader:
    def __init__(self,
                 file_name: str,
                 translation=np.array([0,0,0], dtype=float),
                 rotation_axis=np.array([0,0,1], dtype=float),
                 rotation_degree=0.0, # degree
                 scale=np.array([1,1,1], dtype=float)):

        print("[OBJLoader] Initializing OBJLoader...")
        self.file_name = os.path.join(MODEL_DIR, file_name + ".obj")

        # Numpy and mesh variables
        self.mesh = None
        self.vertices_np = None
        self.edges_np = None
        self.faces_np = None

        # Transform
        self.translation = translation
        self.rotation_axis = rotation_axis / np.linalg.norm(rotation_axis) # normalize
        self.rotation_radian = np.radians(rotation_degree)
        self.scale = scale

        self.load_obj()
        print("[OBJLoader] Finished initializing OBJLoader.\n")

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