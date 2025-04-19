import numpy as np
import matplotlib.pyplot as plt
from src.utils.model_import import OBJLoader

class CylindricalMapping:
    def __init__(self, control_vertices: np.ndarray):
        self.control_vertices = control_vertices
        self.mapping = self.compute_uv_mapping()
        self.num_u, self.num_v = 17, 11
        self.res_u, self.res_v = self.num_u * 5, self.num_v * 5

    def compute_uv_mapping(self):
        y_min = np.min(self.control_vertices[:, 1])
        y_max = np.max(self.control_vertices[:, 1])
        y_range = y_max - y_min if y_max != y_min else 1.0
        mapping = []
        for i in range(len(self.control_vertices)):
            x, y, z = self.control_vertices[i]
            theta = np.arctan2(z, x)  # [-pi, pi]
            u_val = (theta + np.pi) / (2 * np.pi)  # Normalize to [0, 1]
            v_val = (y - y_min) / y_range
            mapping.append([u_val, v_val])
            # print(i, u_val, v_val)
        return np.array(mapping)

    def compute_grid_shape(self):
        u_values = self.mapping[:, 0]
        v_values = self.mapping[:, 1]
        unique_u = np.unique(np.round(u_values, decimals=3))
        unique_v = np.unique(np.round(v_values, decimals=3))
        return unique_u.shape[0], unique_v.shape[0]