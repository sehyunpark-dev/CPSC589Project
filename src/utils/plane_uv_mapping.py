import numpy as np

class ParametricMapping:
    def __init__(self, control_vertices: np.ndarray):
        self.control_vertices = control_vertices

        self.x_min = np.min(control_vertices[:, 0])
        self.x_max = np.max(control_vertices[:, 0])
        self.z_min = np.min(control_vertices[:, 2])
        self.z_max = np.max(control_vertices[:, 2])

        self.mapping = self.compute_uv_mapping()
        self.num_u, self.num_v = self.compute_grid_shape()

    def compute_uv_mapping(self):
        mapping = []
        for i, vtx in enumerate(self.control_vertices):
            x, z = vtx[0], vtx[2]
            u_val = (x - self.x_min) / (self.x_max - self.x_min) if self.x_max != self.x_min else 0.0
            v_val = (z - self.z_min) / (self.z_max - self.z_min) if self.z_max != self.z_min else 0.0
            mapping.append([u_val, v_val])
        mapping_np = np.array(mapping)
        return mapping_np

    def compute_grid_shape(self):
        u_values = self.mapping[:, 0]
        v_values = self.mapping[:, 1]
        unique_u = np.unique(u_values)
        unique_v = np.unique(v_values)
        num_u = unique_u.shape[0]
        num_v = unique_v.shape[0]

        return num_u, num_v