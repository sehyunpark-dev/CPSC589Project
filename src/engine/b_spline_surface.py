import taichi as ti
import numpy as np

MAX_ORDER = 10 # <= order_u, order_v

@ti.data_oriented
class BSplineSurface:
    def __init__(self,
                 control_vertices: np.ndarray,  # shape=(num_vertices,3)
                 uv_mapping: np.ndarray,  # shape=(num_vertices,2)
                 num_u: int, num_v: int,
                 res_u: int, res_v: int,
                 order_u: int = 5, order_v: int = 5,
                 is_cylinder=False):
        """
          - control_vertices: numpy array, shape=(num_vertices, 3)
          - uv_mapping: numpy array, shape=(num_vertices, 2) (u,v) value of each control point
          - num_u, num_v: the grid size of u, v (num_vertices == num_u*num_v)
          - res_u, res_v: resolution of postprocessed surface
          - order_u, order_v: B-spline order (Cubic = 4, Quadratic = 3, ...)
        """
        self.control_vertices_init = control_vertices
        self.control_vertices = control_vertices
        self.uv_mapping = uv_mapping
        self.num_u = num_u
        self.num_v = num_v
        self.res_u = res_u
        self.res_v = res_v
        self.order_u = order_u
        self.order_v = order_v
        self.is_cylinder = is_cylinder

        self.num_control_vertices = control_vertices.shape[0]
        self.m_u = num_u - 1
        self.m_v = num_v - 1

        print("[B-spline] Initializing B-spline Surface...")
        self.control_net_np = None
        self.control_net_field = None
        if self.is_cylinder:
            self.control_net_np = np.zeros(shape=((self.num_u + self.order_u - 1) * self.num_v, 3), dtype=np.float32)
            self.control_net_field = ti.Vector.field(3, dtype=ti.f32, shape=(self.num_u + self.order_u - 1) * self.num_v)
        else:
            self.control_net_np = np.zeros(shape=(self.num_u * self.num_v, 3), dtype=np.float32)
            self.control_net_field = ti.Vector.field(3, dtype=ti.f32, shape=self.num_u * self.num_v)
        self.reorder_control_net_np()

        # 2. Generate Knot vector (NumPy)
        if self.is_cylinder:
            self.U_np = self.make_knot_vector_np(self.num_u, self.order_u, periodic=True)
        else:
            self.U_np = self.make_knot_vector_np(self.num_u, self.order_u, periodic=False)
        self.V_np = self.make_knot_vector_np(self.num_v, self.order_v)
        self.num_U_knot = len(self.U_np)
        self.num_V_knot = len(self.V_np)

        # 3. Make new face indices by u,v order
        self.surface_faces_np = None
        self.surface_faces_field = None
        self.make_faces_np()

        # Knot vector fields
        self.U = ti.field(dtype=ti.f32, shape=self.num_U_knot)
        self.V = ti.field(dtype=ti.f32, shape=self.num_V_knot)
        self.U.from_numpy(self.U_np)
        self.V.from_numpy(self.V_np)

        # 4. Evaluate surface
        self.surface_points_field = ti.Vector.field(3, dtype=ti.f32, shape=(self.res_u * self.res_v))
        self.evaluate_surface_wrapper(self.control_vertices)

        print("[B-spline] Initialization Done.\n")

    ###########################################################################
    # Numpy class functions

    def reorder_control_net_np(self):
        if self.is_cylinder:
            for i in range(len(self.control_vertices)):
                u_val, v_val = self.uv_mapping[i]
                row = int(round(u_val * (self.num_u - 1)))
                col = int(round(v_val * (self.num_v - 1)))
                self.control_net_np[row * self.num_v + col, :] = self.control_vertices[i]
                # print(f"{row} * {self.num_v} + {col} = {row * self.num_v + col}")
            # print("1-------")
            # Copy points with u=0 from the points with u=1 (original)
            for i in range(self.num_v):
                first_index = 0 * self.num_v + i
                last_index = (self.num_u - 1) * self.num_v + i
                self.control_net_np[first_index, :] = self.control_net_np[last_index, :]
                # print(f"{first_index} <- {self.num_u - 1} * {self.num_v} + {i} = {last_index}")
            # print("2-------")
            for i in range(1, self.order_u):
                for j in range(self.num_v):
                    src_idx = i * self.num_v + j
                    dst_idx = (self.num_u - 1 + i) * self.num_v + j
                    self.control_net_np[dst_idx, :] = self.control_net_np[src_idx, :]
                    # print(f"{self.num_u - 1 + i} * {self.num_v} + {j} = {dst_idx} <- {src_idx} = {i} * {self.num_v} + {j}")

            # print(self.control_net_np)
        else:
            for i in range(len(self.control_vertices)):
                u_val, v_val = self.uv_mapping[i]
                row = int(round(u_val * (self.num_u - 1)))
                col = int(round(v_val * (self.num_v - 1)))
                self.control_net_np[row * self.num_v + col, :] = self.control_vertices[i]

        self.control_net_field.from_numpy(self.control_net_np)

    def make_knot_vector_np(self, n_ctrl: int, order: int, periodic: bool=False) -> np.ndarray:
        if periodic:
            L = n_ctrl + 2 * order - 1
            # print(L)
            knots = np.zeros(L, dtype=np.float32)
            knots[0] = 0.0
            knots[-1] = 1.0
            for i in range(1, L-1):
                knots[i] = i / (L-1)
            # print(knots)
            return knots
        else:
            L = n_ctrl + order
            knots = np.zeros(L, dtype=np.float32)
            knots[:order] = 0.0
            knots[-order:] = 1.0
            if L - 2 * order > 0:
                for i in range(order, L - order):
                    knots[i] = (i - order + 1) / (L - 2 * order + 1)
        return knots

    def make_faces_np(self):
        faces = []
        if self.is_cylinder:
            first_range = int(round(((self.order_u - 1) / (self.num_U_knot - 1)) * self.res_u)) - 1
            last_range = int(round(((self.num_u + self.order_u - 2) / (self.num_U_knot - 1)) * self.res_u))
            # print(first_range, last_range)
        else:
            first_range = 0
            last_range = self.res_u - 1

        for i in range(first_range, last_range):
            i_next = (i + 1) % self.res_u
            for j in range(self.res_v - 1):
                # Vertex indices in the flattened 1D array
                v0 = i * self.res_v + j
                v1 = i_next * self.res_v + j
                v2 = i_next * self.res_v + (j + 1)
                v3 = i * self.res_v + (j + 1)

                # Add two triangles per quad
                faces.append([v0, v1, v2])
                faces.append([v0, v2, v3])

        self.surface_faces_np = np.array(faces, dtype=np.int32).reshape(-1)  # Flattened
        self.surface_faces_field = ti.field(dtype=ti.i32, shape=len(self.surface_faces_np))
        self.surface_faces_field.from_numpy(self.surface_faces_np)

    def evaluate_surface_wrapper(self, control_vertices: np.ndarray):
        self.control_vertices = control_vertices
        self.reorder_control_net_np()
        self.evaluate_surface()

    def reset(self):
        self.evaluate_surface_wrapper(self.control_vertices_init)

    ###########################################################################
    # Taichi class functions

    @ti.kernel
    def evaluate_surface(self):
        for idx in range(self.res_u * self.res_v):
            # Convert flat index to 2D (i,j)
            i = idx // self.res_v
            j = idx % self.res_v
            # (u, v) in [0, 1]
            u = ti.cast(i, ti.f32) / ti.cast(self.res_u - 1, ti.f32)
            v = ti.cast(j, ti.f32) / ti.cast(self.res_v - 1, ti.f32)

            self.surface_points_field[idx] = self.de_boor_surface(u, v)
            # print(i, j, u, v, idx, self.surface_points_field[idx])

    @ti.func
    def find_knot_index_u(self, u: ti.f32) -> ti.i32:
        d = self.order_u - 1
        if u == 0.0:
            d = self.order_u - 1
        elif u == 1.0:
            d = self.num_U_knot - self.order_u
        else:
            for i in range(self.order_u, self.num_U_knot - self.order_u):
                if self.U[i] <= u < self.U[i + 1]:
                    d = i
        return d

    @ti.func
    def find_knot_index_u_periodic(self, u: ti.f32) -> ti.i32:
        d = self.order_u - 1
        for i in range(self.order_u - 1, self.num_u + self.order_u - 2):
            if self.U[i] <= u < self.U[i + 1]:
                d = i
        if self.U[self.num_u + self.order_u - 2] <= u:
            d = self.num_u + self.order_u - 2
        # print(u, d)
        return d

    @ti.func
    def find_knot_index_v(self, v: ti.f32) -> ti.i32:
        d = self.order_v - 1
        if v == 0.0:
            d = self.order_v - 1
        elif v == 1.0:
            d = self.num_V_knot - self.order_v
        else:
            for i in range(self.order_v, self.num_V_knot - self.order_v):
                if self.V[i] <= v < self.V[i + 1]:
                    d = i
        return d

    @ti.func
    def de_boor_surface(self, u: ti.f32, v: ti.f32) -> ti.math.vec3:
        d_u = 0
        if self.is_cylinder:
            u_min = self.U[self.order_u - 1]
            u_max = self.U[self.num_u + self.order_u - 2]
            u = ti.max(u_min, ti.min(u, u_max))
            d_u = self.find_knot_index_u_periodic(u)
            # print(u_min, u_max, self.order_u - 1, self.num_u + self.order_u - 2, u, d_u)
        else:
            d_u = self.find_knot_index_u(u)
        d_v = self.find_knot_index_v(v)

        # Temporary matrix C to hold intermediate results after v-direction
        C = ti.Matrix.zero(ti.f32, MAX_ORDER, 3)

        for i in range(self.order_u):  # u-direction
            row_idx = d_u - i

            # D: intermediate control points in v-direction
            D = ti.Matrix.zero(ti.f32, MAX_ORDER, 3)
            for j in range(self.order_v):  # v-direction
                col_idx = d_v - j
                idx = row_idx * self.num_v + col_idx
                for k in ti.static(range(3)):
                    D[j, k] = self.control_net_field[idx][k]

            # v-direction de Boor
            for r_offset in range(self.order_v - 2 + 1):  # r = order_v down to 2
                r = self.order_v - r_offset
                p = d_v
                for s in range(r - 1):
                    denom = self.V[p + r - 1] - self.V[p]
                    omega = (v - self.V[p]) / denom if denom > 1e-6 else 0.0
                    for k in ti.static(range(3)):
                        D[s, k] = omega * D[s, k] + (1.0 - omega) * D[s + 1, k]
                    p -= 1

            # Store result of v-direction curve
            for k in ti.static(range(3)):
                C[i, k] = D[0, k]

        # u-direction de Boor
        for r_offset in range(self.order_u - 2 + 1):  # r = order_u down to 2
            r = self.order_u - r_offset
            p = d_u
            for s in range(r - 1):
                denom = self.U[p + r - 1] - self.U[p]
                omega = (u - self.U[p]) / denom if denom > 1e-6 else 0.0
                for k in ti.static(range(3)):
                    C[s, k] = omega * C[s, k] + (1.0 - omega) * C[s + 1, k]
                p -= 1

        return ti.Vector([C[0, k] for k in ti.static(range(3))])
