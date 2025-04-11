import taichi as ti
import numpy as np
from fontTools.subset.svg import xpath
from fontTools.ttLib.tables.E_B_D_T_ import ebdt_bitmap_format_5

from ..utils.model_import import OBJLoader
from ..engine.solver import XPBDSolver

@ti.data_oriented
class ClothSimulator:
    def __init__(self,
                 mesh: OBJLoader, dt=1.0 / 60.0,
                 gravity=ti.math.vec3(0.0, -9.8, 0.0),
                 stretch_stiffness=5e5, bending_stiffness=5e5,
                 num_substeps=20):
        print("[Simulator] Initializing cloth simulator...")

        ######################################################################
        # [Simulation Parameters]
        # - Mesh(Model), Time step, Gravity, etc.
        ######################################################################
        self.mesh = mesh
        self.dt = dt
        self.gravity = gravity
        self.sim_frame = 0

        #######################################################################
        # [Initialization Data]
        # - Mesh or geometry information needed for field allocation and setup.
        #######################################################################
        self.ti_vertices = None
        self.ti_edges = None
        self.ti_faces = None
        self.ti_faces_flatten = None
        self.num_vertices = 0
        self.num_edges = 0
        self.num_faces = 0

        self.fill_taichi_fields()

        #######################################################################
        # [Simulation State Variables]
        # - Fields that change during simulation (positions, velocities, etc).
        #######################################################################
        # for vertices
        self.x0 = None
        self.x_cur = None
        self.x_tilde = None
        self.v = None
        self.dx = None
        self.dv = None
        self.nc = None
        self.fixed = None   # 1.0 if the vertex can move, 0.0 if not
        self.m_inv = None   # m_inv = 1/m

        # for edges
        self.l0 = None

        # for wind
        self.enable_wind = False
        self.wind_strength = 50.0

        #######################################################################

        self.init_simulation_variables()

        print("[Simulator] Initialization done.")

        self.stretch_stiffness = stretch_stiffness
        self.bending_stiffness = bending_stiffness
        self.num_substeps = num_substeps

        self.xpbd_solver = XPBDSolver(self, self.num_substeps)

    ###########################################################################
    # Class functions

    def fill_taichi_fields(self):
        if self.mesh is None:
            print("Error: the mesh is empty.")
            return False

        try:
            self.num_vertices = self.mesh.vertices_np.shape[0]
            self.num_edges = self.mesh.edges_np.shape[0]
            self.num_faces = self.mesh.faces_np.shape[0]

            self.ti_vertices = ti.Vector.field(3, dtype=ti.f32, shape=self.num_vertices)
            self.ti_edges = ti.Vector.field(2, dtype=ti.i32, shape=self.num_edges)
            self.ti_faces = ti.Vector.field(3, dtype=ti.i32, shape=self.num_faces)
            self.ti_faces_flatten = ti.field(dtype=ti.i32, shape=3 * self.num_faces)

            self.ti_vertices.from_numpy(self.mesh.vertices_np.astype(np.float32))
            self.ti_edges.from_numpy(self.mesh.edges_np.astype(np.int32))
            self.ti_faces.from_numpy(self.mesh.faces_np.astype(np.int32))
            self.ti_faces_flatten.from_numpy(self.mesh.faces_np.flatten().astype(np.int32))
            return True

        except Exception as e:
            print(f"Error filling Taichi fields: {e}")
            return False

    def init_simulation_variables(self):
        try:
            self.x0 = ti.Vector.field(3, dtype=ti.f32, shape=self.num_vertices)
            self.x_cur = ti.Vector.field(3, dtype=ti.f32, shape=self.num_vertices)
            self.x_tilde = ti.Vector.field(3, dtype=ti.f32, shape=self.num_vertices)
            self.v = ti.Vector.field(3, dtype=ti.f32, shape=self.num_vertices)
            self.dx = ti.Vector.field(3, dtype=ti.f32, shape=self.num_vertices)
            self.dv = ti.Vector.field(3, dtype=ti.f32, shape=self.num_vertices)
            self.nc = ti.field(dtype=ti.f32, shape=self.num_vertices)
            self.fixed = ti.field(dtype=ti.f32, shape=self.num_vertices)
            self.m_inv = ti.field(dtype=ti.f32, shape=self.num_vertices)

            self.l0 = ti.field(dtype=ti.f32, shape=self.num_edges)

            self.x0.copy_from(self.ti_vertices)
            self.x_cur.copy_from(self.x0)
            self.x_tilde.copy_from(self.x0)
            self.v.fill(0.0)
            self.dx.fill(0.0)
            self.dv.fill(0.0)
            self.nc.fill(0.0)
            self.fixed.fill(1.0)
            self.m_inv.fill(0.0)

            self.l0.fill(0.0)

            self.init_m_inv_l0()
        except Exception as e:
            print(f"Error initializing Simulation variables: {e}")

    def step(self):
        # XPBD-Based Cloth Simulation
        self.predict_x_tilde()
        if self.enable_wind:
            self.apply_wind()
        self.xpbd_solver.apply_constraints(self.stretch_stiffness, self.bending_stiffness, self.num_substeps)
        self.compute_v()
        self.update_x()
        self.sim_frame += 1

    def reset(self):
        self.x_cur.copy_from(self.x0)
        self.x_tilde.copy_from(self.x0)
        self.v.fill(0.0)
        self.dx.fill(0.0)
        self.dv.fill(0.0)
        self.nc.fill(0.0)
        self.sim_frame = 0

    ###########################################################################
    # Kernel functions

    @ti.kernel
    def init_m_inv_l0(self):
        for i in range(self.num_edges):
            v0, v1 = self.ti_edges[i][0], self.ti_edges[i][1]
            self.l0[i] = (self.x0[v0] - self.x0[v1]).norm()
            self.m_inv[v0] += 0.5 * self.l0[i]
            self.m_inv[v1] += 0.5 * self.l0[i]

    @ti.kernel
    def predict_x_tilde(self):
        for i in range(self.num_vertices):
            self.x_tilde[i] = self.x_cur[i] + \
                              self.fixed[i] * (self.v[i] * self.dt + self.gravity * self.dt * self.dt)

    @ti.kernel
    def apply_wind(self):
        for i in range(self.num_vertices):
            if self.fixed[i] == 1.0:
                base_dir = ti.Vector([1.0, 0.2, 0.0]).normalized()

                angle_offset = 3.0 * (ti.random() - 0.5)
                axis = ti.Vector([0.0, 1.0, 0.0])

                # Rodrigues' rotation formula
                k = axis.normalized()
                cos_theta = ti.cos(angle_offset)
                sin_theta = ti.sin(angle_offset)

                wind_dir = base_dir * cos_theta + k.cross(base_dir) * sin_theta + \
                           k * k.dot(base_dir) * (1.0 - cos_theta)
                wind_dir = wind_dir.normalized()

                random_strength = self.wind_strength * (0.5 + ti.random())  # [0.5, 1.5] * self.wind_strength

                wind_force = wind_dir * random_strength
                self.x_tilde[i] += wind_force * self.dt * self.dt

    @ti.kernel
    def compute_v(self):
        # compute velocities after applying constraints to x_tilde
        for i in range(self.num_vertices):
            self.v[i] = self.fixed[i] * (self.x_tilde[i] - self.x_cur[i]) / self.dt

    @ti.kernel
    def update_x(self):
        for i in range(self.num_vertices):
            self.x_cur[i] += self.fixed[i] * self.v[i] * self.dt