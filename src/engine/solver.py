import taichi as ti
import numpy as np
import os

@ti.data_oriented
class XPBDSolver:
    def __init__(self, simulator,
                 num_substeps: int):
        self.simulator = simulator
        self.num_substeps = num_substeps
        self.dt_sub = self.simulator.dt / self.num_substeps
        self.residual_history = []

    def apply_constraints(self, stretch_stiffness,  dt_sub):
        compliance_stretch = stretch_stiffness * dt_sub * dt_sub

        # Gauss-Seidel
        for _ in range(self.num_substeps):
            self.solve_distance_constraints(compliance_stretch)

    def record_residual(self):
        r = self.compute_constraint_residual()
        self.residual_history.append(r)

    def save_residual_history(self):
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)

        model_name = os.path.basename(self.simulator.mesh.file_name).replace(".obj", "")

        filename = f"residual_history_{model_name}.npy"
        filepath = os.path.join(results_dir, filename)

        np.save(filepath, np.array(self.residual_history))
        print(f"[XPBD] Residual history saved to {filepath}")

    @ti.kernel
    def solve_distance_constraints(self, compliance_stretch: ti.f32, compliance_bending: ti.f32):
        ti.loop_config(serialize=True)
        for i in range(self.simulator.num_edges):
            l0 = self.simulator.l0[i]
            v0, v1 = self.simulator.ti_edges[i][0], self.simulator.ti_edges[i][1]
            x01 = self.simulator.x_tilde[v0] - self.simulator.x_tilde[v1]
            lij = x01.norm()

            C = (lij - l0)
            nabla_C = x01.normalized()
            schur = (self.simulator.fixed[v0] * self.simulator.m_inv[v0] +
                     self.simulator.fixed[v1] * self.simulator.m_inv[v1])

            ld = compliance_stretch * C / (compliance_stretch * schur + 1.0)

            self.simulator.x_tilde[v0] -= self.simulator.fixed[v0] * self.simulator.m_inv[v0] * ld * nabla_C
            self.simulator.x_tilde[v1] += self.simulator.fixed[v1] * self.simulator.m_inv[v1] * ld * nabla_C

    @ti.kernel
    def compute_constraint_residual(self) -> ti.f32:
        total_residual = 0.0
        for i in range(self.simulator.num_edges):
            v0, v1 = self.simulator.ti_edges[i][0], self.simulator.ti_edges[i][1]
            x01 = self.simulator.x_tilde[v0] - self.simulator.x_tilde[v1]
            lij = x01.norm()
            diff = lij - self.simulator.l0[i]
            total_residual += diff * diff
        return total_residual