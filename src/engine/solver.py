import taichi as ti

@ti.data_oriented
class XPBDSolver:
    def __init__(self, simulator,
                 num_substeps: int):
        self.simulator = simulator
        self.num_substeps = num_substeps
        self.dt_sub = self.simulator.dt / self.num_substeps

    def apply_constraints(self, stretch_stiffness, bending_stiffness, dt_sub):
        compliance_stretch = stretch_stiffness * dt_sub * dt_sub
        compliance_bending = bending_stiffness * dt_sub * dt_sub

        # Gauss-Seidel
        for _ in range(self.num_substeps):
            self.solve_distance_constraints(compliance_stretch, compliance_bending)

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