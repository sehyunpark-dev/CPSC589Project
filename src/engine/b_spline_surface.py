import taichi as ti
import numpy as np

@ti.data_oriented
class BSplineSurface:
    def __init__(self,
                 control_vertices: ti.template(),  # shape=(num_u * num_v,)
                 num_u: int, num_v: int,
                 res_u: int, res_v: int,
                 order_u: int = 4, order_v: int = 4):
        """
        control_vertices : cloth mesh의 저해상도 vertex field (num_u * num_v개)
        num_u, num_v     : control grid 사이즈
        res_u, res_v     : B-spline surface 평가 해상도
        order_u, order_v : B-spline의 order (4 = Cubic, 3 = Quadratic 등)
        """
        self.control_vertices = control_vertices
        self.num_u = num_u
        self.num_v = num_v
        self.res_u = res_u
        self.res_v = res_v
        self.order_u = order_u
        self.order_v = order_v

        # knot vector 구성 (표준 knot sequence)
        self.knot_u = self.make_knot_vector(num_u, order_u)  # length = num_u + order_u
        self.knot_v = self.make_knot_vector(num_v, order_v)

        # 최종 샘플된 표면점을 담을 필드
        self.surface_points = ti.Vector.field(3, dtype=ti.f32, shape=(self.res_u, self.res_v))

        # 표면을 즉시 평가
        self.evaluate_surface()

    def make_knot_vector(self, n_ctrl: int, k: int):
        """
        표준 knot sequence (clamped uniform)의 예:
        - knot vector 길이 = n_ctrl + k
        - 맨 앞, 맨 뒤를 k번 중복
        - 중간 구간은 uniform
        예) n_ctrl=4, k=4 (cubic): [0,0,0,0, 1/1, 1,1,1,1]
        """
        m = n_ctrl + k
        knots = np.zeros(m, dtype=np.float32)

        # 맨 앞 k개는 0
        for i in range(k):
            knots[i] = 0.0

        # 맨 뒤 k개는 1
        for i in range(m - k, m):
            knots[i] = 1.0

        # 중간은 균일 분배
        denom = (m - 2*k + 1)
        for i in range(k, m - k):
            knots[i] = (i - k) / denom

        return knots

    def evaluate_surface(self):
        """
        Taichi 커널을 통해 (res_u * res_v) 격자에서 (u, v)를 샘플링하고,
        각각 De Boor 알고리즘으로 표면점을 구함.
        """
        # Python에서 u, v 인자 범위 (0..1)로 res_u, res_v개를 샘플링
        # 커널 내부에선 index → u, v 변환
        self.eval_surface_kernel()

    @ti.kernel
    def eval_surface_kernel(self):
        for i, j in self.surface_points:
            # i in [0..res_u), j in [0..res_v)
            # (u, v) 범위: [0..1]
            u = ti.cast(i, ti.f32) / (self.res_u - 1)
            v = ti.cast(j, ti.f32) / (self.res_v - 1)

            # De Boor 알고리즘으로 (u, v)에 해당하는 3D 점 구함
            # (Python 함수 호출로 구현)
            point = BSplineSurface.de_boor_surface(
                self.control_vertices,
                self.num_u, self.num_v,
                self.knot_u, self.knot_v,
                self.order_u, self.order_v,
                u, v
            )
            self.surface_points[i, j] = point

    @staticmethod
    def de_boor_surface(ctrl_verts: ti.template(),
                        num_u: int, num_v: int,
                        U: ti.types.ndarray(),  # knot_u
                        V: ti.types.ndarray(),  # knot_v
                        k_u: int, k_v: int,
                        u: float, v: float) -> ti.math.vec3:
        """
        De Boor 알고리즘 (반복문 버전).
        ctrl_verts.shape = (num_u * num_v,)
        (u, v) in [0..1].
        U, V는 numpy knot vectors, 길이는 num_u + k_u, num_v + k_v

        반환: 3D point.
        """
        # 1) delta_U(u), delta_V(v) 계산: d_u, d_v
        d_u = BSplineSurface.find_knot_index(U, u, num_u, k_u)
        d_v = BSplineSurface.find_knot_index(V, v, num_v, k_v)

        # 2) 먼저 v방향으로 De Boor
        #    각 행(i)별로 1D B-spline curve (j방향)
        #    결과: C[i], i=[0..k_u-1]
        # -> 그 후 u방향 De Boor

        # Python에서는 ctrl_verts[i*num_v + j]로 접근해야 하나,
        # 여기선 ti.template()로 받아와 => ctrl_verts[idx] 로 가능

        # 'temp'는 (k_v,) 크기 Vector[3]를 여러 개(= k_u개) 관리
        # Python list로 처리
        row_points = []
        for i in range(k_u):
            row_points.append(ti.Vector([0.0, 0.0, 0.0]))

        # (k_u번 반복) -> 각 i행에 대해, v방향 De Boor
        for i in range(k_u):
            row_i = d_u - i  # control row index
            # j: [0..k_v], (v방향) -> slice
            # 임시 list(D) 크기 k_v
            D = []
            for jj in range(k_v):
                col_j = d_v - jj
                idx = row_i * num_v + col_j
                D.append(ctrl_verts[idx])  # Vector3

            # (bending) r in [k_v..2 step -1]
            # => de boor 1D
            for r in range(k_v, 1, -1):
                p = d_v
                for s in range(r - 1):
                    denom = (U[p + r - 1] - U[p])  # 여기선 사실 V[] 써야 함 => V...
                    # 그러나 pseudo code 상에서 v->p => we use V
                    # Actually, this part needs to refer to V; let's fix it:
                    denom = (V[p + r - 1] - V[p])
                    if denom < 1e-9:
                        omega = 0.0
                    else:
                        omega = (v - V[p]) / denom
                    D[s] = omega * D[s] + (1 - omega) * D[s + 1]
                p -= 1

            row_points[i] = D[0]

        # 이제 row_points[i] (i=0..k_u-1)에 대해 u방향 De Boor
        # 다시 list(C) = row_points[:]
        C = row_points[:]  # copy
        # do same process with U, d_u, k_u
        for r in range(k_u, 1, -1):
            p = d_u
            for s in range(r - 1):
                denom = (U[p + r - 1] - U[p])
                if denom < 1e-9:
                    omega = 0.0
                else:
                    omega = (u - U[p]) / denom
                C[s] = omega * C[s] + (1 - omega) * C[s + 1]
            p -= 1

        return C[0]

    @staticmethod
    def find_knot_index(knot: np.ndarray, param: float, n_ctrl: int, k: int) -> int:
        """
        delta_U(u): knot vector에서 param ∈ (knot[i], knot[i+1]) 만족하는 i 리턴
        단, param == 1.0일 때는 가장 마지막(= n_ctrl + k - 2)쪽 처리를 고려.
        """
        m = n_ctrl + k
        if param >= 1.0:
            return n_ctrl - 1  # 끝점
        for i in range(m - 1):
            if knot[i] <= param < knot[i + 1]:
                return i
        return n_ctrl - 1  # fallback