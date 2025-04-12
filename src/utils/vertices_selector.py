import numpy as np
import taichi as ti

@ti.data_oriented
class VerticesSelector:
    def __init__(self,
                 window_width: int,
                 window_height: int,
                 camera: ti.ui.Camera,
                 canvas,
                 ti_vertices: ti.Vector.field,
                 num_vertices: int):
        self.window_width = window_width
        self.window_height = window_height
        self.camera = camera
        self.canvas = canvas
        self.ti_vertices = ti_vertices
        self.num_vertices = num_vertices

        self.aspect = self.window_width / self.window_height
        self.drag_start = None
        self.drag_end = None
        self.is_dragging = False
        self.selected_indices = ti.field(dtype=ti.i32, shape=self.num_vertices)
        self.selected_indices.fill(0)

        # for drawing a selection rectangular
        self.ti_rect_vertices = ti.Vector.field(2, dtype=ti.f32, shape=4)
        self.ti_rect_indices = ti.Vector.field(2, dtype=ti.i32, shape=4)

    def on_mouse_press(self, screen_x: float, screen_y: float):
        self.is_dragging = True
        self.drag_start = np.array([screen_x, screen_y])

    def on_mouse_drag(self, screen_x: float, screen_y: float):
        if self.is_dragging and self.drag_start is not None:
            self.drag_end = np.array([screen_x, screen_y])

    def on_mouse_release(self, screen_x: float, screen_y: float):
        self.is_dragging = False
        self.drag_end = np.array([screen_x, screen_y])
        if self.drag_start is not None and self.drag_end is not None:
            self.compute_selection()

    def get_rect_lines(self):
        if self.drag_start is None or self.drag_end is None:
            return None

        x_min = min(self.drag_start[0], self.drag_end[0])
        x_max = max(self.drag_start[0], self.drag_end[0])
        y_min = min(self.drag_start[1], self.drag_end[1])
        y_max = max(self.drag_start[1], self.drag_end[1])

        vertices_np = np.array([
            [x_min, y_min],
            [x_min, y_max],
            [x_max, y_max],
            [x_max, y_min],
        ], dtype=np.float32)
        indices_np = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32)

        self.ti_rect_vertices.from_numpy(vertices_np)
        self.ti_rect_indices.from_numpy(indices_np)
        self.canvas.lines(vertices=self.ti_rect_vertices, indices=self.ti_rect_indices,
                          width=0.002, color=(0.0, 0.0, 1.0))

    def compute_selection(self):
        x_min = min(self.drag_start[0], self.drag_end[0])
        x_max = max(self.drag_start[0], self.drag_end[0])
        y_min = min(self.drag_start[1], self.drag_end[1])
        y_max = max(self.drag_start[1], self.drag_end[1])

        # compute screen coords
        proj_matrix = self.camera.get_projection_matrix(aspect=self.aspect).T
        view_matrix = self.camera.get_view_matrix().T
        transform = proj_matrix @ view_matrix
        transform_ti = ti.Matrix([[ti.cast(transform[i, j], ti.f32) for j in range(4)] for i in range(4)])

        self.compute_selection_kernel(self.ti_vertices, transform_ti,
                                      float(x_min), float(y_min),
                                      float(x_max), float(y_max),
                                      float(self.window_width), float(self.window_height))

    @ti.kernel
    def compute_selection_kernel(self,
                                 vertices: ti.template(),
                                 transform: ti.template(),
                                 x_min: ti.f32, y_min: ti.f32,
                                 x_max: ti.f32, y_max: ti.f32,
                                 win_width: ti.f32, win_height: ti.f32):
        for i in range(self.num_vertices):
            # world coord -> 4D homogeneous vector
            v_world = ti.Vector([vertices[i][0], vertices[i][1], vertices[i][2], 1.0])
            clip = transform @ v_world  # clip space coord
            ndc = clip / clip[3] # divide by w

            # NDC [-1, 1] → screen coord [0, 1]
            # x: (ndc_x + 1) / 2
            # y: (1 - (ndc_y + 1) / 2)
            screen_x = (ndc[0] + 1.0) / 2.0
            screen_y = (ndc[1] + 1.0) / 2.0

            if x_min <= screen_x <= x_max and y_min <= screen_y <= y_max:
                self.selected_indices[i] = 1
