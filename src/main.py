################################################################################
# CPSC 589 : Modeling for Computer Graphics Project
# Real-Time Cloth Simulation Using Position-Based Dynamics and Parametric Surface Reconstruction
# Sehyeon Park
################################################################################

import taichi as ti
import numpy as np
import time
import platform

from trimesh.graph import facets

from src.utils.model_import import OBJLoader
from src.utils.camera import CameraController
from src.utils.vertices_selector import VerticesSelector
from src.engine.plane_uv_mapping import ParametricMapping
from src.engine.cylinder_uv_mapping import CylindricalMapping
from src.engine.b_spline_surface import BSplineSurface
from src.engine.simulator import ClothSimulator

def init_taichi():
    system = platform.system()
    machine = platform.machine()
    selected_arch = ti.gpu
    if system == 'Darwin':
        if machine == 'x86_64':
            selected_arch = ti.cpu
        else:
            selected_arch = ti.arm64
    elif system in ['Linux', 'Windows']:
        selected_arch = ti.gpu

    ti.init(arch=selected_arch, default_fp=ti.f32, device_memory_GB=8)
    print(f"[Taichi Init] System: {system}, "
          f"Arch: {machine}, "
          f"Using Taichi arch: {selected_arch}\n")

def create_window(window_width: int, window_height: int):
    window = ti.ui.Window("CPSC 589 Project", (window_width, window_height))
    gui = window.get_gui()
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    return window, gui, canvas, scene, camera

def setup_camera(camera, init_x, init_y, init_z):
    camera.position(init_x, init_y, init_z)
    camera.lookat(0.0, 0.0, 0.0)
    camera.fov(45.0)

@ti.kernel
def extract_selected_particles(x_cur: ti.template(), selected_indices: ti.template(),
                               selected_positions: ti.template(), num_vertices: ti.i32):
    for i in range(num_vertices):
        if selected_indices[i] == 1:
            selected_positions[i] = x_cur[i]
        else:
            selected_positions[i] = ti.Vector([-10000.0, -10000.0, -10000.0])

@ti.kernel
def fix_selected_particles(selected: ti.template(), fixed: ti.template(), num_vertices: ti.i32):
    for i in range(num_vertices):
        if selected[i] == 1:
            fixed[i] = 0.0
            print("Vertex ", i, "is fixed")

@ti.kernel
def reset_fixed(selected: ti.template(), fixed: ti.template(), num_vertices: ti.i32):
    for i in range(num_vertices):
        selected[i] = 0.0
        fixed[i] = 1.0

###############################################################################

def main():
    # Initialization
    init_taichi()

    target_fps = 60.0
    frame_duration = 1.0 / target_fps

    window_width, window_height = (800, 600)
    window, gui, canvas, scene, camera = create_window(window_width, window_height)

    # Camera setup
    camera_pos = np.array([-3.0, 3.0, 3.0])
    new_camera_pos = camera_pos

    setup_camera(camera, camera_pos[0], camera_pos[1], camera_pos[2])
    canvas.set_background_color((1.0, 1.0, 1.0))

    # gui variables
    sim_running = False
    sim_frame = 0
    stretch_stiffness_gui = 5e5
    bending_stiffness_gui = 5e5
    substeps_gui = 20
    use_bspline = True

    # Load objects (model, uv_mapper, simulator, etc.)
    model_8 = OBJLoader("plane_8", rotation_axis=[0,0,1], rotation_degree=90)
    model_64 = OBJLoader("plane_64", rotation_axis=[0,0,1], rotation_degree=90)
    skirt = OBJLoader("skirt")

    # model_8
    uv_mapper_8 = ParametricMapping(model_8.vertices_np)
    simulator_8 = ClothSimulator(model_8, dt=0.03, stretch_stiffness=5e5, bending_stiffness=5e5, num_substeps=20)
    b_spline_8 = BSplineSurface(model_8.vertices_np, uv_mapper_8.mapping,
                                num_u=9, num_v=9, res_u=65, res_v=65, order_u=4, order_v=4)
    selector_8 = VerticesSelector(window_width, window_height, camera, canvas,
                                  simulator_8.ti_vertices, simulator_8.num_vertices)
    selected_positions_8 = ti.Vector.field(3, dtype=ti.f32, shape=simulator_8.num_vertices)

    # model_64
    uv_mapper_64 = ParametricMapping(model_64.vertices_np)
    simulator_64 = ClothSimulator(model_64, dt=0.03, stretch_stiffness=5e5, bending_stiffness=5e5, num_substeps=20)
    b_spline_64 = BSplineSurface(model_64.vertices_np, uv_mapper_64.mapping,
                                 num_u=65, num_v=65, res_u=100, res_v=100, order_u=4, order_v=4)
    selector_64 = VerticesSelector(window_width, window_height, camera, canvas,
                                   simulator_64.ti_vertices, simulator_64.num_vertices)
    selected_positions_64 = ti.Vector.field(3, dtype=ti.f32, shape=simulator_64.num_vertices)

    uv_mapper_skirt = CylindricalMapping(skirt.vertices_np)
    simulator_skirt = ClothSimulator(skirt, dt=0.03, stretch_stiffness=5e5, bending_stiffness=5e5, num_substeps=20)
    b_spline_skirt = BSplineSurface(skirt.vertices_np, uv_mapper_skirt.mapping,
                                    num_u=uv_mapper_skirt.num_u, num_v=uv_mapper_skirt.num_v,
                                    res_u=uv_mapper_skirt.res_u, res_v=uv_mapper_skirt.res_v,
                                    order_u=4, order_v=4, is_cylinder=True)
    selector_skirt = VerticesSelector(window_width, window_height, camera, canvas,
                                      simulator_skirt.ti_vertices, simulator_skirt.num_vertices)
    selected_positions_skirt = ti.Vector.field(3, dtype=ti.f32, shape=simulator_skirt.num_vertices)

    # Load Utility objects (camera controller, vertices selector, etc.)
    camera_controller = CameraController()

    # init value
    simulator = simulator_8
    b_spline = b_spline_8
    selector = selector_8
    selected_positions = selected_positions_8

    def gui_options():
        nonlocal simulator, b_spline, selector, selected_positions
        nonlocal sim_running, sim_frame
        nonlocal stretch_stiffness_gui, bending_stiffness_gui, substeps_gui
        nonlocal use_bspline

        with gui.sub_window("Options", 0.0, 0.0, 0.3, 0.7) as sub:
            if sub.button("Start/Pause"):
                sim_running = not sim_running

            if sub.button("Stop"):
                sim_running = False
                sim_frame = 0
                simulator.reset()
                b_spline.reset()

            if sub.button("Use model_8"):
                current_model = "model_8"
                simulator = simulator_8
                b_spline = b_spline_8
                selector = selector_8
                selected_positions = selected_positions_8
                sim_running = False
                sim_frame = 0
                simulator.reset()
                b_spline.reset()

            if sub.button("Use model_64"):
                current_model = "model_64"
                simulator = simulator_64
                b_spline = b_spline_64
                selector = selector_64
                selected_positions = selected_positions_64
                sim_running = False
                sim_frame = 0
                simulator.reset()
                b_spline.reset()

            if sub.button("Use skirt"):
                current_model = "skirt"
                simulator = simulator_skirt
                b_spline = b_spline_skirt
                selector = selector_skirt
                selected_positions = selected_positions_skirt
                sim_running = False
                sim_frame = 0
                simulator.reset()
                b_spline.reset()

            stretch_stiffness_gui = sub.slider_float("Stretch Stiffness", stretch_stiffness_gui, 1e2, 1e6)
            bending_stiffness_gui = sub.slider_float("Bending Stiffness", bending_stiffness_gui, 1e2, 1e6)
            substeps_gui = sub.slider_int("Substeps", substeps_gui, 1, 100)
            simulator.stretch_stiffness = stretch_stiffness_gui
            simulator.bending_stiffness = bending_stiffness_gui
            simulator.num_substeps = substeps_gui

            use_bspline = sub.checkbox("Use B-spline Surface", use_bspline)
            simulator.enable_wind = sub.checkbox("Enable Wind", simulator.enable_wind)

            frame_str = "# Frame : " + str(sim_frame)
            vertices = "# vertices : " + str(simulator.num_vertices)
            faces = "# faces : " + str(simulator.num_faces)
            edges = "# edges : " + str(simulator.num_edges)
            sub.text(frame_str)
            sub.text(vertices)
            sub.text(faces)
            sub.text(edges)

    while window.running:
        frame_start = time.time()

        scene.set_camera(camera)
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.point_light((10.0, 10.0, 10.0), color=(0.5, 0.5, 0.5))
        gui_options()

        ################################################################################
        # Event handler

        if window.get_event(ti.ui.PRESS):
            # Vertices selector
            if window.event.key == ti.ui.LMB:
                cursor_pos = window.get_cursor_pos()
                selector.on_mouse_press(cursor_pos[0], cursor_pos[1])

            # Virtual trackball (Rotation)
            elif window.event.key == ti.ui.RMB:
                cursor_pos = window.get_cursor_pos()
                camera_controller.on_mouse_press(cursor_pos[0], cursor_pos[1])

            # Zoom in
            elif window.event.key == ti.ui.UP:
                new_camera_pos = camera_controller.zoom(camera_pos, is_zoom_in=True)
                camera_pos = new_camera_pos
                camera.position(camera_pos[0], camera_pos[1], camera_pos[2])
                camera.lookat(0.0, 0.0, 0.0)

            # Zoom out
            elif window.event.key == ti.ui.DOWN:
                new_camera_pos = camera_controller.zoom(camera_pos, is_zoom_in=False)
                camera_pos = new_camera_pos
                camera.position(camera_pos[0], camera_pos[1], camera_pos[2])
                camera.lookat(0.0, 0.0, 0.0)

            elif window.event.key == 'f':
                if selector.selected_indices is not None:
                    fix_selected_particles(selector.selected_indices, simulator.fixed, simulator.num_vertices)

            elif window.event.key == 'r':
                reset_fixed(selector.selected_indices, simulator.fixed, simulator.num_vertices)

        if window.get_event(ti.ui.RELEASE):
            # Vertices selector
            if window.event.key == ti.ui.LMB:
                cursor_pos = window.get_cursor_pos()
                selector.on_mouse_release(cursor_pos[0], cursor_pos[1])

            # Virtual trackball (Rotation)
            elif window.event.key == ti.ui.RMB:
                camera_controller.on_mouse_release()
                camera_pos = new_camera_pos

        if camera_controller.is_mouse_down:
            cursor_pos = window.get_cursor_pos()
            new_quat = camera_controller.on_mouse_drag(cursor_pos[0], cursor_pos[1])
            rot_mat = new_quat.rotation_matrix

            new_camera_pos = rot_mat @ camera_pos
            camera.position(new_camera_pos[0], new_camera_pos[1], new_camera_pos[2])
            camera.lookat(0.0, 0.0, 0.0)
            # Do not renew the camera position. Because it will be accumulated!

        if selector.is_dragging:
            cursor_pos = window.get_cursor_pos()
            selector.on_mouse_drag(cursor_pos[0], cursor_pos[1])
            selector.get_rect_lines()

        ################################################################################
        # Simulator
        if sim_running:
            simulator.step()
            x_cur_np = simulator.x_cur.to_numpy()
            b_spline.evaluate_surface_wrapper(x_cur_np)
            sim_frame += 1

        ################################################################################
        # Canvas Renderer
        if selector.selected_indices is not None:
            extract_selected_particles(simulator.x_cur, selector.selected_indices, selected_positions,
                                       simulator.num_vertices)
            scene.particles(selected_positions, radius=0.01, color=(0.0, 0.0, 1.0))

        if use_bspline:
            scene.mesh(b_spline.surface_points_field, indices=b_spline.surface_faces_field, color=(1.0, 1.0, 0.0))
        else:
            scene.mesh(simulator.x_cur, indices=simulator.ti_faces_flatten, color=(1.0, 1.0, 0.0))
        scene.mesh(simulator.x_cur, indices=simulator.ti_faces_flatten, color=(0.0, 0.0, 0.0), show_wireframe=True)
        canvas.scene(scene)
        window.show()

        # frame_end = time.time()
        # elapsed_time = frame_end - frame_start
        # sleep_time = frame_duration - elapsed_time
        # if sleep_time > 0:
        #     time.sleep(sleep_time)

if __name__ == '__main__':
    main()