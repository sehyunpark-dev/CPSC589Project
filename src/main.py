################################################################################
# CPSC 589 : Modeling for Computer Graphics Project
# Real-Time Cloth Simulation Using Position-Based Dynamics and Parametric Surface Reconstruction
# Sehyeon Park
################################################################################

import taichi as ti
import numpy as np
import platform

from src.utils.model_import import OBJLoader
from src.utils.camera import CameraController
from src.utils.vertices_selector import VerticesSelector
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
def reset_fixed(fixed: ti.template(), num: ti.i32):
    for i in range(num):
        fixed[i] = 1.0

###############################################################################

def main():
    # Initialization
    init_taichi()
    window_width, window_height = (800, 600)
    window, gui, canvas, scene, camera = create_window(window_width, window_height)

    # Camera setup
    camera_pos = np.array([5.0, 5.0, 5.0])
    new_camera_pos = camera_pos

    setup_camera(camera, camera_pos[0], camera_pos[1], camera_pos[2])
    canvas.set_background_color((1.0, 1.0, 1.0))

    # Load objects
    model = OBJLoader("plane_8")
    simulator = ClothSimulator(model, dt=0.001)
    camera_controller = CameraController()
    vertices_selector = VerticesSelector(window_width, window_height,
                                         camera, canvas,
                                         simulator.ti_vertices, simulator.num_vertices)
    selected_positions = ti.Vector.field(3, dtype=ti.f32, shape=simulator.num_vertices)

    sim_running = False
    sim_frame = 0

    def gui_options():
        nonlocal simulator, sim_running, sim_frame
        with gui.sub_window("Options", 0.0, 0.0, 0.3, 0.7) as sub:
            if sub.button("Start/Pause"):
                sim_running = not sim_running
            if sub.button("Stop"):
                sim_running = False
                sim_frame = 0
                simulator.reset()
            frame_str = "# Frame : " + str(sim_frame)
            sub.text(frame_str)

    while window.running:
        scene.set_camera(camera)
        scene.ambient_light((0.5, 0.5, 0.5))
        gui_options()

        ################################################################################
        # Event handler

        if window.get_event(ti.ui.PRESS):
            # Virtual trackball (Rotation)
            if window.event.key == ti.ui.LMB:
                cursor_pos = window.get_cursor_pos()
                camera_controller.on_mouse_press(cursor_pos[0], cursor_pos[1])

            # Vertices selector
            elif window.event.key == ti.ui.RMB:
                cursor_pos = window.get_cursor_pos()
                vertices_selector.on_mouse_press(cursor_pos[0], cursor_pos[1])

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
                if vertices_selector.selected_indices is not None:
                    fix_selected_particles(vertices_selector.selected_indices, simulator.fixed, simulator.num_vertices)

            elif window.event.key == 'h':
                reset_fixed(simulator.fixed, simulator.num_vertices)

        if window.get_event(ti.ui.RELEASE):
            # Virtual trackball (Rotation)
            if window.event.key == ti.ui.LMB:
                camera_controller.on_mouse_release()
                camera_pos = new_camera_pos

            # Vertices selector
            if window.event.key == ti.ui.RMB:
                cursor_pos = window.get_cursor_pos()
                vertices_selector.on_mouse_release(cursor_pos[0], cursor_pos[1])

        if camera_controller.is_mouse_down:
            cursor_pos = window.get_cursor_pos()
            new_quat = camera_controller.on_mouse_drag(cursor_pos[0], cursor_pos[1])
            rot_mat = new_quat.rotation_matrix

            new_camera_pos = rot_mat @ camera_pos
            camera.position(new_camera_pos[0], new_camera_pos[1], new_camera_pos[2])
            camera.lookat(0.0, 0.0, 0.0)
            # Do not renew the camera position. Because it will be accumulated!

        if vertices_selector.is_dragging:
            cursor_pos = window.get_cursor_pos()
            vertices_selector.on_mouse_drag(cursor_pos[0], cursor_pos[1])
            vertices_selector.get_rect_lines()

        ################################################################################
        # Simulator
        if sim_running:
            simulator.step()
            sim_frame += 1

        ################################################################################
        # Canvas Renderer
        if vertices_selector.selected_indices is not None:
            extract_selected_particles(simulator.x_cur, vertices_selector.selected_indices, selected_positions,
                                       simulator.num_vertices)
            scene.particles(selected_positions, radius=0.01, color=(0.0, 0.0, 1.0))

        scene.mesh(simulator.x_cur, indices=simulator.ti_faces_flatten, color=(1.0, 0.0, 0.0))
        scene.mesh(simulator.x_cur, indices=simulator.ti_faces_flatten, color=(0.0, 0.0, 0.0), show_wireframe=True)
        canvas.scene(scene)
        window.show()

if __name__ == '__main__':
    main()