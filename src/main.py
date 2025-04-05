################################################################################
# CPSC 589 : Modeling for Computer Graphics Project
# Real-Time Cloth Simulation Using Position-Based Dynamics and Parametric Surface Reconstruction
# Sehyeon Park
################################################################################

import taichi as ti
import platform

from src.utils.model_import import OBJLoader
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
    print(f"[Taichi Init] System: {system},"
          f" Arch: {machine}, "
          f"Using Taichi arch: {selected_arch}\n")

def create_window():
    window = ti.ui.Window("CPSC 589 Project", (800, 600))
    gui = window.get_gui()
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    return window, gui, canvas, scene, camera

def setup_camera(camera):
    camera.position(5.0, 5.0, 5.0)
    camera.lookat(0.0, 0.0, 0.0)
    camera.fov(40.0)

###############################################################################

def main():
    init_taichi()
    window, gui, canvas, scene, camera = create_window()
    setup_camera(camera)
    canvas.set_background_color((1.0, 1.0, 1.0))

    model = OBJLoader("plane_8", scale=[1.0, 1.0, 1.0])
    simulator = ClothSimulator(model)

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
        setup_camera(camera)
        scene.set_camera(camera)
        scene.ambient_light((0.5, 0.5, 0.5))
        gui_options()

        if sim_running:
            simulator.step()
            sim_frame += 1

        scene.mesh(simulator.x_cur, indices=simulator.ti_faces_flatten, color=(1.0, 0.0, 0.0))
        scene.mesh(simulator.x_cur, indices=simulator.ti_faces_flatten, color=(0.0, 0.0, 0.0), show_wireframe=True)
        canvas.scene(scene)
        window.show()

if __name__ == '__main__':
    main()