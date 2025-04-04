################################################################################
# CPSC 589 : Modeling for Computer Graphics Project
# Real-Time Cloth Simulation Using Position-Based Dynamics and Parametric Surface Reconstruction
# Sehyeon Park
################################################################################

import taichi as ti
import platform

from model_import import OBJLoader

system = platform.system()
machine = platform.machine()
SELECTED_ARCH = ti.gpu
if system == 'Darwin':
    if machine == 'x86_64':
        SELECTED_ARCH = ti.cpu
    else:
        SELECTED_ARCH = ti.arm64
elif system in ['Linux', 'Windows']:
    SELECTED_ARCH = ti.gpu

ti.init(arch=SELECTED_ARCH, default_fp=ti.f32, device_memory_GB=8)
print(f"[Taichi Init] System: {system}, Arch: {machine}, Using Taichi arch: {SELECTED_ARCH}\n")

sim_running = False
sim_frame = 0

window = ti.ui.Window("CPSC 589 Project", (800, 600))
gui = window.get_gui()
canvas = window.get_canvas()
scene = window.get_scene()
camera = ti.ui.Camera()

canvas.set_background_color((1.0, 1.0, 1.0))
camera.position(5.0, 5.0, 5.0)
camera.lookat(0.0, 0.0, 0.0)
camera.fov(40.0)

model = OBJLoader("plane_8", scale=[2.0, 2.0, 2.0])

# Option variables
def gui_options():
    with gui.sub_window("Options", 0.0, 0.0, 0.3, 0.7) as sub:
        start_pause_button = sub.button("Start/Pause")
        stop_button = sub.button("Stop")
        frame_str = "# Frame : " + str(sim_frame)
        sub.text(frame_str)

while window.running:
    camera.position(5.0, 5.0, 5.0)
    camera.lookat(0.0, 0.0, 0.0)
    scene.set_camera(camera)
    scene.ambient_light((0.5, 0.5, 0.5))

    gui_options()

    if sim_running:
        sim_frame += 1

    scene.mesh(model.ti_vertices, indices=model.ti_faces, color=(1.0, 0.0, 0.0))
    canvas.scene(scene)
    window.show()