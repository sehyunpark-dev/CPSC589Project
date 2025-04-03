################################################################################
# CPSC 589 : Modeling for Computer Graphics Project
# Real-Time Cloth Simulation Using Position-Based Dynamics and Parametric Surface Reconstruction
# Sehyeon Park
################################################################################

import taichi as ti
import numpy as np

from model_import import OBJLoader

ti.init(arch=ti.gpu)

global sim_running
global sim_frame

sim_running = False
sim_frame = 0

window = ti.ui.Window("CPSC 589 Project", (800, 600))
gui = window.get_gui()
canvas = window.get_canvas()

canvas.set_background_color((1.0, 1.0, 1.0))

# Option variables
def gui_options():
    with gui.sub_window("Options", 0.0, 0.0, 0.3, 0.7) as sub:
        start_pause_button = sub.button("Start/Pause")
        stop_button = sub.button("Stop")

        frame_str = "# Frame : " + str(sim_frame)
        sub.text(frame_str)

while window.running:
    gui_options()

    if sim_running:
        sim_frame += 1

    window.show()