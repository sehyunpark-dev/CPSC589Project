################################################################################
# CPSC 589 : Modeling for Computer Graphics
# Sehyeon Park
################################################################################
from turtledemo.penrose import start

import taichi as ti
import numpy as np

ti.init(arch=ti.metal)

global sim_running
global sim_frame

window = ti.ui.Window("CPSC 589 Project", (800, 600))
gui = window.get_gui()
canvas = window.get_canvas()

canvas.set_background_color((1.0, 1.0, 1.0))

def gui_options():
    # Option variables

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