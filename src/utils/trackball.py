import numpy as np
from pyquaternion import Quaternion

class VirtualTrackball:
    def __init__(self, window_width: int, window_height: int, radius :float = 1.0):
        self.width = window_width
        self.height = window_height
        self.radius = radius
        self.current_quat = Quaternion()
        self.last_pos = None
        self.is_mouse_down = False

    def map_to_sphere(self, x: float, y: float) -> np.ndarray:
        # normalize (x,y) coord from [0,1] to [-1,1]
        x_norm = 2.0 * x - 1.0
        y_norm = 2.0 * y - 1.0
        d = np.sqrt(x_norm ** 2 + y_norm ** 2)
        if d < self.radius:
            z = np.sqrt(self.radius ** 2 - d ** 2)
        else:
            z = 0.0
        return np.array([x_norm, y_norm, z], dtype=np.float32)

    def on_mouse_press(self, x: float, y: float):
        self.is_mouse_down = True
        self.last_pos = self.map_to_sphere(x, y)

    def on_mouse_release(self):
        self.is_mouse_down = False
        self.last_pos = None

    def on_mouse_drag(self, x: float, y: float) -> Quaternion:
        if not self.is_mouse_down:
            return self.current_quat

        current_pos = self.map_to_sphere(x, y)
        if self.last_pos is None:
            self.last_pos = current_pos
            return self.current_quat

        # Calculate rotation axis using the cross product
        axis = np.cross(current_pos, self.last_pos)
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-6:
            axis /= axis_norm
        else:
            axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # Calculate rotation angle using the dot product
        dot_val = np.dot(self.last_pos, current_pos)
        dot_val = np.clip(dot_val, -1.0, 1.0)  # Ensure numerical stability
        angle = np.arccos(dot_val)

        # Create a quaternion representing the incremental rotation
        delta_quat = Quaternion(axis=axis, angle=angle)
        self.current_quat = delta_quat * self.current_quat
        self.last_pos = current_pos
        return self.current_quat

    def reset(self):
        self.current_quat = Quaternion()
        self.last_pos = None