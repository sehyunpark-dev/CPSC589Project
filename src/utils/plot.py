import numpy as np
import matplotlib.pyplot as plt
import os

current_dir = os.getcwd()
results_dir = os.path.abspath(os.path.join(current_dir, "../..", "results"))
file_model_8 = os.path.join(results_dir, "residual_history_plane_8.npy")
file_model_64 = os.path.join(results_dir, "residual_history_plane_64.npy")

residuals_8 = np.load(file_model_8)
residuals_64 = np.load(file_model_64)

plt.figure(figsize=(7, 5))
plt.plot(residuals_8, label="model_8 + B-spline", linestyle="--", linewidth=2)
plt.plot(residuals_64, label="model_64 (Full XPBD)", linestyle="-", linewidth=2)
plt.yscale("log")
plt.xlabel("Frame")
plt.ylabel("Constraint Residual Energy")
plt.title("Constraint Convergence Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()