import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Drawing function
class Plotter:
    def __init__(self, mem_data, l, v_min, v_max, node):
        self.mem_data = mem_data
        self.l = l
        self.v_min = v_min
        self.v_max = v_max
        self.node = node

    def update(self, frame):
        plt.clf()  # Clear previous graphics
        mem = self.mem_data[:, frame].reshape(self.l, self.l)
        cax = plt.imshow(mem, cmap='viridis', vmin=self.v_min, vmax=self.v_max)  # plt
        plt.colorbar(cax)
        plt.title(f'Time: {int(frame * self.node.dt)} ms')
        return plt

    def create_animation(self, frames, interval, output_file):
        ani = FuncAnimation(plt.figure(figsize=(6, 5)), self.update, frames=frames, interval=interval)
        ani.save(output_file, writer='pillow')
        print(f"GIF saved as {output_file}")

    def plot_snapshots(self, data, frame):
        plt_data = data.reshape(self.l, self.l, -1)  # Data reorganization
        plt_frame = plt_data[:, :, frame-1]
        max_index = np.argmax(plt_frame)
        max_value = np.max(plt_frame)
        print("max_index:", max_index, "max_value:", max_value) # print max node V
        plt.figure(figsize=(6, 5))
        plt.imshow(plt_frame, cmap='viridis', vmin=np.min(plt_frame), vmax=np.max(plt_frame))  # plt
        plt.colorbar()
        plt.title(f'Time: {int(frame*self.node.dt)} ms')
        plt.show()

    def plot_potentials(self, data, indices, labels):
        plt.figure(figsize=(6, 5))
        for index, label in zip(indices, labels):
            plt.plot(data[index, :], label=label)   # plt
        plt.legend()
        plt.show()