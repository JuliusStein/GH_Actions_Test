import matplotlib.pyplot as plt

import numpy as np

# evenly sample 1000 times from 0 seconds to 12 seconds
times = np.linspace(0, 2, 1000)  # seconds; shape-(1000,)

# measuring pressure at the specified times
f1 = 1  # Hertz
f2 = 2  # Hertz
f3 = 4  # Hertz
amp = 0.02  # Pascals


def p(f):
    return amp * np.sin(2 * np.pi * f * times)  # Pascals; shape-(1000,)


fig3 = plt.figure(constrained_layout=True)
gs = fig3.add_gridspec(3, 4)

f3_ax1 = fig3.add_subplot(gs[0, :-2])
f3_ax1.set_title("1 Hz")
f3_ax1.plot(times, p(f1), color="red")

f3_ax2 = fig3.add_subplot(gs[1, :-2])
f3_ax2.set_title("2 Hz")
f3_ax2.plot(times, p(f2), color="orange")

f3_ax3 = fig3.add_subplot(gs[2, :-2])
f3_ax3.set_title("4 Hz")
f3_ax3.plot(times, p(f3), color="green")

f3_ax4 = fig3.add_subplot(gs[:, 2:])
f3_ax4.plot(times, p(f1) + p(f2) + p(f3))
f3_ax4.set_xlabel("Time [seconds]")
f3_ax4.set_ylabel("Pressure Relative to Standing Air [Pascals]")
f3_ax4.set_title("Superposition")
f3_ax4.plot(times, p(f1), color="red", alpha=0.2)
f3_ax4.plot(times, p(f2), color="orange", alpha=0.2)
f3_ax4.plot(times, p(f3), color="green", alpha=0.2)

f3_ax1.grid(True)
f3_ax2.grid(True)
f3_ax3.grid(True)
f3_ax4.grid(True)

fig3.savefig("superposition_plot.png")