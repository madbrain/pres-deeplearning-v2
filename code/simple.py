import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(5,5)

p = np.random.permutation(100)
X = np.linspace(-1, 1, 100)[p]
data = 2 * X + np.random.randn(100)[p]
epochs = 5
learningRate = 0.01
w = np.array([0, 0])
divide = 5

def animate(i):
    global w
    for z in range(divide):
        x = np.array([X[i % len(X)], 1])
        gw = 2 * (w * x - data[i % len(X)]) * x
        w = w - learningRate * gw

    ax.clear()
    color = [ "green" if (i % len(X)) == j else "blue" for j in range(len(X)) ]
    ax.scatter(X, data, c=color)

    ax.set_ylim(-3, 3)

    x_vals = np.array(ax.get_xlim())
    y_vals = w[0] * x_vals + w[1]
    ax.plot(x_vals, y_vals, 'r')
    print(w)

ani = FuncAnimation(fig, animate, frames=len(X)*epochs//divide, interval=2, repeat=True)
plt.close()

# Save the animation as an animated GIF
ani.save("lin_reg_anim.gif", dpi=300, writer=PillowWriter(fps=10))

# https://spatialthoughts.com/2022/01/14/animated-plots-with-matplotlib/

