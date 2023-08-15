# Function to create animation from images.
import matplotlib
from matplotlib import animation, rc

matplotlib_fname.rcParams["animation.embed_limit"] = 2**128
matplotlib.rcParams["savefig.pad_inches"] = 0
rc("animation", html="jshtml")


def create_animation(images):
    fig = plt.figure(figsize=(6, 9))
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    im = ax.imshow(images[0], cmap="gray")
    plt.close(fig)

    def animate_func(i):
        im.set_array(images[i])
        return [im]

    return animation.FuncAnimation(
        fig, animate_func, frames=len(images), interval=1000 / 10
    )
