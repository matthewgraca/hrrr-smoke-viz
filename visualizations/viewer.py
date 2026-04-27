import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('file')
args = parser.parse_args()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def show_npy_images_with_slider(path_to_npy: str) -> None:
    """
    Load a NumPy .npy file containing grayscale images of shape (samples, x, y)
    and display them with a slider to browse through samples.

    Mouse wheel scroll will move the slider by 1 image at a time.
    """
    images = np.load(path_to_npy, allow_pickle=True)['data']

    if images.ndim != 3:
        raise ValueError(
            f"Expected array of shape (samples, x, y), got shape {images.shape}"
        )

    num_samples = images.shape[0]
    current_idx = 0

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.18)

    img_artist = ax.imshow(images[current_idx], cmap="gray")
    ax.set_title(f"Image {current_idx} / {num_samples - 1}")
    ax.axis("off")

    slider_ax = fig.add_axes([0.15, 0.06, 0.7, 0.04])
    sample_slider = Slider(
        ax=slider_ax,
        label="Sample",
        valmin=0,
        valmax=num_samples - 1,
        valinit=current_idx,
        valstep=1,
    )

    def set_image(idx: int) -> None:
        nonlocal current_idx
        idx = max(0, min(num_samples - 1, idx))
        current_idx = idx
        img_artist.set_data(images[current_idx])
        ax.set_title(f"Image {current_idx} / {num_samples - 1}")
        fig.canvas.draw_idle()

    def update_from_slider(val) -> None:
        set_image(int(sample_slider.val))

    def on_scroll(event) -> None:
        if event.button == "up":
            new_idx = current_idx + 1
        elif event.button == "down":
            new_idx = current_idx - 1
        else:
            return

        new_idx = max(0, min(num_samples - 1, new_idx))

        if new_idx != current_idx:
            sample_slider.set_val(new_idx)

    sample_slider.on_changed(update_from_slider)
    fig.canvas.mpl_connect("scroll_event", on_scroll)

    plt.show()

if __name__ == "__main__":
    show_npy_images_with_slider(args.file)
