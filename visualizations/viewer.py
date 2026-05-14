import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('file', help='Expects a .npz file with the keys \'data\', \'start_date\', and \'end_date\'')
args = parser.parse_args()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def show_npy_images_with_slider(path_to_npy: str) -> None:
    """
    Load a NumPy .npy file containing grayscale images of shape (samples, x, y)
    and display them with a slider to browse through samples.

    Mouse wheel scroll will move the slider by 1 image at a time.
    """
    data = np.load(path_to_npy, allow_pickle=True)
    images = data['data']
    dates = pd.date_range(data['start_date'].item(), data['end_date'].item(), freq='h', inclusive='left')

    if images.ndim != 3:
        raise ValueError(
            f"Expected array of shape (samples, x, y), got shape {images.shape}"
        )

    num_samples = images.shape[0]
    current_idx = 0

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.18)

    img_artist = ax.imshow(images[current_idx], cmap="gray")
    ax.set_title(dates[current_idx].strftime('%Y-%m-%d %H:%M:%S UTC'))
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
        ax.set_title(dates[current_idx].strftime('%Y-%m-%d %H:%M:%S UTC'))
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

    def on_key(event) -> None:
        if event.key == "right":
            new_idx = current_idx + 1
        elif event.key == "left":
            new_idx = current_idx - 1
        else:
            return

        new_idx = max(0, min(num_samples - 1, new_idx))
        if new_idx != current_idx:
            sample_slider.set_val(new_idx)

    sample_slider.on_changed(update_from_slider)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()

if __name__ == "__main__":
    show_npy_images_with_slider(args.file)
