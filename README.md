# hrrr-smoke-viz
Visualizing smoke forecasts from HRRR.

# Running
Just pop a virtual environment and run the notebook.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

In order to run the notebooks that use `wgrib2`, you'll have to use conda:

```bash
conda create hrrrenv
conda activate hrrrenv
conda install --file conda_reqs.txt
```
# Output
![](images/full_forecast.gif)

# Content
- `hrrr_smoke_viz` describes how to use `Herbie` to download HRRR data and visualize it.
- `hrrr_to_np_manual` describes how to download HRRR data and convert it to `numpy` manually. Consider this deprecated for `hrrr_to_convlstm`
- `hrrr_to_convlstm` describes the entire pipeline of creating the 5D tensor (samples, frames, rows, columns, channels) used as input for the ConvLSTM 
    - Downloading subsetted HRRR data
    - Using `Herbie's` `wrgrib2` wrapper to subregion the data
    - Converting the frames to `numpy` format
    - Downsampling the frames 
    - Creating multiple samples using a sliding window of frames
