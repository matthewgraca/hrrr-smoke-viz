# hrrr-smoke-viz
Working with HRRR data to eventually use to train a Convolutional LSTM

# Setup environment 
You have two options:
- Python virtual environment (only runs notebooks without `wgrib2` and `tensorflow`)
- Conda environment (recommended, allows you to run all of the notebooks)

```bash
conda env create -f environment.yml
conda activate hrrrenv
```
# Output of `hrrr_smoke_viz`: 12 hour forecast
![](images/full_forecast.gif)

# Content
- `hrrr_smoke_viz` describes how to use `Herbie` to download HRRR data and visualize it.
- `hrrr_to_convlstm_input` describes the entire pipeline of creating the 5D tensor (samples, frames, rows, columns, channels) used as input for the ConvLSTM 
    - Downloading subsetted HRRR data
    - Using `Herbie's` `wrgrib2` wrapper to subregion the data
    - Converting the frames to `numpy` format
    - Downsampling the frames 
    - Creating multiple samples using a sliding window of frames
- `hrrr-convlstm_experiment` is an entire experiment with using hrrr data and convlstms
    - Data preprocessing
    - Model definition and training
    - Inference, with next-frame and next-5 frame prediction
