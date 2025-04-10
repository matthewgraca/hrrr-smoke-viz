# hrrr-smoke-viz
Working with HRRR data to eventually use to train a Convolutional LSTM

# Setup environment 
Conda environment 
- CPU: `conda env create -f env/cpu_environment.yml`
- GPU: `conda env create -f env/gpu_environment.yml`

# Output of `hrrr_smoke_viz`: 12 hour forecast
![](images/full_forecast.gif)

# Content
- `hrrr_smoke_viz` describes how to use `Herbie` to download HRRR data and visualize it.
    - **Use this notebook if you want to learn how to use Herbie, download HRRR data, and visualize the results.**
- `hrrr-airnow-convlstm_experiment` is the entire experiment combining AirNow and HRRR as two channels to the convlstm
    - **Use this notebook if you want to learn how to combine HRRR and AirNow data for convlstm training, as well as see the inventory of preprocessing functions that you can use yourself**
