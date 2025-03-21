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
