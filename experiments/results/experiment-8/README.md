# Dropout
## Predictions and left-shift by 1
![](exp_08_a.png)
![](exp_08_b.png)
## Predictions with outlier removed, and left-shift by 1
![](exp_08_c.png)
![](exp_08_d.png)
## Thoughts
- Dropout seems to be helping, which indicates the left-shift issue is likely one of the ways in which a timeseries predictive model can be overfit.
# Batch Normalization
- Just doesn't work, need to tweak it. Loss stays at ~20 for 150 epochs.
