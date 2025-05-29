# Summary
Contains obsolete work or small-scale tests. It's likely these will either be reorganized or deleted sometime in the future.
# Contents
- `standard_scaler_experiment.py` tests if performing standard scaling is better manually with array methods, numpy methods, or `sklearn` methods.
    - numpy is both faster and simpler than every other option.
- `hrrr_to_np_manual` describes how to download HRRR data and convert it to `numpy` manually. Consider this deprecated for `hrrr_to_convlstm`
- `hrrr_to_convlstm_input` describes the entire pipeline of creating the 5D tensor (samples, frames, rows, columns, channels) used as input for the ConvLSTM 
    - **Use this notebook if you want to know how to convert HRRR data into a usable input for the convlstm**
    - Downloading subsetted HRRR data
    - Using `Herbie's` `wrgrib2` wrapper to subregion the data
    - Converting the frames to `numpy` format
    - Downsampling the frames 
    - Creating multiple samples using a sliding window of frames
- `hrrr-convlstm_experiment` is an entire experiment with using hrrr data and convlstms
    - **Use this notebook if you want to use HRRR data for convlstm training with next-frame and 5-frame predictions**
    - Data preprocessing
    - Model definition and training
    - Inference, with next-frame and next-5 frame prediction
