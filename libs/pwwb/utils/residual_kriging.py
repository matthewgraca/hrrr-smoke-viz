import numpy as np
import skgstat as skg
from tqdm import tqdm

class ResidualKriging:
    def __init__(
        self,
        dim=40,
        verbose=0,   # 0 = all, 1 = progress bar + errors, 2 = errors only
        # kriging settings
        min_points=3,
        max_points=25,
        mode='exact',
        # variogram settings
        estimator='matheron',
        model='spherical',
        bin_func='kmeans',
        use_nugget=False,
        n_lags=10
    ):
        # members
        self.dim = dim
        self.VERBOSE = verbose

        self.kriging_kwargs = {
            'min_points' : min_points,
            'max_points' : max_points
        }

        self.variogram_kwargs = {
            'estimator' : estimator,
            'model' : model,
            'bin_func' : bin_func,
            'use_nugget' : use_nugget,
            'n_lags' : n_lags
        }

    ### NOTE: Public methods

    def interpolate_frames(self, sensor_frames, model_frames):
        """
        Interpolates frames

        Currently generates a new semi-variogram for each frame.
        """
        first_frame = sensor_frames[0]
        if not self._validate_grid_is_interpolatable(first_frame):
            return sensor_frames

        interpolated_grids = [
            self._interpolate_frame(
                sensor_frame,
                model_frame,
                self.dim,
            )
            for sensor_frame, model_frame in (
                tqdm(zip(sensor_frames, model_frames))
                if self.VERBOSE < 2 else zip(sensor_frames, model_frames)
            )
        ]

        return np.array(interpolated_grids)

    ### NOTE: Helpers

    def _validate_grid_is_interpolatable(self, unInter):
        """
        Checks for sensor values (numbers) and nan values to interpolate.
        
        If there are no nans, that means nothing can be interpolated.
        If there are only nans, then there are no real values to interpolate
            with.
        """
        sensor_indices = np.where(~np.isnan(unInter))
        if len(sensor_indices[0]) == 0:
            print(
                "No non-nan points found on grid, returning uninterpolated frame.\n"
                "Note: non-nan points are used to determine sensor locations."
            )
            return False

        x_dim, y_dim = unInter.shape

        if np.isnan(unInter).all():
            raise ValueError("Every value is nan; no value to interpolate.")

        return True

    def _get_sensor_coords(self, grid):
        """
        Initializes where the locations of the sensors are based on whether the
            pixel is NaN or not. This means that we expect the frame to contain
            ALL the sensor values (pre-imputed), and non-sensor locations to be
            NaN.

        Returns coordinates as list of pairs.
        """
        return list(zip(*np.where(~np.isnan(grid))))

    def _get_sensor_vals(self, grid):
        """
        Grabs the sensor values from a grid. Assumes that coordinates with 
            a real value is a proper sensor value, while coordinate with a
            NaN are not sensor values.
        """
        return grid[list(np.where(~np.isnan(grid)))]

    ### NOTE: The core interpolation methods

    def _interpolate_frame(self, sensor_frame, model_frame, dim):
        """
        Interpolate a frame using Residual Kriging. 
        Generates a unique semi-variogram per frame.
        """
        coords = self._get_sensor_coords(sensor_frame)
        X, Y = zip(*coords)
        xx, yy = np.mgrid[0:dim, 0:dim]

        residuals = sensor_frame[X, Y] - model_frame[X, Y]

        V = skg.Variogram(
            coordinates=coords,
            values=residuals,
            **self.variogram_kwargs
        )
        ok = skg.OrdinaryKriging(V, **self.kriging_kwargs)

        res_field = ok.transform(xx.flatten(), yy.flatten()).reshape(xx.shape)
        final_field = res_field + model_frame

        return final_field
