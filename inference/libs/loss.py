import tensorflow as tf
import numpy as np
from itertools import product

class NHoodMAE(tf.keras.losses.Loss):
    def __init__(
        self, sensors, dim, source_weight=25, nhood_weight=5, bg_weight=1, r=2
    ):
        '''
        Args:
            sensors: sensor info in the form of {'sensor name' : (x, y)}
            dim: the dimensions of the grid (assumes height = width)
            source_weight: weight assigned to the exact sensor location
            nhood_weight: weight assigned to the radius (r) around the sensor
            bg_weight: weight assigned to everything else (background)
            r: radius of the neighborhood
        '''
        super().__init__()
        self.sensors = sensors
        self.weights = self._get_weights(
            list(sensors.values()), dim, source_weight, nhood_weight, bg_weight, r
        )

    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true - y_pred) * self.weights)

    #### NOTE long block of funcs that find weights
    def _in_bounds(self, x, y, bound):
        '''
        ensures that neighborhood pair that is outside the dimensions doesn't
            get counted
        '''
        x_in_bound = x >= 0 and x < bound
        y_in_bound = y >= 0 and y < bound

        return x_in_bound and y_in_bound
    
    def _find_neighbors(self, sources, radius, dim):
        '''
        Finds the (x, y) pairs that serve as the neighbors of the sources
        '''
        n_hood = set(product(range(-radius, radius + 1), repeat=2))
        n_hood.remove((0, 0))
        neighbors = set()
        for x, y in sources:
            for a, b in n_hood:
                f, g = x + a, y + b
                if self._in_bounds(f, g, dim):
                    neighbors.add((f, g))

        return neighbors

    def _determine_weights(
        self, sources, n_hood, dim, source_weight, nhood_weight, bg_weight
    ):
        '''
        applies the proper weights to:
            - the background
            - the neighborhoods
            - the sources
        based on the list of pairs in sources and nhood
        '''
        weights = np.full((dim, dim), bg_weight)
        for (x, y) in n_hood:
            weights[x, y] = nhood_weight
        for (x, y) in sources:
            weights[x, y] = source_weight 

        return weights
    
    def _get_weights(
        self, sensor_locations, dim, source_weight, nhood_weight, bg_weight, radius
    ):
        sensor_coords = set(sensor_locations)
        neighbors = self._find_neighbors(sensor_coords, radius, dim)
        weights = self._determine_weights(
            sensor_coords, neighbors, dim, source_weight, nhood_weight, bg_weight
        )
        return weights

    # for saving and loading models using this loss
    def get_config(self):
        config = super().get_config()
        config.update({'sensors': self.sensors})
        return config
