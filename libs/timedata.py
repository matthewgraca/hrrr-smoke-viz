import pandas as pd
import numpy as np

class TimeData:
    def __init__(
        self,
        start_date="2023-08-02",
        end_date="2025-08-02",
        dim=40,
        cyclical=True,
        month=True,
        day_of_week=False,
        day_of_month=False,
        verbose=True,
    ):
        dates = pd.date_range(
            start=start_date,
            end=end_date,
            freq='h',
            inclusive='left'
        )

        self.data = self._frames_temporal_encoding_of(
            dates, 
            dim, 
            month,
            day_of_week,
            day_of_month,
            cyclical,
            verbose
        )

        if verbose: print("✅ Encoding complete!")
        
    def _frames_temporal_encoding_of(
        self,
        dates,
        dim=40,
        month=True,
        day_of_week=True,
        day_of_month=True,
        cyclical=False,
        verbose=True
    ):
        '''
        Creates frames of temporal encodings for a given start and end date.
            Each time option will have its own channel that is scaled.
            For instance, hour 14 -> a frame with elements of 14/23, according
            to whatever dimension is passed.

        Cyclical flag determines if cyclical encoding should be used.

        Will always at least generate hourly temporal encodings; other options
            can be toggled.

        Returned in the order:
            - Month (sin, cos)
            - Day of month (sin, cos)
            - Day of week (sin, cos)
            - Hour (sin, cos)
        '''

        toggles = [
            (month, '%-m', 12),
            (day_of_week ,'%w', 6),
            (day_of_month, '%-d', 31),
            (True, '%-H', 23)
        ]

        options = {
            pattern : max_val
            for toggle, pattern, max_val in toggles
            if toggle
        }

        if verbose: print(self._temporal_encoding_msg(dates, options))

        temporal_encoded_data = []
        for pattern, max_val in options.items():
            a = np.full((len(dates)), dates.strftime(pattern), dtype='float')
            if cyclical:
                a = 2 * np.pi * a / max_val

                b = np.sin(a) 
                b = self._expand_val_across_dimensions(b, dim)
                temporal_encoded_data.append(b)

                c = np.cos(a) 
                c = self._expand_val_across_dimensions(c, dim)
                temporal_encoded_data.append(c)
            else:
                a = a / max_val
                a = self._expand_val_across_dimensions(a, dim)
                temporal_encoded_data.append(a)
        
        return np.concatenate(temporal_encoded_data, axis=-1)
        
    def _temporal_encoding_msg(self, dates, options):
        msg = [
            f"{dates.strftime(option).unique().astype('string').to_numpy()}"
            for option, toggle in options.items()
            if toggle
        ]

        return (
            f"🕒 Encoding the following options:\n"
            f"{"\n".join(msg)}"
        )

    def _expand_val_across_dimensions(self, data, dim):
        # expand and repeat values to the x and y dimension 
        data = np.repeat(data[:, np.newaxis, np.newaxis], dim, axis=1)
        data = np.repeat(data, dim, axis=2)
        data = np.expand_dims(data, axis=-1)

        return data 

