import pandas as pd


AVAIL_DATASET = {}


class Dataset(pd.DataFrame):
    _meta = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.field_info()

    def field_info(self):
        """Statistic on the field types. Save the result in `_meta`, called in init
        """
        # todo:
        #  1. record the categorical/numerical data amount
        #  2. count the possible output for categorical
        pass