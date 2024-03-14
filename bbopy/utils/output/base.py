import pandas as pd


class Output:
    r"""The results of an experiment.

    It contains all the results of an experiment, like objective values, computational times, etc.

    Attributes:
        data: a dictionary of list. Each list tracks a specifica parameter along iterations or generations.
    """

    def __init__(self):
        r"""Initializes the dictionary to track results of an experiment."""
        self.data = {
            "index": [],
            "ask_times": [],
            "eval_times": [],
            "tell_times": [],
            "x": [],
            "y": []
        }

    def append(self, **kwargs):
        r"""Append a new set of parameters to the `data` dictionary.

        This is usually called at each iteration or generation of an experiment to update the results.

        Args:
            **kwargs: all the parameter to be added to `data`.
                If a provided kwarg is not in `data`, it will be ignored.
                If a parameter in `data` is not provided in the kwargs, None will be added instead,
                to keep the list dimensions coherent along all parameters.
        """
        for key in self.data:
            self.data[key].append(kwargs.get(key, None))

    def to_pandas(self):
        r"""Converts the `data` dictionary into a pandas Dataframe.

        Returns:
            A pandas Dataframe with a column for each parameter in `data`.
            In particular, `x` and `y` columns are exploded to handle multidimensional data.
        """
        res = pd.DataFrame(self.data)
        res = res.explode(["x", "y"]).reset_index(drop=True)
        x_data = pd.DataFrame(res["x"].tolist())
        x_data.columns = [f"x_{col}" for col in x_data.columns]
        y_data = pd.DataFrame(res["y"].tolist())
        y_data.columns = [f"y_{col}" for col in y_data.columns]
        return pd.concat([res.drop(columns=["x", "y"]), y_data, x_data], axis=1)
