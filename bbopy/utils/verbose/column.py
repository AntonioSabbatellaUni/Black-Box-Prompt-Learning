from typing import Union

import torch


class Column:

    def __init__(self, name: str, width: int) -> None:
        self.values = []
        self.name = name
        self.width = width

    def __getitem__(self, item):
        if (item == 0) or (item == "header"):
            return self._header()
        return _format_text(self.values[item], self.width)

    def append(self, value: Union[str, int, float]) -> None:
        self.values.append(value)

    def _header(self) -> str:
        return self.name.center(self.width)


def _format_text(value: Union[str, int, float, torch.Tensor], width: int) -> str:
    if value is not None:
        if isinstance(value, torch.Tensor):
            value = value.item()
        if isinstance(value, float):
            text = _number_to_text(value, width)
        else:
            text = str(value)
    else:
        text = "-"
    return text.center(width)


def _number_to_text(number: float, width: int) -> str:
    abs_number = abs(number)
    if abs_number >= 1e5 or abs_number * 1e5 < 1:
        return f"{number:.{width - 7}E}"
    elif 1e5 > abs_number >= 1e4:
        return f"{number:.{width - 9}f}"
    elif 1e4 > abs_number >= 1e3:
        return f"{number:.{width - 8}f}"
    elif 1e3 > abs_number >= 1e2:
        return f"{number:.{width - 7}f}"
    elif 1e2 > abs_number >= 10:
        return f"{number:.{width - 6}f}"
    else:
        return f"{number:.{width - 5}f}"
