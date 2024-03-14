from typing import List, Union

from bbopy.utils.verbose import Column


class Table:
    def __init__(
            self,
            column_names: List[str],
            width: Union[List[int], int],
            title: str = None
    ) -> None:
        if isinstance(width, int):
            width = [width] * len(column_names)
        self.columns = [Column(name, w) for name, w in zip(column_names, width)]
        self.title = title

    def __getitem__(self, item: Union[int, str]) -> str:
        if (item == 0) or (item == "header"):
            return self._row(item, border=True)
        elif item == "title":
            return self._title()
        return self._row(item)

    def append(self, values: List[Union[int, float, str]]) -> None:
        for col, val in zip(self.columns, values):
            col.append(val)

    def _row(self, index: int, border: bool = False) -> str:
        row = "| " + " | ".join([col[index] for col in self.columns]) + " |"
        if border:
            border_line = "-" * len(row)
            row = f"{border_line}\n{row}\n{border_line}"
        return row

    def _title(self) -> str:
        total_width = len(self["header"].split("\n")[1]) - 2
        border_line = "-" * (total_width + 2)
        return f"{border_line}\n|{self.title.center(total_width)}|"
