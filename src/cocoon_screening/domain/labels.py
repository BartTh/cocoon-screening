from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class LabelSet:

    names: list[str]
    colors: dict[str, str]  # label name -> hex color, e.g. "#07891F"

    def index_of(self, name: str) -> int:
        return self.names.index(name)

    def name_for(self, index: int) -> str:
        return self.names[index]

    def color_for(self, name: str) -> str:
        return self.colors.get(name, "#FFFFFF")

    def color_for_index(self, index: int) -> str:
        name = self.name_for(index)
        return self.color_for(name)

    def __contains__(self, name: str) -> bool:
        return name in self.names

    def __len__(self) -> int:
        return len(self.names)

def load_labels(path: Path) -> LabelSet:
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    return LabelSet(
        names=data["LABEL_LIST"],
        colors=data["LABEL_COLOR"],
    )
