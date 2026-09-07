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
        if not 0 <= index < len(self.names):
            raise ValueError(f"Invalid class index: {index}")
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

    if not isinstance(data, dict):
        raise ValueError(f"Invalid labels configuration: {path}")
    names = data.get("LABEL_LIST")
    colors = data.get("LABEL_COLOR")
    if (not isinstance(names, list) or not names
            or any(not isinstance(n, str) or not n for n in names)
            or len(set(names)) != len(names) or not isinstance(colors, dict)):
        raise ValueError("Labels need unique names and a color mapping.")
    import re
    if any(not re.fullmatch(r"#[0-9a-fA-F]{6}([0-9a-fA-F]{2})?", str(colors.get(n, '')))
           for n in names):
        raise ValueError("Each label needs a valid hex color.")

    return LabelSet(
        names=data["LABEL_LIST"],
        colors=data["LABEL_COLOR"],
    )
