"""Speaker (animal) list of the ECAPA-TDNN classifier, loaded from a JSON file.

The classifier's output index i means "speakers[i]". That mapping is NOT stored in the model weights, so it
has to be supplied: this file is the single source of truth. Keep it next to the weights as `current.json`
(found automatically), or choose any other list in the GUI.

File format (a plain JSON list is accepted too):
    {"name": "current", "note": "...", "speakers": ["Bachelor-cut-in-40", "FB", ...]}

ORDER MATTERS: the position in the file is the class index. The classifier was trained with the names in
alphabetical order (the app's code always did `sorted(...)`), so the shipped lists are written that way.
If a classifier was trained with another order, put the names in that order here.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

SPEAKER_LIST_FILENAME = "current.json"


class SpeakerListError(ValueError):
    """The speaker list cannot be used. The message is written for the person using the GUI."""


def find_speaker_list_for_weights(weights_path) -> Path | None:
    """`current.json` in the same folder as the classifier weights, else None."""
    if not weights_path:
        return None
    p = Path(weights_path).parent / SPEAKER_LIST_FILENAME
    return p if p.is_file() else None


@dataclass(frozen=True)
class SpeakerList:
    speakers: tuple
    name: str = ""
    source: str = ""

    def __len__(self) -> int:
        return len(self.speakers)

    @property
    def id_to_speaker(self) -> dict:
        return dict(enumerate(self.speakers))

    @property
    def speaker_to_id(self) -> dict:
        return {s: i for i, s in enumerate(self.speakers)}

    def describe(self, max_names: int = 5) -> str:
        shown = ", ".join(self.speakers[:max_names])
        more = f" (+{len(self) - max_names} more)" if len(self) > max_names else ""
        label = f"'{self.name}': " if self.name else ""
        return f"{label}{len(self)} speakers: {shown}{more}  |  class index = position in the file"

    @classmethod
    def from_dict(cls, data, source: str = "") -> "SpeakerList":
        name = Path(source).stem if source else ""
        if isinstance(data, dict):
            name = str(data.get("name") or name)
            data = data.get("speakers")
        if not isinstance(data, list):
            raise SpeakerListError('A speaker list is a JSON list of names, or an object with a "speakers" list.')
        names = []
        for i, raw in enumerate(data):
            if not isinstance(raw, str) or not raw.strip():
                raise SpeakerListError(f"Entry {i + 1} is not a usable name ({raw!r}).")
            n = raw.strip()
            if re.search(r"[\t\r\n]", n):
                raise SpeakerListError(f"Entry {i + 1} ({n!r}) contains a tab or line break; label files are "
                                       f"tab-separated.")
            names.append(n)
        if not names:
            raise SpeakerListError("The speaker list is empty.")
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise SpeakerListError(f"Duplicate speaker name(s): {', '.join(dupes)}.")
        return cls(speakers=tuple(names), name=name, source=source)

    @classmethod
    def from_file(cls, path) -> "SpeakerList":
        p = Path(path)
        if not p.is_file():
            raise SpeakerListError(f"Speaker list not found: {p}")
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise SpeakerListError(f"Could not read {p.name}: {e}")
        return cls.from_dict(data, source=str(p))


def explain_class_count_mismatch(error_text: str, n_listed: int) -> str | None:
    """Turn PyTorch's 'size mismatch ... shape torch.Size([A, ...]) from checkpoint ... current model is
    torch.Size([B, ...])' into a sentence about the speaker list. None if the error is something else."""
    if "size mismatch" not in error_text:
        return None
    m = re.search(r"copying a param with shape torch\.Size\(\[(\d+)[^\]]*\]\) from checkpoint, "
                  r"the shape in current model is torch\.Size\(\[(\d+)", error_text)
    if m and int(m.group(2)) == n_listed:
        return (f"The classifier weights were trained for {m.group(1)} speakers, but the speaker list has "
                f"{n_listed}. Choose the speaker list that belongs to these weights.")
    return ("The classifier weights do not fit the speaker list (different number of speakers). Choose the "
            "speaker list that belongs to these weights.")
