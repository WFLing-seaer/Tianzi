import base64
import contextlib

import pathvalidate


def to_filename(name: str) -> str:
    with contextlib.suppress(UnicodeEncodeError):
        if pathvalidate.is_valid_filename(name, "windows", fs_encoding="gbk") and not name.startswith("B32"):
            return name
    return f"B32{base64.b32encode(name.encode("utf-8")).decode("ascii")}"


def from_filename(filename: str) -> str:
    if filename.startswith("B32"):
        return base64.b32decode(filename[3:].encode("ascii")).decode("utf-8")
    return filename
