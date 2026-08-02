import atexit
import csv
import pathlib
from itertools import chain

import thefuzz.process

fp = pathlib.Path(__file__).parent / "alias.tsv"

ALIAS: dict[str, str] = {}

if fp.exists():
    with fp.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            key, val = row[0], row[1]
            ALIAS[key] = val


def _save() -> None:
    with fp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for key, val in ALIAS.items():
            writer.writerow([key, val])


atexit.register(_save)


def getitem(key: str) -> str | None:
    return ALIAS.get(key)


def setitem(key: str, val: str) -> None:
    ALIAS[key] = val


def delitem(key: str) -> None:
    del ALIAS[key]


def fuzzkey(key: str) -> str | None:
    fuzz = thefuzz.process.extractOne(key, ALIAS.keys())
    return fuzz and fuzz[0]


def fuzzkeys(key: str, k: int, tr: int):
    fuzzs = thefuzz.process.extract(key, ALIAS.keys(), limit=k)
    return fuzzs and [fuzz[0] for fuzz in fuzzs if fuzz[1] >= tr]


def fuzzval(val: str) -> str | None:
    fuzz = thefuzz.process.extractOne(val, ALIAS.values())
    return fuzz and fuzz[0]


def fuzzvals(val: str, k: int, tr: int):
    fuzzs = thefuzz.process.extract(val, ALIAS.values(), limit=k)
    return fuzzs and [fuzz[0] for fuzz in fuzzs if fuzz[1] >= tr]


def fuzzkv(korv: str) -> str | None:
    fuzz = thefuzz.process.extractOne(korv, chain(ALIAS.keys(), ALIAS.values()))
    return fuzz and fuzz[0]


def fuzzkvs(korv: str, k: int, tr: int):
    fuzzs = thefuzz.process.extract(korv, chain(ALIAS.keys(), ALIAS.values()), limit=k)
    return fuzzs and [fuzz[0] for fuzz in fuzzs if fuzz[1] >= tr]
