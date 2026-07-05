from collections import defaultdict
from typing import TypeVar

D = TypeVar("D")


class Cache[T]:
    def __init__(self, name: str):
        self.name = name
        self.caches: defaultdict[str, dict[str, T]] = defaultdict(dict[str, T])

    def __setitem__(self, slc: slice[str, str, None], value: T):
        field, name = slc.start, slc.stop
        if not name:
            return
        if field not in self.caches:
            self.caches[field] = {}
        self.caches[field][name] = value

    def __getitem__(self, slc: slice[str, str, None]) -> T:
        field, name = slc.start, slc.stop
        return self.caches.get(field, {})[name]

    def get(self, field: str, name: str, default: D = None) -> T | D:
        return self.caches.get(field, {}).get(name, default)

    def clear(self, field: str | None = None):
        if field is None:
            self.caches.clear()
        elif field in self.caches:
            self.caches[field].clear()

    def __eq__(self, other: Cache):
        return self.name == other.name and self.caches == other.caches
