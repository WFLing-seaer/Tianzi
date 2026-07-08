import asyncio
import datetime
import pathlib
from typing import cast

import awkward
from anyio import Path
from cachetools import TTLCache

from . import pqload, unitutil
from .colproto import ColProtoABC
from .headparser import TColSpec
from .schemas import Schemas

LEX_PATH = Path(__file__ or ".").parent / "lexicons"

LOAD_CACHE: TTLCache[str, Lexicon] = TTLCache[str, "Lexicon"](32, 1800)


all_lexicons = [p.stem for p in (pathlib.Path(__file__ or ".").parent / "lexicons").glob("*.pq")]
print(f"debug: {all_lexicons=} {__file__=} {LEX_PATH=}")


class Lexicon:
    schemas: Schemas
    fp: Path
    name: str

    @classmethod
    async def load(cls, name: str) -> Lexicon:
        if name not in all_lexicons:
            raise FileNotFoundError
        inst = cls.__new__(cls)
        inst.fp = await (LEX_PATH / f"{name}.pq").absolute()
        inst.name = name

        if name in LOAD_CACHE:
            return LOAD_CACHE[name]
        inst.schemas = Schemas(cast(dict[TColSpec, ColProtoABC], pqload.load(await asyncio.to_thread(awkward.from_parquet, inst.fp))))
        LOAD_CACHE[name] = inst
        return inst

    async def fmt_metadata(self):
        file_full_path = str(self.fp)
        file_stat = await self.fp.stat()
        file_size = file_stat.st_size
        file_size_num, file_size_amp = unitutil.byte2size(file_size)
        file_last_updated = file_stat.st_mtime
        file_last_updated_str = datetime.datetime.fromtimestamp(file_last_updated).strftime("%Y-%m-%d %H:%M:%S")
        array_size = sum(getattr(arr.data, "nbytes", 0) for arr in self.schemas.cols.values())
        arr_size_num, arr_size_amp = unitutil.byte2size(array_size)
        column_names = self.schemas.cols.keys()
        column_heads = self.schemas.orig_heads
        size = len(next(iter(self.schemas.cols.values())).data)
        size_num, size_amp = unitutil.num2ch(size)

        return f"{(
            f"词库「{self.name}」 {f"{size_num:.2f}" if size_amp else f"{size_num:d}"}{size_amp}词头\n"
            f"  文件：{file_full_path}\n"
            f"  {file_size_num:.2f}{file_size_amp} SSD {arr_size_num:.2f}{arr_size_amp} RAM\n"
            f"  上次更新：{file_last_updated_str}\n"
            "\n"
            f"  {"列、组：" if any(column_head.group for column_head in column_heads) else "列："}\n"
            f"  - {"\n  - ".join(
                f"{column_name}{
                    f"【{";".join(
                        f"{k}{
                            f"({",".join(v)})" if v is not None else ""
                        }"
                        for k,v in column_head.group.items())}】"
                    if column_head.group else ""
                }"
                for column_name, column_head in zip(column_names, column_heads)
            )}"
            )}{(
            "\n"
            "  Schema：\n"
            f"  - {"\n  - ".join(
                f"{gname} ({",".join(type(gschema).__name__ for gschema in gschemas)})"
                for gname, gschemas in self.schemas.schemas.items()
                if gschemas
            )
            }"
            ) if self.schemas.schemas else ""}"

    def query(self, qstr: str = "", colname: str | None = None):
        col = self.schemas.get_col(colname)
        return col.tostr(col.data[self.schemas.query_pop(qstr)])

    def __getitem__(self, key: slice[str, int, None]):
        col = self.schemas.get_col(key.start)
        return col.tostr(col.data[key.stop])


async def query(lexname: str, qstr: str = "", colname: str | None = None):
    return (await Lexicon.load(lexname)).query(qstr, colname)
