import random
from collections.abc import Callable

from .nameutil import Names

filters = Names[Callable[[str], str]]()


filters.named("lower", "小写")(str.lower)
filters.named("upper", "大写")(str.upper)
filters.named("title", "标题")(str.title)
filters.named("capitalize", "首字母大写")(str.capitalize)
filters.named("casefold")(str.casefold)
filters.named("swapcase", "交换大小写")(str.swapcase)
filters.named("strip")(str.strip)
filters.named("rstrip")(str.rstrip)
filters.named("lstrip")(str.lstrip)
filters.named("expandtabs")(str.expandtabs)


@filters.named("麻将", "麻将牌", "mahjong")
def mahjong(s: str) -> str:
    buf: list[str] = []
    retbuf: list[str] = []
    charset = {"m": "🀇🀈🀉🀊🀋🀌🀍🀎🀏", "p": "🀙🀚🀛🀜🀝🀞🀟🀠🀡", "s": "🀐🀑🀒🀓🀔🀕🀖🀗🀘", "z": "🀀🀁🀂🀃🀄🀅🀆"}
    typemap = {"万": "m", "萬": "m", "饼": "p", "筒": "p", "条": "s", "索": "s", "字": "z"}
    nummap = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    singlemap = {"东": "🀀", "南": "🀁", "西": "🀂", "北": "🀃", "中": "🀄", "发": "🀅", "白": "🀆"}
    for c in s:
        if c in typemap or c in charset:
            for bc in buf:
                try:
                    retbuf.append(charset[typemap.get(c, c)][int(nummap.get(bc, bc)) - 1])
                except ValueError, IndexError:
                    retbuf.append(bc)
            buf.clear()
        elif c in nummap or c in "123456789":
            buf.append(c)
        elif c in singlemap:
            retbuf.extend(buf)
            buf.clear()
            retbuf.append(singlemap[c])
        else:
            retbuf.extend(buf)
            buf.clear()
            retbuf.append(c)
    retbuf.extend(buf)
    return "".join(retbuf)


MORSE = {
    "A": ".-",
    "B": "-...",
    "C": "-.-.",
    "D": "-..",
    "E": ".",
    "F": "..-.",
    "G": "--.",
    "H": "....",
    "I": "..",
    "J": ".---",
    "K": "-.-",
    "L": ".-..",
    "M": "--",
    "N": "-.",
    "O": "---",
    "P": ".--.",
    "Q": "--.-",
    "R": ".-.",
    "S": "...",
    "T": "-",
    "U": "..-",
    "V": "...-",
    "W": ".--",
    "X": "-..-",
    "Y": "-.--",
    "Z": "--..",
    "0": "-----",
    "1": ".----",
    "2": "..---",
    "3": "...--",
    "4": "....-",
    "5": ".....",
    "6": "-....",
    "7": "--...",
    "8": "---..",
    "9": "----.",
}


@filters.named("摩斯", "摩尔斯", "摩斯码", "摩尔斯码", "摩斯电码", "摩尔斯电码", "morse")
def morse(s: str) -> str:
    return "/".join(MORSE.get(c.upper(), c) for c in s)


@filters.named("乱序", "shuffle")
def shuffle(s: str) -> str:
    l = list(s)
    random.shuffle(l)
    return "".join(l)
