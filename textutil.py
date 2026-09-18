from logging import getLogger

logger = getLogger("textutil")


def glued(s: str) -> str:
    return "\u2060".join(s).replace("\u2060\n", "\n").replace("\n\u2060", "\n").replace("\u2060 \u2060", "\u00a0")


def find_outmost_bracket(bracket: tuple[str, str], s: str) -> list[tuple[int, int]]:
    subs: list[tuple[int, int]] = []
    L, R = bracket
    ll, rl = len(L), len(R)
    i = 0
    while True:
        i = s.find(L, i)
        if i < 0:
            return subs
        stack = 1
        start = i
        i += ll
        while stack > 0:
            j = s.find(R, i)
            k = s.find(L, i)
            if j < 0:
                return subs
            if 0 <= k < j:
                stack += 1
                i = k + ll
            else:
                stack -= 1
                i = j + rl
                if stack == 0:
                    subs.append((start, i))
