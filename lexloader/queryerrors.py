# 查询串错误类型：各Schema实现抛出，Schemas查询层与translator共享。
# 三者都是SyntaxError的子类，translator据此分配E43.4/E60.8/E73.21c；
# 普通SyntaxError仍然表示“查询不适用于该词库”（E43.3）。


class QuerySyntaxError(SyntaxError):
    """查询串语法无效（E43.4）。"""


class QueryTooComplexError(SyntaxError):
    """查询串超出复杂度上限（E60.8）。"""


class QueryRangeError(SyntaxError):
    """查询串中的索引/区间越界（E73.21c）。"""
