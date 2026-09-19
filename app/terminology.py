"""术语一致性审校: 在选定的断句方案上检查术语的规范写法与识别变体。

流程:
  1. 规范词与变体先做 NFKC 归一化与大小写归一(casefold), 再折叠空白;
     字幕侧同样逐词归一。
  2. 命中可以跨词但不允许跨字幕段: 在每段所辖词上构造两种归一文本 —— 词间
     带空格(英文 "dynamic programming" 一类)与不带空格(中文 "人工智能" 被
     切成 "人工"/"智能" 两个词的情形)—— 收集全部出现位置并映射回词下标。
  3. 同一位置出现嵌套/重叠候选时, 不按术语逐个匹配定夺, 而是把所有候选放进
     「加权区间调度」区间动态规划, 选择非重叠集合, 字典序目标为:
       (a) 覆盖归一字符数最多(不计空格);
       (b) 其中非规范(变体)命中最少;
       (c) 再按术语录入顺序(及变体顺序、起点)稳定唯一地决胜。

术语表校验(返回术语序号, 输入由调用方回显保留):
  - 规范词归一后为空: empty_canonical;
  - 同一归一识别串(无论作为谁的规范词或变体)归属两个不同术语: variant_conflict。
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass


def normalize(s: str) -> str:
    """NFKC 归一 + 大小写归一 + 空白折叠。"""
    return " ".join(unicodedata.normalize("NFKC", s).casefold().split())


@dataclass
class _Pattern:
    """一个识别串(规范词或某个变体)及其归一形态。"""
    term_index: int       # 术语序号, 1 起
    raw: str              # 用户录入原文(回显用)
    key: str              # 归一带空格形态
    compact: str          # 归一去全部空格形态
    is_canonical: bool
    order: int            # 在该术语内的录入顺序(规范词为 0, 变体从 1 起)


@dataclass
class _Candidate:
    s: int                # 段内词区间 [s, e)
    e: int
    chars: int            # 覆盖的归一字符数(不计空格)
    pat: _Pattern


def _error(kind: str, terms_echo: list[dict], **fields) -> dict:
    return {"ok": False, "error": {"kind": kind, **fields}, "terms": terms_echo}


def _build_patterns(terms: list[dict]):
    """解析术语表并做归属校验, 返回 (patterns_by_term, error)。

    同一归一识别串在不同术语下出现即归属冲突; 规范词为空立即报错。
    """
    owner: dict[str, int] = {}  # 归一识别串 -> 术语序号
    by_term: list[list[_Pattern]] = []
    for ti, t in enumerate(terms, start=1):
        canonical = t.get("canonical", "")
        if not str(canonical).strip():
            return None, _error("empty_canonical", terms, term_index=ti)
        pats: list[_Pattern] = []

        def add(raw: str, is_canonical: bool, order: int) -> dict | None:
            key = normalize(raw)
            if not key:  # 纯空白变体忽略
                return None
            if key in owner and owner[key] != ti:
                return _error("variant_conflict", terms, term_index=ti,
                              owner_index=owner[key], pattern=raw)
            owner.setdefault(key, ti)
            pats.append(_Pattern(ti, raw, key, key.replace(" ", ""),
                                 is_canonical, order))
            return None

        err = add(canonical, True, 0)
        if err:
            return None, err
        for vi, v in enumerate(t.get("variants", []), start=1):
            err = add(v, False, vi)
            if err:
                return None, err
        by_term.append(pats)
    return by_term, None


def _occurrence_spans(haystack: str, sub: str, word_at: list[int]):
    """在 haystack 中收集 sub 的全部出现, 返回 (首词, 尾词后一位置) 集合。

    word_at[pos] 给出 haystack 每个字符所属的段内词下标; sub 自身不含首尾
    空格, 因此起终点必落在词字符上。
    """
    if not sub:
        return set()
    out = set()
    start = haystack.find(sub)
    while start != -1:
        out.add((word_at[start], word_at[start + len(sub) - 1] + 1))
        start = haystack.find(sub, start + 1)
    return out


def _select(cands: list[_Candidate], m: int) -> list[_Candidate]:
    """区间动态规划: 在段内 m 个词上选非重叠候选集合。

    每个位置 j 只保留一个最优状态(键取最大):
      (覆盖字符数, -变体命中数, 决胜元组);
    决胜元组由各入选候选的 (-术语序号, -种类(规范0/变体1), -术语内顺序, -起点)
    顺序拼接: 录入顺序靠前、规范写法优先时稳定胜出; 两个不同候选集合的
    字符数与变体数即便相同, 决胜元组也必不同, 结果唯一确定。
    """
    by_end: list[list[_Candidate]] = [[] for _ in range(m + 1)]
    for c in cands:
        by_end[c.e].append(c)

    dp = [None] * (m + 1)
    take: list[_Candidate | None] = [None] * (m + 1)
    dp[0] = (0, 0, ())
    for j in range(1, m + 1):
        best = dp[j - 1]  # 第 j-1 个词不被任何候选覆盖
        best_cand = None
        for c in by_end[j]:
            p = c.pat
            prev = dp[c.s]
            tie = prev[2] + ((-p.term_index, -(0 if p.is_canonical else 1),
                              -p.order, -c.s),)
            key = (prev[0] + c.chars,
                   prev[1] - (0 if p.is_canonical else 1), tie)
            if best is None or key > best:
                best, best_cand = key, c
        dp[j], take[j] = best, best_cand

    chosen: list[_Candidate] = []
    j = m
    while j > 0:
        c = take[j]
        if c is None:
            j -= 1
        else:
            chosen.append(c)
            j = c.s
    chosen.reverse()
    return chosen


def review(words: list[dict], segment_ranges: list[list[int]],
           terms: list[dict]) -> dict:
    """对给定断句方案(段词范围序列)执行术语一致性审校。"""
    terms_echo = [{"canonical": t.get("canonical", ""),
                   "variants": list(t.get("variants", []))} for t in terms]
    n = len(words)

    ranges: list[tuple[int, int]] = []
    prev = -1
    for r in segment_ranges:
        a, b = int(r[0]), int(r[1])
        if a < 0 or b >= n or a > b:
            raise ValueError(f"审校段词范围 [{a}, {b}] 非法(合法 0–{n - 1})")
        if a <= prev:
            raise ValueError(f"审校段词范围 [{a}, {b}] 与前序段重叠或乱序")
        prev = b
        ranges.append((a, b))

    by_term, err = _build_patterns(terms_echo)
    if err is not None:
        return err

    # 逐词归一(NFKC 可能在词内引入空白), 供两种段文本拼接
    word_norms: list[str] = []
    for w in words:
        word_norms.append(normalize(w["text"]))

    seg_results = []
    agg: dict[int, dict] = {}
    total_canonical = total_variant = 0

    for seg_index, (wa, wb) in enumerate(ranges):
        nwords = wb - wa + 1
        # 带空格形态(词间一个空格)与无空格形态, 及字符 -> 段内词下标映射
        spaced_parts, compact_parts = [], []
        compact_map = []
        for li in range(nwords):
            norm = word_norms[wa + li]
            spaced_parts.append(norm)
            for ch in norm:
                if not ch.isspace():
                    compact_parts.append(ch)
                    compact_map.append(li)
        spaced = " ".join(spaced_parts)
        # 词间连接空格归左词(模式首尾无空格, 不会在连接空格上起止)
        word_at_spaced: list[int] = []
        for li, part in enumerate(spaced_parts):
            if li:
                word_at_spaced.append(li - 1)
            word_at_spaced.extend([li] * len(part))
        compact = "".join(compact_parts)

        cands: list[_Candidate] = []
        seen_span: set[tuple] = set()  # (术语, 顺序, s, e) 跨两种形态去重
        for pats in by_term:
            for p in pats:
                spans = _occurrence_spans(spaced, p.key, word_at_spaced)
                spans |= _occurrence_spans(compact, p.compact, compact_map)
                for s, e in spans:
                    ident = (p.term_index, p.order, s, e)
                    if ident in seen_span:
                        continue
                    seen_span.add(ident)
                    cands.append(_Candidate(s, e, len(p.compact), p))

        hits = []
        for c in _select(cands, nwords):
            p = c.pat
            g_first, g_last = wa + c.s, wa + c.e - 1
            surface = " ".join(words[wi]["text"] for wi in range(g_first, g_last + 1))
            hit = {
                "term_index": p.term_index,
                "canonical": terms_echo[p.term_index - 1]["canonical"],
                "kind": "canonical" if p.is_canonical else "variant",
                "variant": None if p.is_canonical else p.raw,
                "word_range": [g_first, g_last],
                "text": surface,
            }
            hits.append(hit)
            rec = agg.setdefault(p.term_index, {
                "term_index": p.term_index,
                "canonical_count": 0, "variant_count": 0,
                "variants": {},
            })
            if p.is_canonical:
                rec["canonical_count"] += 1
                total_canonical += 1
            else:
                rec["variant_count"] += 1
                total_variant += 1
                rec["variants"][p.raw] = rec["variants"].get(p.raw, 0) + 1

        seg_results.append({
            "index": seg_index,
            "word_range": [wa, wb],
            "hits": hits,
        })

    term_summary = []
    for ti in sorted(agg):
        rec = agg[ti]
        order_list = terms_echo[ti - 1]["variants"]
        term_summary.append({
            "term_index": rec["term_index"],
            "canonical": terms_echo[ti - 1]["canonical"],
            "canonical_count": rec["canonical_count"],
            "variant_count": rec["variant_count"],
            # 按变体在术语中的录入顺序(取首次出现位置)稳定列出
            "variants": sorted(
                ({"variant": v, "count": n} for v, n in rec["variants"].items()),
                key=lambda d: order_list.index(d["variant"])),
        })

    return {
        "ok": True,
        "canonical_count": total_canonical,
        "variant_count": total_variant,
        "hit_count": total_canonical + total_variant,
        "segments": seg_results,
        "terms": term_summary,
        "inconsistent": [ti for ti in sorted(agg) if agg[ti]["variant_count"] > 0],
    }
