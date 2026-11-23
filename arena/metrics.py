import math


def score(win, draw, n):
    # score = (thắng + 0,5 × hòa) / tổng ván (theo đề)
    return (win + 0.5 * draw) / n if n else 0.0


def wilson(successes, n, z=1.96):
    # Khoảng tin cậy Wilson 95% cho một tỷ lệ. Khi hòa tính nửa thắng thì chỉ là xấp xỉ.
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))  # chặn sai số số thực


def percentile(values, q):
    # Phân vị theo cách "nearest rank"; q trong (0, 1].
    s = sorted(values)
    return s[min(len(s) - 1, math.ceil(q * len(s)) - 1)]
