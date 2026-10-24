# -*- coding: utf-8 -*-
"""
图 3：MeanSparse 稀疏化强度超参数 alpha 的搜索曲线
--------------------------------------------------
读  : results_ms/alpha_search.json
出  : results_ms/fig3_alpha_search.pdf   （矢量图，黑白打印可辨）

黑白可辨的做法：两条线都用黑色，靠【线型 + 标记形状】区分
    实线 + 圆圈  -> AT + MeanSparse
    虚线 + 方块  -> AT+SWA + MeanSparse
"""

import json
import os

import matplotlib
matplotlib.use("Agg")            # 无界面环境必须有，否则卡死
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "results_ms", "alpha_search.json")
OUT = os.path.join(HERE, "results_ms", "fig3_alpha_search.pdf")

# ---------- 中文字体（Windows）----------
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    if not os.path.exists(SRC):
        raise SystemExit("找不到 %s，请先跑 mean_sparse_repro.py" % SRC)

    with open(SRC, encoding="utf-8") as f:
        data = json.load(f)

    # 5.8in 宽：跨栏 figure* + width=0.8\textwidth 时接近 1:1，字不会被压小
    fig, ax = plt.subplots(figsize=(5.8, 3.7), dpi=300)

    styles = [
        ("AT",     "-",  "o", "AT + MeanSparse"),
        ("AT+SWA", "--", "s", "AT+SWA + MeanSparse"),
    ]

    # 两条线的最优点都在 alpha=0.45 且 y 值接近，
    # 故一条标注在下方、一条在上方，避免文字叠在一起
    offsets = {"AT": (12, -22), "AT+SWA": (12, 14)}

    for key, ls, mk, label in styles:
        rows = data[key]
        xs = [r["alpha"] for r in rows]
        ys = [r["pgd10"] for r in rows]
        ax.plot(
            xs, ys,
            linestyle=ls, marker=mk,
            color="k",                 # 黑白：统一黑色
            linewidth=1.4, markersize=5,
            markerfacecolor="white",   # 空心标记，打印更清楚
            markeredgecolor="k",
            label=label,
        )

        # 标出最优点（只写 alpha，不写数字，避免与表 1 的完整评测值对不上）
        i = max(range(len(ys)), key=lambda j: ys[j])
        ax.annotate(
            r"最优 $\alpha$=%g" % xs[i],
            xy=(xs[i], ys[i]),
            xytext=offsets[key],
            textcoords="offset points",
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="k", linewidth=0.8),
        )
        print("%-7s 最优 alpha=%g  pgd10=%.2f" % (key, xs[i], ys[i]))

    ax.set_xlabel(r"稀疏化强度 $\alpha$")
    ax.set_ylabel("PGD-10 鲁棒准确率 (%)")
    ax.set_xticks([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50])
    ax.grid(True, linestyle=":", linewidth=0.6, color="0.6")
    # 图例放左上：最优点都在图右侧，放右下会被压住
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
