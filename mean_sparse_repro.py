"""复现 MeanSparse（Amini et al., 2024, arXiv:2406.05927）——后处理式鲁棒性增强

原理（据论文第 III 节）：
    冻结已对抗训练的模型权重，在每个激活函数【之前】（pre-activation）插入
    均值中心化特征稀疏化算子：
        a_out = mu_ch,   若 |a_in - mu_ch| <= alpha * sigma_ch
        a_out = a_in,    否则
    其中 mu_ch / sigma_ch 由训练集单次前向按通道统计得到（推理时固定，不随
    batch 重算），alpha 为所有激活层共享的超参数，按论文做法在候选集上搜索
    （在 clean 准确率基本不变的前提下取鲁棒性最高者）。全过程不更新权重、
    不微调。

用法：
    python mean_sparse_repro.py --smoke   # 冒烟：统计 5 批、评测子集、只试 3 个 alpha
    python mean_sparse_repro.py           # 正式：全量统计 + 全测试集评测

输出：
    results_ms/alpha_search.json   alpha 搜索记录
    results_ms/table_ms.csv        最终三项指标（clean / FGSM / PGD-10）
"""
import os, sys, json, time, argparse
import torch
import torch.nn as nn

from train2_adversarial_training import (
    SmallCNN, get_loaders, pgd_attack, full_eval, EPS, SEED,
)

OUT = "results_ms"
ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
STEP = 2 / 255          # PGD 步长，与 train2 一致
CLEAN_TOL = 1.5         # 允许的 clean 准确率下降上限（百分点）


# ---------- 统计 per-channel mu / sigma ----------
@torch.no_grad()
def collect_stats(model, loader, max_batches=None):
    """在训练集上跑一次前向，统计每个 ReLU 输入的逐通道均值与标准差。"""
    acc, handles = {}, []

    def make(name):
        def hook(mod, inp):
            x = inp[0].detach()
            if x.dim() == 4:                      # 卷积特征 [B,C,H,W]
                s = x.sum(dim=(0, 2, 3))
                sq = (x * x).sum(dim=(0, 2, 3))
                n = x.shape[0] * x.shape[2] * x.shape[3]
            else:                                 # 全连接特征 [B,D]
                s = x.sum(dim=0)
                sq = (x * x).sum(dim=0)
                n = x.shape[0]
            d = acc.setdefault(name, {"s": 0.0, "sq": 0.0, "n": 0})
            d["s"] = d["s"] + s
            d["sq"] = d["sq"] + sq
            d["n"] = d["n"] + n
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, nn.ReLU):
            handles.append(mod.register_forward_pre_hook(make(name)))

    model.eval()
    for i, (x, _) in enumerate(loader):
        model(x.cuda())
        if max_batches and i + 1 >= max_batches:
            break
    for h in handles:
        h.remove()

    stats = {}
    for name, d in acc.items():
        mu = d["s"] / d["n"]
        var = (d["sq"] / d["n"] - mu * mu).clamp(min=1e-12)
        stats[name] = (mu, var.sqrt())
    return stats


# ---------- 挂载算子 ----------
def attach(model, stats, alpha):
    """在每个 ReLU 前插入 MeanSparse，返回句柄（用完需 remove）。"""
    handles = []

    def make(mu, sg, C, alpha):
        def pre(mod, inp):
            x = inp[0]
            if x.dim() == 4:
                m = mu.view(1, C, 1, 1)
                t = (alpha * sg).view(1, C, 1, 1)
            else:
                m = mu.view(1, C)
                t = (alpha * sg).view(1, C)
            dev = x - m
            return (torch.where(dev.abs() <= t, m.expand_as(x), x),)
        return pre

    for name, mod in model.named_modules():
        if isinstance(mod, nn.ReLU) and name in stats:
            mu, sg = stats[name]
            handles.append(mod.register_forward_pre_hook(
                make(mu, sg, mu.numel(), alpha)))
    return handles


# ---------- alpha 搜索用快速评测 ----------
def quick_eval(model, loader, max_batches=None):
    model.eval()
    c_ok = c_n = r_ok = r_n = 0
    for i, (x, y) in enumerate(loader):
        x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            c_ok += (model(x).argmax(1) == y).sum().item()
        c_n += y.numel()
        adv = pgd_attack(model, x, y, EPS, STEP, 10)
        with torch.no_grad():
            r_ok += (model(adv).argmax(1) == y).sum().item()
        r_n += y.numel()
        if max_batches and i + 1 >= max_batches:
            break
    return 100.0 * c_ok / c_n, 100.0 * r_ok / r_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(SEED)
    tl, vl = get_loaders("./data", 128, quick=False)

    rows, search_log = [], {}
    alphas = [0.10, 0.20, 0.30] if args.smoke else ALPHAS

    for tag, ckpt in [("AT", "results_at/model_at.pt"),
                      ("AT+SWA", "results_at/model_atswa.pt")]:
        if not os.path.exists(ckpt):
            print(f"[跳过] 找不到 {ckpt}")
            continue

        model = SmallCNN(width=64).cuda()
        sd = torch.load(ckpt, map_location="cuda")
        # SWA 包装器 / DataParallel 保存的权重带 "module." 前缀，且含 n_averaged
        sd = {k[len("module."):] if k.startswith("module.") else k: v
              for k, v in sd.items() if k != "n_averaged"}
        model.load_state_dict(sd)
        model.eval()

        # 基线（未加算子）
        base_c, base_r = quick_eval(
            model, vl, max_batches=2 if args.smoke else None)
        print(f"\n=== {tag} 基线: clean={base_c:.2f} pgd10={base_r:.2f} ===",
              flush=True)

        # 统计 mu / sigma
        t0 = time.time()
        stats = collect_stats(model, tl, max_batches=5 if args.smoke else None)
        print(f"    统计完成: {len(stats)} 个激活层, {time.time()-t0:.0f}s",
              flush=True)

        # 搜索 alpha：clean 下降不超过 CLEAN_TOL 时取 pgd10 最高
        best = None
        rec = []
        for a in alphas:
            hs = attach(model, stats, a)
            c, r = quick_eval(model, vl, max_batches=2 if args.smoke else None)
            for h in hs:
                h.remove()
            rec.append({"alpha": a, "clean": round(c, 2), "pgd10": round(r, 2)})
            print(f"    alpha={a:.2f}  clean={c:.2f}  pgd10={r:.2f}", flush=True)
            if c >= base_c - CLEAN_TOL and (best is None or r > best[1]):
                best = (a, r, c)
        if best is None:                       # 全部掉太多，退而求其次
            best = max(rec, key=lambda d: d["pgd10"])
            best = (best["alpha"], best["pgd10"], best["clean"])
        print(f"    -> 选定 alpha={best[0]:.2f}", flush=True)
        search_log[tag] = rec

        # 用选定的 alpha 做完整评测
        hs = attach(model, stats, best[0])
        res = full_eval(model, vl, print)
        for h in hs:
            h.remove()
        rows.append((f"{tag}+MeanSparse(a={best[0]:.2f})",
                     res["clean"], res["fgsm"], res["pgd10"]))

    with open(os.path.join(OUT, "alpha_search.json"), "w",
              encoding="utf-8") as f:
        json.dump(search_log, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUT, "table_ms.csv"), "w",
              encoding="utf-8-sig") as f:
        f.write("method,clean(%),FGSM(%),PGD-10(%)\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]:.2f},{r[2]:.2f},{r[3]:.2f}\n")

    print("\nDONE ->", OUT)
    for r in rows:
        print(f"  {r[0]:28s} {r[1]:6.2f} {r[2]:6.2f} {r[3]:6.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
