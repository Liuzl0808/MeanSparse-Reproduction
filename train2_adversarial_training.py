# -*- coding: utf-8 -*-
"""
E2 + E4：对抗训练复现 + SWA 改进实验
=====================================
一次运行，顺序训练三个模型并自动汇总成论文表 2/表 3 和图 3：

  1) Standard    标准训练（无防御）          —— 基线
  2) AT          PGD 对抗训练（Madry 2018）  —— 复现的防御方法
  3) AT + SWA    对抗训练 + 权重平均改进      —— 本文改进（Izmailov 2018 的 SWA 思想）

用法（paper-code 目录下）：
  python train2_adversarial_training.py            # 完整实验（约 40-60 分钟，挂机）
  python train2_adversarial_training.py --quick    # 冒烟测试（约 2 分钟）
  python train2_adversarial_training.py --selftest # 5 秒环境自检

输出：results_at/ 文件夹
  table2_3.csv            论文表 2/表 3（三种方法的 干净/FGSM/PGD 准确率）
  fig3_training_curves.png / .pdf   论文图 3（训练曲线，灰度矢量版为 .pdf）
  table2_3_meta.json      复现性信息（种子/版本/超参，论文附录直接抄）
  run_log.txt             完整日志

论文对应关系：
  - PGD 对抗训练损失函数 = 论文式(2) 的实现
  - SWA 权重平均 = 论文式(3) 的实现： w_swa <- n/(n+1) * w_swa + 1/(n+1) * w
  - 依据文献：Madry et al. ICLR 2018；Izmailov et al. UAI 2018 (SWA)
"""

import argparse
import copy
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- 参数 ----------
EPS = 8 / 255          # 扰动上限（与 demo1 一致）
ALPHA = 2 / 255        # PGD 步长
PGD_TRAIN_STEPS = 7    # 对抗训练时攻击步数（Madry 原文设置）
PGD_EVAL_STEPS = 10    # 评测时攻击步数（更严格）
SEED = 42

# 学习率：对抗训练的梯度噪声远大于标准训练，必须用更小的学习率 + 热身，
# 否则第一步的大梯度会把网络推入饱和死区，loss 卡在 ln10（=2.303）不再下降，
# 模型退化为对所有类别输出相同概率（准确率 10%）。这是实测确认的失败模式。
LR_STD = 0.05          # 标准训练峰值学习率
LR_AT = 0.02           # 对抗训练峰值学习率
WARMUP_EPOCHS = 1      # 热身轮数（从 0 线性升到峰值）
SWA_START_RATIO = 0.5  # AT+SWA：从后半程开始平均权重
SWA_LR_RATIO = 0.5     # AT+SWA：SWA 阶段的恒定学习率比例（0.5×峰值）


# ---------- 模型（与 demo1 相同结构，保证可比） ----------
class SmallCNN(nn.Module):
    """VGG 风格小网络，width 可调。输入 [B,3,32,32]，输出 10 类。"""

    def __init__(self, width: int = 32, num_classes: int = 10):
        super().__init__()
        w = width

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            block(3, w), block(w, w),
            nn.MaxPool2d(2),                      # 32 -> 16
            block(w, w * 2), block(w * 2, w * 2),
            nn.MaxPool2d(2),                      # 16 -> 8
            block(w * 2, w * 4), block(w * 4, w * 4),
            nn.MaxPool2d(2),                      # 8 -> 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(w * 4 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------- 数据 ----------
def get_loaders(root, batch_size, quick):
    import torchvision
    import torchvision.transforms as T

    tf = T.ToTensor()  # 保持 [0,1] 像素空间，扰动定义与 demo1 一致
    train = torchvision.datasets.CIFAR10(
        root, train=True, download=True, transform=tf)
    test = torchvision.datasets.CIFAR10(
        root, train=False, download=True, transform=tf)

    if quick:  # 冒烟测试只取一小部分
        g = torch.Generator().manual_seed(SEED)
        idx = torch.randperm(len(train), generator=g)[:5000].tolist()
        train = torch.utils.data.Subset(train, idx)
        idx2 = torch.randperm(len(test), generator=g)[:1000].tolist()
        test = torch.utils.data.Subset(test, idx2)

    tl = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True,
        num_workers=0, generator=torch.Generator().manual_seed(SEED))
    vl = torch.utils.data.DataLoader(
        test, batch_size=256, shuffle=False, num_workers=0)
    return tl, vl


# ---------- 攻击（论文式(1)的实现，与 demo1 同源） ----------
def fgsm_attack(model, x, y, eps):
    x_adv = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    grad = torch.autograd.grad(loss, x_adv)[0]
    return (x + eps * grad.sign()).clamp(0, 1).detach()


def pgd_attack(model, x, y, eps, alpha, steps):
    delta = torch.zeros_like(x).uniform_(-eps, eps)
    for _ in range(steps):
        delta.requires_grad_(True)
        loss = F.cross_entropy(model(x + delta), y)
        grad = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            delta = delta + alpha * grad.sign()
            delta = torch.clamp(delta, -eps, eps)
            delta = (x + delta).clamp(0, 1) - x
    return (x + delta).detach()


# ---------- 评测 ----------
@torch.no_grad()
def clean_acc(model, loader, max_batches=None):
    model.eval()
    hit = tot = 0
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x, y = x.cuda(), y.cuda()
        hit += (model(x).argmax(1) == y).sum().item()
        tot += y.numel()
    return 100.0 * hit / tot


def adv_acc(model, loader, attack, eps, alpha, steps, max_batches=None):
    """attack: 'fgsm' 或 'pgd'"""
    model.eval()
    hit = tot = 0
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x, y = x.cuda(), y.cuda()
        if attack == "fgsm":
            xa = fgsm_attack(model, x, y, eps)
        else:
            xa = pgd_attack(model, x, y, eps, alpha, steps)
        with torch.no_grad():
            hit += (model(xa).argmax(1) == y).sum().item()
        tot += y.numel()
    return 100.0 * hit / tot


def full_eval(model, loader, log):
    r = {}
    r["clean"] = clean_acc(model, loader)
    r["fgsm"] = adv_acc(model, loader, "fgsm", EPS, ALPHA, 1)
    r["pgd10"] = adv_acc(model, loader, "pgd", EPS, ALPHA, PGD_EVAL_STEPS)
    log(f"    clean={r['clean']:.2f}%  FGSM={r['fgsm']:.2f}%  "
        f"PGD-10={r['pgd10']:.2f}%")
    return r


# ---------- 训练 ----------
def train_model(mode, epochs, tl, vl, log, quick):
    """mode: 'standard' | 'at' | 'atswa'"""
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # 与 demo1 完整版保持同一模型宽度（width=64），保证表 1 与表 2/3 可比
    model = SmallCNN(width=32 if quick else 64).cuda()
    base_lr = LR_STD if mode == "standard" else LR_AT
    opt = torch.optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=5e-4)

    # 逐迭代调度：先热身 1 个 epoch 线性升到峰值，再余弦衰减到 0
    steps_per_epoch = max(1, len(tl))
    total_iters = epochs * steps_per_epoch
    warmup_iters = max(1, WARMUP_EPOCHS * steps_per_epoch)

    def lr_lambda(it):
        if it < warmup_iters:
            return (it + 1) / warmup_iters
        p = (it - warmup_iters) / max(1, total_iters - warmup_iters)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # SWA 设置：从后半程开始平均（覆盖最后 50% 的 epoch）
    swa_model = None
    swa_start = max(1, int(round(epochs * SWA_START_RATIO)))
    if mode == "atswa":
        swa_model = torch.optim.swa_utils.AveragedModel(model)

    # 逐迭代调度：先热身 1 个 epoch 线性升到峰值，再余弦衰减到 0；
    # AT+SWA 在 SWA 阶段改用恒定学习率（Izmailov 等 2018 的原始配方）：
    # 若继续余弦衰减到 0，各快照权重过于接近，平均后增益趋近于 0（已实测）。
    steps_per_epoch = max(1, len(tl))
    total_iters = epochs * steps_per_epoch
    warmup_iters = max(1, WARMUP_EPOCHS * steps_per_epoch)
    swa_start_iter = (swa_start - 1) * steps_per_epoch

    def lr_lambda(it):
        if it < warmup_iters:
            return (it + 1) / warmup_iters
        if mode == "atswa" and it >= swa_start_iter:
            return SWA_LR_RATIO
        p = (it - warmup_iters) / max(1, total_iters - warmup_iters)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    hist = {"loss": [], "robust": []}
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tot_loss = nb = 0
        for x, y in tl:
            x, y = x.cuda(), y.cuda()
            if mode == "standard":
                xin = x
            else:  # 'at' / 'atswa'：在扰动样本上训练（论文式(2)）
                # 攻击前向必须冻结 BN：否则攻击的多次前向会用对抗样本的
                # 统计量污染 running_mean/var，导致评测时（用 running stats）
                # 与训练时（用 batch stats）严重不一致。
                was_training = model.training
                model.eval()
                xin = pgd_attack(model, x, y, EPS, ALPHA, PGD_TRAIN_STEPS)
                model.train(was_training)
            opt.zero_grad()
            loss = F.cross_entropy(model(xin), y)
            loss.backward()
            opt.step()
            sched.step()
            tot_loss += loss.item()
            nb += 1

        if mode == "atswa" and ep >= swa_start:
            swa_model.update_parameters(model)

        # 每 epoch 在测试集子集上做一次快速鲁棒评测（画曲线用）
        sub = 4 if not quick else 2   # 4*256 = 1024 张
        rob = adv_acc(model, vl, "pgd", EPS, ALPHA, PGD_EVAL_STEPS,
                      max_batches=sub)
        hist["loss"].append(tot_loss / nb)
        hist["robust"].append(rob)
        log(f"    epoch {ep}/{epochs} | loss {tot_loss/nb:.4f} | "
            f"PGD-10(sub) {rob:.2f}% | {time.time()-t0:.0f}s")

    # SWA 收尾：用训练数据重估 BN 统计量（关键步骤，缺了准确率会虚低）
    if mode == "atswa" and swa_model.n_averaged > 0:
        log("    recalibrating BatchNorm for SWA model ...")
        torch.optim.swa_utils.update_bn(tl, swa_model, device="cuda")
        return model, swa_model, hist
    return model, None, hist


# ---------- 画图（图 3：训练曲线对比） ----------
def setup_cjk():
    import matplotlib
    matplotlib.use("Agg")
    for f in ("SimHei", "Microsoft YaHei", "Noto Sans CJK SC", "SimSun"):
        try:
            import matplotlib.font_manager as fm
            if any(f.lower() in x.name.lower() for x in fm.fontManager.ttflist):
                matplotlib.rcParams["font.sans-serif"] = [f]
                return True
        except Exception:
            pass
    return False


def plot_curves(hist_at, hist_swa, out_png, out_pdf, cjk):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    ep = range(1, len(hist_at["loss"]) + 1)

    axes[0].plot(ep, hist_at["loss"], "o-", color="0.35",
                 label="AT" if cjk else "AT")
    if hist_swa:
        ep2 = range(1, len(hist_swa["loss"]) + 1)
        axes[0].plot(ep2, hist_swa["loss"], "s--", color="0.0",
                     label="AT+SWA")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("train loss")
    axes[0].set_title("(a) Training Loss" if not cjk else "(a) 训练损失")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(ep, hist_at["robust"], "o-", color="0.35", label="AT")
    if hist_swa:
        ep2 = range(1, len(hist_swa["loss"]) + 1)
        axes[1].plot(ep2, hist_swa["robust"], "s--", color="0.0",
                     label="AT+SWA")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("PGD-10 accuracy (%)")
    axes[1].set_title("(b) Robust Accuracy" if not cjk else "(b) 鲁棒准确率")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_pdf)  # 灰度矢量版，论文用
    plt.close(fig)


# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: 5000 samples, 2 epochs")
    ap.add_argument("--outdir", default="results_at",
                    help="输出目录（默认 results_at）")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    logf = open(os.path.join(outdir, "run_log.txt"), "a",
                encoding="utf-8")

    def log(msg):
        print(msg)
        logf.write(msg + "\n")
        logf.flush()

    if args.selftest:
        import torchvision  # noqa
        x = torch.randn(8, 3, 32, 32, device="cuda", requires_grad=True)
        m = SmallCNN().cuda()
        m(x).sum().backward()
        log("SELF TEST PASSED (CUDA + backward OK)")
        return 0

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    epochs = 2 if args.quick else args.epochs
    bs = 64 if args.quick else args.batch_size
    tag = "QUICK" if args.quick else "FULL"

    log("=" * 60)
    log(f"  E2+E4 adversarial training suite  [{tag}]  {time.ctime()}")
    log(f"  eps={EPS:.5f} alpha={ALPHA:.5f} "
        f"train_steps={PGD_TRAIN_STEPS} eval_steps={PGD_EVAL_STEPS} "
        f"lr_std={LR_STD} lr_at={LR_AT} warmup={WARMUP_EPOCHS}ep")
    log("=" * 60)

    tl, vl = get_loaders("./data", bs, args.quick)

    results, hists = {}, {}

    # 1) Standard
    log("[1/3] Standard training (no defense) ...")
    m_std, _, _ = train_model("standard", epochs, tl, vl, log, args.quick)
    log("  final evaluation (Standard):")
    results["Standard"] = full_eval(m_std, vl, log)
    del m_std
    torch.cuda.empty_cache()

    # 2) AT (PGD adversarial training, Madry 2018)
    log("[2/3] PGD adversarial training (AT, Madry 2018) ...")
    m_at, _, hists["AT"] = train_model("at", epochs, tl, vl, log, args.quick)
    log("  final evaluation (AT):")
    results["AT"] = full_eval(m_at, vl, log)
    torch.save(m_at.state_dict(), os.path.join(
        outdir, "model_at.pt"))  # 留存：改进实验/补图用

    # 3) AT + SWA (本文改进)
    log("[3/3] AT + SWA (our improvement) ...")
    m_swa_base, m_swa, hists["AT+SWA"] = train_model(
        "atswa", epochs, tl, vl, log, args.quick)
    if m_swa is not None:
        log("  final evaluation (AT+SWA):")
        results["AT+SWA"] = full_eval(m_swa, vl, log)
        torch.save(m_swa.state_dict(), os.path.join(
            outdir, "model_atswa.pt"))
    else:
        log("  [WARN] SWA not triggered (too few epochs), skipping")

    # ---- 汇总表（论文表 2/表 3） ----
    log("-" * 60)
    log("  SUMMARY (paper Table 2/3)")
    log("-" * 60)
    with open(os.path.join(outdir, "table2_3.csv"),
              "w", encoding="utf-8-sig") as f:
        f.write("method,clean(%),FGSM(%),PGD-10(%)\n")
        for k, r in results.items():
            f.write(f"{k},{r['clean']:.2f},{r['fgsm']:.2f},"
                    f"{r['pgd10']:.2f}\n")
            log(f"  {k:<10} clean={r['clean']:.2f}  FGSM={r['fgsm']:.2f}"
                f"  PGD-10={r['pgd10']:.2f}")

    # ---- 图 3：训练曲线 ----
    cjk = setup_cjk()
    plot_curves(hists["AT"], hists.get("AT+SWA"),
                os.path.join(outdir, "fig3_training_curves.png"),
                os.path.join(outdir, "fig3_training_curves.pdf"), cjk)
    log(f"  plot font cjk={cjk}")
    log("  figure saved: fig3_training_curves.png / .pdf")

    # ---- 复现性信息 ----
    meta = {
        "seed": SEED, "epochs": epochs, "batch_size": bs,
        "quick": args.quick, "eps": EPS, "alpha": ALPHA,
        "pgd_train_steps": PGD_TRAIN_STEPS,
        "pgd_eval_steps": PGD_EVAL_STEPS,
        "optimizer": "SGD(momentum=0.9, wd=5e-4) + 1-epoch warmup + cosine",
        "lr_standard": LR_STD, "lr_adversarial": LR_AT,
        "swa_start_epoch": max(1, int(round(epochs * SWA_START_RATIO))),
        "swa_lr_ratio": SWA_LR_RATIO, "warmup_epochs": WARMUP_EPOCHS,
        "model": "SmallCNN(width=32)" if args.quick else "SmallCNN(width=64)",
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "results": results, "timestamp": time.ctime(),
    }
    with open(os.path.join(outdir, "table2_3_meta.json"),
              "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    log("=" * 60)
    log(f"  DONE. All outputs are in: {outdir}/")
    log("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
