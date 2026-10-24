# -*- coding: utf-8 -*-
"""
demo1_adversarial_examples.py
=====================================================================
论文：面向图像分类器的对抗性攻击与鲁棒防御方法研究
目的：第 1 个实验 —— 训练一个受害者模型，对其施加 FGSM 与 PGD 攻击，
      生成对抗样本可视化图（论文图 2 的素材）+ 准确率下降数据（表 1 的素材）。

对应论文中的公式：
    FGSM:  x_adv = x + eps * sign( grad_x L(f(x), y) )          ---- 式(1)
    PGD :  x_adv^{t+1} = Clip_{x,eps}( x_adv^t + alpha*sign(grad) )  ---- 式(2)

用法：
    python demo1_adversarial_examples.py --selftest     # 5 秒：只验证环境/显卡，不下载数据
    python demo1_adversarial_examples.py --quick        # 约 5 分钟：小样本快速跑通全流程
    python demo1_adversarial_examples.py                # 正式跑（默认 3 轮，约 10-20 分钟）

输出（全部落在本脚本同目录的 results/ 下）：
    fig2_adversarial_examples.png   彩色版（放幻灯片/预览用）
    fig2_adversarial_examples.pdf   灰度矢量版（放进论文用，黑白印刷友好）
    table1_accuracy.csv             干净/攻击后准确率数据（论文表 1 的素材）
    run_log.txt                     完整运行日志（复现性凭证，务必保留）
=====================================================================
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------
# 0. 固定随机种子 —— 这是论文可复现性的硬性要求
#    同一份代码、同一个种子，结果必须一致；论文里要写明用了多少个种子
# ---------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 关掉非确定性算法加速，牺牲一点速度换取「同种子同结果」
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

LOG_PATH = OUT / "run_log.txt"
_log_fh = None


def log(msg: str) -> None:
    """同时打印到屏幕和写进日志文件 —— 日志是论文复现性的凭证。"""
    global _log_fh
    line = msg
    print(line, flush=True)
    if _log_fh is None:
        _log_fh = open(LOG_PATH, "a", encoding="utf-8")
    _log_fh.write(line + "\n")
    _log_fh.flush()


# ---------------------------------------------------------------------
# 1. 参数
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Adversarial examples demo (FGSM / PGD)")
    p.add_argument("--selftest", action="store_true",
                   help="only verify torch/GPU installation, no dataset download")
    p.add_argument("--quick", action="store_true",
                   help="quick mode: small subset, 1 epoch, small model")
    p.add_argument("--epochs", type=int, default=3, help="training epochs (default 3)")
    p.add_argument("--bs", type=int, default=0,
                   help="batch size; 0 = auto by GPU memory")
    p.add_argument("--subset", type=int, default=0,
                   help="use only N training images; 0 = full 50000")
    p.add_argument("--eps", type=float, default=8/255,
                   help="L-inf perturbation budget (default 8/255)")
    p.add_argument("--steps", type=int, default=10, help="PGD steps (default 10)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", type=str, default=str(HERE / "data"))
    p.add_argument("--workers", type=int, default=0,
                   help="dataloader workers; 0 is safest on Windows")
    return p.parse_args()


# ---------------------------------------------------------------------
# 2. 环境自检（--selftest）
# ---------------------------------------------------------------------
def selftest() -> int:
    log("=" * 66)
    log("  SELF TEST : checking Python / PyTorch / GPU")
    log("=" * 66)
    log(f"  Python      : {sys.version.split()[0]}   ({sys.executable})")
    try:
        import torch
    except ImportError:
        log("  [FAIL] PyTorch is NOT installed.")
        log("         Run:  pip install torch torchvision --index-url "
            "https://download.pytorch.org/whl/cu128")
        return 1

    log(f"  PyTorch     : {torch.__version__}")
    log(f"  CUDA build  : {torch.version.cuda}")
    avail = torch.cuda.is_available()
    log(f"  CUDA avail  : {avail}")
    if avail:
        n = torch.cuda.device_count()
        log(f"  GPU count   : {n}")
        for i in range(n):
            p = torch.cuda.get_device_properties(i)
            log(f"    [{i}] {p.name} | VRAM {p.total_memory/1024**3:.1f} GB "
                f"| SM {p.multi_processor_count} | cc {p.major}.{p.minor}")
    else:
        log("  [WARN] GPU not available -> will run on CPU (much slower).")
        log("         If you DO have an NVIDIA card, install the CUDA build:")
        log("         pip install torch torchvision --index-url "
            "https://download.pytorch.org/whl/cu124")

    # 一次真实的前向+反向，确认能算
    try:
        dev = "cuda" if avail else "cpu"
        t = torch.randn(8, 3, 32, 32, device=dev, requires_grad=True)
        m = torch.nn.Conv2d(3, 16, 3, padding=1).to(dev)
        out = m(t)
        out.sum().backward()
        log(f"  Forward/backward on {dev.upper()} : OK")
        if avail:
            log(f"  VRAM allocated after test: "
                f"{torch.cuda.memory_allocated()/1024**2:.0f} MB")
    except Exception as e:
        msg = str(e)
        log(f"  [FAIL] forward/backward error: {msg[:300]}")
        if ("no kernel image" in msg or "sm_120" in msg
                or "not compatible" in msg or "blackwell" in msg.lower()):
            log("  >> 原因: RTX 50 系显卡(Blackwell 架构)需要 cu128 版 PyTorch,")
            log("     你装的大概率是 cu124 或 CPU 版。修复命令:")
            log("     pip install torch torchvision --index-url "
                "https://download.pytorch.org/whl/cu128")
        return 1

    try:
        import torchvision
        log(f"  torchvision : {torchvision.__version__}")
    except ImportError:
        log("  [WARN] torchvision NOT installed -> cannot load CIFAR-10.")
        log("         Run:  pip install torchvision")
        return 1

    try:
        import matplotlib
        log(f"  matplotlib  : {matplotlib.__version__}")
    except ImportError:
        log("  [WARN] matplotlib NOT installed -> cannot draw figures.")
        log("         Run:  pip install matplotlib")
        return 1

    log("-" * 66)
    log("  SELF TEST PASSED. Environment is ready.")
    log("  Next:  python demo1_adversarial_examples.py --quick")
    log("=" * 66)
    return 0


# ---------------------------------------------------------------------
# 3. 受害者模型：一个小型 VGG 风格 CNN
#    选它的原因：参数量小（可在一张消费级显卡上反复做消融实验），
#    且结构简单、方便在论文里用文字描述清楚。
# ---------------------------------------------------------------------
def build_model(num_classes: int = 10, width: int = 64):
    import torch.nn as nn

    def block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    class SmallCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                block(3, width),                # 32x32 -> 16x16
                block(width, width * 2),        # 16x16 -> 8x8
                block(width * 2, width * 4),    # 8x8   -> 4x4
            )
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(width * 4, num_classes)

        def forward(self, x):
            x = self.features(x)
            x = self.pool(x).flatten(1)
            return self.fc(x)

    return SmallCNN()


# ---------------------------------------------------------------------
# 4. 对抗攻击（论文式(1)、式(2) 的实现）
#    mean/std 是数据集归一化参数；攻击必须在「像素空间 [0,1]」里做，
#    所以要先反归一化再攻击，attack 完再重新归一化送进模型。
# ---------------------------------------------------------------------
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def denorm(x, mean, std):
    import torch
    m = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    return (x * s + m).clamp(0, 1)


def norm(x, mean, std):
    import torch
    m = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    return (x - m) / s


def fgsm_attack(model, x_pix, y, eps, mean, std):
    """单步 FGSM：x_adv = x + eps * sign(grad_x Loss)。对应论文式(1)。"""
    import torch.nn.functional as F
    x = x_pix.clone().detach().requires_grad_(True)
    logits = model(norm(x, mean, std))
    loss = F.cross_entropy(logits, y)
    grad, = __import__("torch").autograd.grad(loss, x)
    x_adv = (x + eps * grad.sign()).clamp(0, 1).detach()
    return x_adv


def pgd_attack(model, x_pix, y, eps, alpha, steps, mean, std, rand_init=True):
    """多步 PGD：迭代 FGSM + 投影回 eps 球。对应论文式(2)。"""
    import torch
    import torch.nn.functional as F
    if rand_init:
        x_adv = (x_pix + torch.empty_like(x_pix).uniform_(-eps, eps)).clamp(0, 1).detach()
    else:
        x_adv = x_pix.clone().detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(norm(x_adv, mean, std))
        loss = F.cross_entropy(logits, y)
        grad, = torch.autograd.grad(loss, x_adv)
        x_adv = x_adv.detach() + alpha * grad.sign()
        # 投影：既限制在 eps 球内，也限制在合法像素范围 [0,1]
        x_adv = torch.min(torch.max(x_adv, x_pix - eps), x_pix + eps).clamp(0, 1).detach()
    return x_adv


# ---------------------------------------------------------------------
# 5. 数据
# ---------------------------------------------------------------------
def loaders(args, bs):
    import torch
    from torch.utils.data import DataLoader, Subset
    import torchvision
    import torchvision.transforms as T

    tf = T.Compose([T.ToTensor()])          # 只转成 tensor，归一化在模型前做
    root = args.data_dir
    train_full = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=tf)
    test_full = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=tf)

    train_ds, test_ds = train_full, test_full
    if args.subset > 0:
        g = torch.Generator().manual_seed(args.seed)
        tr_idx = torch.randperm(len(train_full), generator=g)[:args.subset].tolist()
        te_idx = torch.randperm(len(test_full), generator=g)[:max(1000, args.subset // 5)].tolist()
        train_ds, test_ds = Subset(train_full, tr_idx), Subset(test_full, te_idx)
        log(f"  [subset mode] train={len(train_ds)}  test={len(test_ds)}")

    pin = torch.cuda.is_available()
    tr = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=args.workers,
                    pin_memory=pin, drop_last=True)
    te = DataLoader(test_ds, batch_size=min(bs, 250), shuffle=False, num_workers=args.workers,
                    pin_memory=pin)
    return tr, te, train_ds, test_ds


def auto_batch_size(explicit: int, quick: bool) -> int:
    import torch
    if explicit > 0:
        return explicit
    if not torch.cuda.is_available():
        return 32 if quick else 64
    vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    if vram >= 20:
        bs = 256
    elif vram >= 11:
        bs = 128
    elif vram >= 7:
        bs = 96
    else:
        bs = 64
    if quick:
        bs = min(bs, 64)
    return bs


# ---------------------------------------------------------------------
# 6. 训练 / 评测
# ---------------------------------------------------------------------
def train(model, tr, dev, epochs, lr=0.05):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * max(1, len(tr)), pct_start=0.2)
    crit = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        tot = cor = 0
        run = 0.0
        for x, y in tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out = model(norm(x, MEAN, STD))
            loss = crit(out, y)
            loss.backward()
            opt.step()
            sched.step()
            run += loss.item() * y.size(0)
            cor += (out.argmax(1) == y).sum().item()
            tot += y.size(0)
        log(f"  epoch {ep}/{epochs} | loss {run/max(1,tot):.4f} | "
            f"train_acc {100*cor/max(1,tot):.2f}% | {time.time()-t0:.1f}s")
    return model


def evaluate(model, loader, dev, attack=None, eps=0.0, steps=10, tag="clean"):
    """返回 (准确率, 该批次第一个样本的原始图/对抗图/标签) —— 图用于可视化。"""
    import torch
    import torch.nn.functional as F

    model.eval()
    tot = cor = 0
    demo = None
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        x_pix = x            # DataLoader 只做了 ToTensor，数据本身已是 [0,1] 像素空间
        if attack == "fgsm":
            x_adv = fgsm_attack(model, x_pix, y, eps, MEAN, STD)
        elif attack == "pgd":
            x_adv = pgd_attack(model, x_pix, y, eps, eps / max(1, steps) * 2.5,
                               steps, MEAN, STD)
        else:
            x_adv = x_pix
        with torch.no_grad():
            out = model(norm(x_adv, MEAN, STD))
        pred = out.argmax(1)
        cor += (pred == y).sum().item()
        tot += y.size(0)
        if demo is None:
            demo = (x_pix.detach().cpu(), x_adv.detach().cpu(),
                    y.detach().cpu(), pred.detach().cpu())
    acc = 100.0 * cor / max(1, tot)
    log(f"  [{tag:>6}] accuracy = {acc:.2f}%   (n={tot})")
    return acc, demo


# ---------------------------------------------------------------------
# 7. 画图（论文图 2）
# ---------------------------------------------------------------------
CIFAR10_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]
# 中文论文里图内标签用中文更易读
CIFAR10_NAMES_ZH = ["飞机", "汽车", "鸟", "猫", "鹿",
                    "狗", "蛙", "马", "船", "卡车"]


def cls_name(i, cjk):
    return CIFAR10_NAMES_ZH[i] if cjk else CIFAR10_NAMES[i]


def setup_cjk_font():
    """尽量用中文字体；找不到就退回英文标签（论文里图注是中文，图内可用英文）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for name in ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC", "SimSun"]:
        try:
            path = font_manager.findfont(font_manager.FontProperties(family=name),
                                         fallback_to_default=False)
            if path and os.path.exists(path):
                plt.rcParams["font.sans-serif"] = [name]
                plt.rcParams["axes.unicode_minus"] = False
                return True, name
        except Exception:
            continue
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    return False, "DejaVu Sans"


def draw_figure(demo_clean, demo_fgsm, demo_pgd, eps, out_png, out_pdf, cjk):
    """论文图 2：三行（原图 / FGSM / PGD），每张图标注模型预测类别。
    注意：matplotlib 的 ax.axis('off') 会连 ylabel 一起隐藏，所以行标签
    必须用 ax.text 手绘（这是本函数第一版行标签"消失"的原因）。"""
    import numpy as np
    import matplotlib.pyplot as plt

    def cells(demo):
        """返回 (图像张量, 真实标签, 预测标签) 或 None。"""
        if demo is None:
            return None
        x, xadv, y, pred = demo
        return x[:8].numpy(), y[:8].numpy(), pred[:8].numpy()

    rows = []
    if demo_clean is not None:
        rows.append(("原图" if cjk else "Clean", cells(demo_clean)))
        rows.append(("FGSM 攻击" if cjk else "FGSM", cells(demo_fgsm)))
        rows.append(("PGD-10 攻击" if cjk else "PGD-10", cells(demo_pgd)))
    else:
        rows = [("FGSM", cells(demo_fgsm)), ("PGD", cells(demo_pgd))]

    ncol = 4
    nrow = len(rows)
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(ncol * 1.28, nrow * 1.62 + 0.35))
    if nrow == 1:
        axes = np.array([axes])

    # 每个样本的真实标签（放在最下一行图的下方）
    true_names = None
    for _, cl in rows:
        if cl is not None:
            true_names = [cls_name(int(i), cjk) for i in cl[1]]
            break

    for r, (label, cl) in enumerate(rows):
        for c in range(ncol):
            ax = axes[r][c]
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if cl is None or c >= cl[0].shape[0]:
                continue
            imgs, ys, preds = cl
            ax.imshow(np.clip(np.transpose(imgs[c], (1, 2, 0)), 0, 1))
            # 预测类别（错判前加 × 前缀 —— 因为 SimHei 无粗体，
            # 灰度印刷时靠 × 而非字重区分，这是刻意的设计）
            ok = preds[c] == ys[c]
            mark = "" if ok else "×"
            ax.text(0.03, 0.97, mark + cls_name(int(preds[c]), cjk),
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=6.5,
                    color="black" if ok else "#b91c1c",
                    bbox=dict(facecolor="white", alpha=0.72,
                              edgecolor="none", pad=1.0))
            # 行标签：手绘（axis off 后 ylabel 不可见）
            if c == 0:
                ax.text(-0.05, 0.5, label, transform=ax.transAxes,
                        ha="right", va="center", fontsize=9)
            # 真实类别：只在最后一行下方
            if r == nrow - 1 and true_names is not None:
                ax.text(0.5, -0.09, f"真: {true_names[c]}" if cjk
                        else f"true: {true_names[c]}",
                        transform=ax.transAxes, ha="center", va="top",
                        fontsize=6.5, color="0.25")

    eps_disp = f"{eps:.4f}" if eps < 0.1 else f"{eps:.3f}"
    title = (f"Adversarial examples on CIFAR-10 (L-inf eps = 8/255 = {eps_disp})"
             if not cjk else
             f"CIFAR-10 上的对抗样本（L∞ 扰动预算 ε = 8/255 ≈ {eps_disp}）")
    fig.suptitle(title, fontsize=11)
    fig.subplots_adjust(left=0.105, right=0.995, top=0.90, bottom=0.085,
                        wspace=0.06, hspace=0.10)

    fig.savefig(out_png, dpi=200)
    # 灰度矢量 PDF —— 计算机学报模板要求「图黑白、矢量」
    try:
        for ax in fig.axes:
            for im in ax.get_images():
                arr = im.get_array()
                try:
                    g = arr[..., :3] @ np.array([0.299, 0.587, 0.114])
                    im.set_array(g)
                    im.set_cmap("gray")
                    im.set_clim(0, 1)
                except Exception:
                    pass
        fig.savefig(out_pdf)
    except Exception as e:
        log(f"  [warn] grayscale PDF failed: {e}")
    plt.close(fig)
    log(f"  figure saved: {out_png.name} / {out_pdf.name}")


# ---------------------------------------------------------------------
# 8. 主流程
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    if args.selftest:
        return selftest()

    log("")
    log("=" * 66)
    log("  EXPERIMENT 1 : Adversarial attacks on a CIFAR-10 classifier")
    log(f"  started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 66)
    log(f"  args: epochs={args.epochs} quick={args.quick} subset={args.subset} "
        f"eps={args.eps} steps={args.steps} seed={args.seed}")

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bs = auto_batch_size(args.bs, args.quick)
    log(f"  device={dev.upper()}  batch_size={bs}")
    if dev == "cuda":
        p = torch.cuda.get_device_properties(0)
        log(f"  GPU: {p.name} ({p.total_memory/1024**3:.1f} GB)")
    else:
        log("  [WARN] running on CPU - results are valid but slow.")
    log(f"  torch={torch.__version__}  cuda_build={torch.version.cuda}")

    if args.quick:
        if args.subset == 0:
            args.subset = 5000
        if args.epochs == 3:
            args.epochs = 1
    width = 32 if args.quick else 64

    # --- data ---
    log("-" * 66)
    log("  [1/4] loading CIFAR-10 ...")
    try:
        tr, te, train_ds, test_ds = loaders(args, bs)
    except Exception as e:
        log(f"  [FAIL] dataset error: {e}")
        log("  If the download stalls, get cifar-10-python.tar.gz manually from")
        log("  https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")
        log(f"  and place it into: {args.data_dir}\\cifar-10-batches-py\\ parent dir,")
        log("  then re-run. (torchvision expects the tar.gz in the data root)")
        return 1
    log(f"  train={len(train_ds)}  test={len(test_ds)}")

    # --- model + train ---
    log("-" * 66)
    log("  [2/4] building victim model ...")
    model = build_model(10, width).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    log(f"  model: SmallCNN(width={width})  params={nparam/1e6:.2f} M")

    log(f"  training {args.epochs} epoch(s) ...")
    t0 = time.time()
    train(model, tr, dev, args.epochs)
    train_time = time.time() - t0
    log(f"  training done in {train_time/60:.1f} min")

    # --- attacks ---
    log("-" * 66)
    log("  [3/4] evaluating clean accuracy and adversarial accuracy ...")
    results = {}
    acc_clean, demo_clean = evaluate(model, te, dev, None, tag="clean")
    results["clean"] = acc_clean

    eps = args.eps
    acc_fgsm, demo_fgsm = evaluate(model, te, dev, "fgsm", eps=eps,
                                   steps=1, tag="FGSM")
    results["fgsm"] = acc_fgsm

    acc_pgd, demo_pgd = evaluate(model, te, dev, "pgd", eps=eps,
                                 steps=args.steps, tag="PGD")
    results["pgd"] = acc_pgd

    drop_fgsm = acc_clean - acc_fgsm
    drop_pgd = acc_clean - acc_pgd
    log("-" * 66)
    log(f"  SUMMARY  clean={acc_clean:.2f}%  FGSM={acc_fgsm:.2f}% "
        f"(drop {drop_fgsm:.2f})  PGD={acc_pgd:.2f}% (drop {drop_pgd:.2f})")
    log("  >> 这就是论文要讨论的核心现象：干净准确率很高，")
    log("     但一个肉眼几乎看不出差异的扰动就能让准确率崩塌。")

    # --- save table ---
    log("-" * 66)
    log("  [4/4] saving results ...")
    csv_path = OUT / "table1_accuracy.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["attack", "eps", "steps", "accuracy(%)", "drop(vs clean)"])
        w.writerow(["None (clean)", 0, 0, f"{acc_clean:.2f}", "0.00"])
        w.writerow(["FGSM", f"{eps:.6f}", 1, f"{acc_fgsm:.2f}", f"{drop_fgsm:.2f}"])
        w.writerow(["PGD", f"{eps:.6f}", args.steps, f"{acc_pgd:.2f}", f"{drop_pgd:.2f}"])
    log(f"  table saved: {csv_path.name}")

    meta = {
        "seed": args.seed, "epochs": args.epochs, "batch_size": bs,
        "subset": args.subset, "eps": eps, "pgd_steps": args.steps,
        "device": dev, "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": (torch.cuda.get_device_properties(0).name if dev == "cuda" else "CPU"),
        "model": f"SmallCNN(width={width})", "params_M": round(nparam / 1e6, 3),
        "train_minutes": round(train_time / 60, 2), "results": results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (OUT / "table1_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    cjk, fontname = setup_cjk_font()
    log(f"  plot font: {fontname} (cjk={cjk})")
    draw_figure(demo_clean, demo_fgsm, demo_pgd, eps,
                OUT / "fig2_adversarial_examples.png",
                OUT / "fig2_adversarial_examples.pdf", cjk)

    log("=" * 66)
    log("  DONE. All outputs are in:")
    log(f"  {OUT}")
    log("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
