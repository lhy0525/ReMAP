import os
import logging
import argparse
import torch
from torchvision import transforms
from clip.clip import load, tokenize, available_models
from util.utils import eval_all_class, visualize
import csv
from datetime import datetime
import torch.nn.functional as F
from dataset import *


def _ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def setup_logger(out_dir, log_file=''):
    """
    创建既写入文件又输出到控制台的 logger。
    日志格式：YYYY-MM-DD HH:MM:SS,mmm - INFO - message
    """
    _ensure_dir(out_dir)
    if not log_file:
        ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        log_file = f"test_results_{ts}.log"
    log_path = os.path.join(out_dir, log_file)

    logger = logging.getLogger("eval")
    logger.setLevel(logging.INFO)
    # 避免重复添加 handler（例如在同一进程多次运行）
    logger.handlers = []

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info(f"[Logger] writing to {log_path}")
    return logger, log_path

def load_checkpoint(clip_model, path, device=None, strict=False):
    """
    健壮的权重加载（支持目录/文件，处理DDP的'module.'前缀；
    若位置编码shape不一致，自动插值到当前模型shape，避免报错）
    """

    if os.path.isdir(path):
        # 与训练保存约定一致（可以改成 last.pt 看你自己的保存）
        path = os.path.join(path, "last.pt")

    # 优先使用模型所在设备
    model_device = next(clip_model.parameters()).device
    target_device = torch.device(device) if device is not None else model_device
    map_loc = (lambda storage, loc: storage.cuda(target_device.index)) if target_device.type == "cuda" else "cpu"

    checkpoint = torch.load(path, map_location=map_loc)
    state = checkpoint.get("model_state_dict", checkpoint)

    # 去掉 DDP 前缀
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    # 对齐 visual.positional_embedding
    key = "visual.positional_embedding"
    if key in state and hasattr(clip_model, "visual") and hasattr(clip_model.visual, "positional_embedding"):
        pe_ckpt  = state[key]
        pe_model = clip_model.visual.positional_embedding
        if pe_ckpt.shape != pe_model.shape:
            with torch.no_grad():
                cls_token = pe_ckpt[:1, :]               # [1, C]
                grid_ckpt = pe_ckpt[1:, :]               # [Hc*Wc, C]
                C         = grid_ckpt.shape[-1]
                old_side  = int((grid_ckpt.shape[0]) ** 0.5)
                new_side  = int((pe_model.shape[0] - 1) ** 0.5)

                grid_ckpt = grid_ckpt.transpose(0, 1).reshape(1, C, old_side, old_side)
                grid_new  = F.interpolate(grid_ckpt, size=(new_side, new_side),
                                           mode="bilinear", align_corners=False)
                grid_new  = grid_new.reshape(1, C, new_side * new_side).transpose(1, 2).squeeze(0)

                pe_new = torch.cat([cls_token, grid_new], dim=0)
                pe_new = pe_new.to(dtype=pe_model.dtype, device=pe_model.device)
                state[key] = pe_new
                print(f"[load_checkpoint] resized positional_embedding {tuple(pe_ckpt.shape)} -> {tuple(pe_new.shape)}")

    clip_model.load_state_dict(state, strict=strict)
    return checkpoint

def build_all_test_datasets(args, clip_transform, target_transform):
    """与 main.py 同步构造全部可测数据集，并用 test_dataset 过滤"""
    test_dataset_mvtec   = MVTecDataset (root=args.data_dir, train=False, category=None, transform=clip_transform, gt_target_transform=target_transform)
    test_dataset_isic    = ISICDataset  (root=args.data_dir, train=False, category=None, transform=clip_transform, gt_target_transform=target_transform)
    test_dataset_clinic  = ClinicDBDataset(root=args.data_dir, train=False, category=None, transform=clip_transform, gt_target_transform=target_transform)
    test_dataset_colon   = ColonDBDataset (root=args.data_dir, train=False, category=None, transform=clip_transform, gt_target_transform=target_transform)
    test_dataset_visa    = VisaDataset   (root=args.data_dir, train=False, category=None, transform=clip_transform, gt_target_transform=target_transform)
    test_dataset_btad    = BTADDataset   (root=args.data_dir, train=False, category=None, transform=clip_transform, gt_target_transform=target_transform)
    test_dataset_dtd     = DTDDataset    (root=args.data_dir, train=False, category=None, transform=clip_transform, gt_target_transform=target_transform)
    test_dataset_brainmri= BrainMRIDataset(root=args.data_dir, train=False, category=None, transform=clip_transform, gt_target_transform=target_transform)
    test_dataset_br35h   = Br35HDataset  (root=args.data_dir, train=False, category=None, transform=clip_transform, gt_target_transform=target_transform)
    test_dataset_dagm    = DAGMDataset   (root=args.data_dir, train=False, category=None, transform=clip_transform, gt_target_transform=target_transform)
    test_dataset_kvasir  = KvasirDataset (root=args.data_dir, train=False, category=None, transform=clip_transform, gt_target_transform=target_transform)

    all_test_dataset_dict = {
        "mvtec":  test_dataset_mvtec,
        "visa":   test_dataset_visa,
        "btad":   test_dataset_btad,
        "dtd":    test_dataset_dtd,
        "dagm":   test_dataset_dagm,
        "isic":   test_dataset_isic,
        "clinic": test_dataset_clinic,
        "colon":  test_dataset_colon,
        "brainmri": test_dataset_brainmri,
        "br35h":  test_dataset_br35h,
        "kvasir": test_dataset_kvasir,
    }

    # 为空：评测全部；不空：只评测指定
    if len(args.test_dataset) < 1:
        test_dataset_dict = all_test_dataset_dict
    else:
        test_dataset_dict = {name: all_test_dataset_dict[name] for name in args.test_dataset}

    # 和 main.py 一样：把训练用的数据集从列表里去掉
    if args.dataset in test_dataset_dict:
        del test_dataset_dict[args.dataset]

    return test_dataset_dict

def main():
    parser = argparse.ArgumentParser("Evaluate only (no training)")
    parser.add_argument('--clip_download_dir', type=str, default='./download/clip/')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'visa'])
    parser.add_argument('--model', type=str, default="ViT-L/14@336px")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=518)
    parser.add_argument('--test_dataset', nargs='+', type=str, default=[], help='只评测这些数据集名（留空=评测全部）')
    parser.add_argument('--weight', type=str, required=True, default='./weight/best_maxf1_mvtec.pt', help='checkpoint 文件或目录')
    parser.add_argument('--load_strict', action='store_true', help='严格匹配 state_dict（不建议在跨构图时使用）')
    parser.add_argument('--device', type=str, default=None, help="如 'cuda:0'，默认自动选")
    parser.add_argument('--prompt_len', type=int, default=12, help='prompt length')
    # 评测相关
    parser.add_argument('--fewshot', type=int, default=0, help='>0 时启用 few-shot memorybank')
    parser.add_argument('--vis', type=int, default=0, help='可视化开关；0 仅跑指标')
    parser.add_argument('--vis_dir', type=str, default='./vis', help='可视化输出目录')
    # （可选）memorybank 融合时会用到 alpha
    parser.add_argument('--alpha', type=float, default=0.5, help='memorybank 融合权重')
    parser.add_argument('--memory_layers',  nargs='+', type=int, default=[6, 12, 18, 24], help='choose resnet layers to store and compare features')
    # 与模型插入逻辑保持一致（insert 中会读取）
    parser.add_argument('--adapter_layers', nargs='+', type=int, default=-1, help='Transformer block 层号（从0开始）；-1表示所有层：--adapter_layers -1 或 6 12 18')
    
    parser.add_argument('--result_dir', type=str, default='./results', help='评测结果目录')

    parser.add_argument('--log_file', type=str, default='', help='日志文件名（留空自动生成）')

    args = parser.parse_args()

    # 设备
    if args.device is None:
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(args.device)

    # 加载 CLIP backbone 和预处理
    clip_model, clip_transform = load(name=args.model,
                                      jit=(args.model not in available_models()),
                                      device=device,
                                      download_root=args.clip_download_dir)

    # 和 main.py 完全一致的图像/GT 预处理与分辨率覆盖
    clip_transform.transforms[0] = transforms.Resize(size=(args.img_size, args.img_size),
                                                     interpolation=transforms.InterpolationMode.BICUBIC)
    clip_transform.transforms[1] = transforms.CenterCrop(size=(args.img_size, args.img_size))
    target_transform = transforms.Compose([
        transforms.Resize(size=clip_transform.transforms[0].size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])

    # 冻结参数 & 设备
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)
    clip_model = clip_model.to(device)

    # 插入可学习 prompt / 适配器（这里会读取 args.adapter_layers）
    clip_model.insert(args=args, tokenizer=tokenize, device=device)

    # —— 仅做评测：不训练，直接加载权重 ——
    _ = load_checkpoint(clip_model, args.weight, device=device, strict=args.load_strict)


    # Logger（既写文件也打印到控制台）
    logger, log_path = setup_logger(args.result_dir, args.log_file)
    logger.info(f"[Args] {vars(args)}")
    # 构建测试集（与 main.py 完全一致）
    test_dataset_dict = build_all_test_datasets(args, clip_transform, target_transform)

    # 评测循环（与 main.py 等效）
    # dummy_logger = type("Dummy", (), {"info": print})()
    dummy_logger = logger
    results = []

    for dataset_name, test_ds in test_dataset_dict.items():
        logger.info(f"--------------------------- {dataset_name} ---------------------------")
        try:
            # 约定：eval_all_class 最好返回该数据集的指标 dict（如：f1_px/precision/recall/thr 等）
            metrics = eval_all_class(clip_model, dataset_name, test_ds, args, dummy_logger, device)

            if args.vis == 1:
                visualize(clip_model, test_ds, args, test_ds.transform, device)
            # 兼容兜底：若未返回 dict，也照样写入 CSV
            if metrics is None:
                metrics = {}
            if not isinstance(metrics, dict):
                metrics = {"value": metrics}
            metrics.setdefault("dataset", dataset_name)

            # 可选：打印关键指标一行
            if "f1_px" in metrics:
                logger.info(f"[{dataset_name}] f1_px={metrics['f1_px']:.4f}")
            

            results.append(metrics)
        except Exception as e:
            logger.exception(f"[{dataset_name}] evaluation failed: {e}")
            results.append({"dataset": dataset_name, "error": str(e)})
        logger.info("--------------------------------------------------------------------")

    logger.info("[Summary]")
    for r in results:
        ds = r.get("dataset", "<unknown>")
        if "error" in r:
            logger.info(f"{ds}: ERROR: {r['error']}")
            continue
        items = []
        for k, v in sorted(r.items()):
            if k == "dataset":
                continue
            if isinstance(v, float):
                items.append(f"{k}={v:.4f}")
            else:
                items.append(f"{k}={v}")
        logger.info(f"{ds}: " + ", ".join(items))

if __name__ == "__main__":
    main()
