from clip.clip import load, tokenize, available_models
import torch
from dataset import *
from torchvision import transforms
import argparse
import numpy as np
import random
import torch.nn.functional as F
from tqdm import tqdm
import logging
import os
from util.utils import calculate_metrics
import copy
from loss import dice_loss, focal_loss

# === 在 main.py 顶部：新增一个导入 ===
@torch.no_grad()
def evaluate_maxf1_px(clip_model, dataset, args, device):
    """在整个 val set 上汇总像素级分数与标签，返回阈值自由的 max-F1。"""
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    clip_model.eval()

    all_scores = []
    all_labels = []

    for items in loader:
        imgs, _, gts = items[:3]
        imgs = imgs.to(device)
        gts  = gts.to(device)

        # 取得概率分数图（不二值化）
        _, pred_masks, _, _ = clip_model.detect_forward_seg(imgs, args=args)

        # 将 GT 对齐到预测尺寸，并二值化为 {0,1}
        gts = F.interpolate(gts, size=pred_masks.shape[-2:], mode='bilinear')
        gts = (gts > 0.5).float()

        # 展平成一维，累计
        all_scores.append(pred_masks.detach().cpu().reshape(-1))
        all_labels.append(gts.detach().cpu().reshape(-1))

    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy()

    # 直接用 utils.py 的 calculate_metrics 取 max-F1（内部会扫阈值）
    metrics = calculate_metrics(scores, labels)
    return float(metrics['max-F1']) / 100.0  # 调整为0-1范围的值
def setup_seed(seed):
    if seed == -1:
        seed = random.randint(0, 1000)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed
def _ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_checkpoint(clip_model, optimizer, epoch, args, path):
    # 统一保存完整权重与训练状态
    checkpoint = {
        "model_state_dict": clip_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "args": vars(args)
    }
    _ensure_dir(os.path.dirname(path))
    torch.save(checkpoint, path)

def load_checkpoint(clip_model, path, device="cpu", strict=False):
    import os
    import torch
    import torch.nn.functional as F

    # 支持目录：默认读 last.pt
    if os.path.isdir(path):
        path = os.path.join(path, "last.pt")

    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)  # 兼容纯 state_dict 的 ckpt

    # 处理可能存在的 'module.' 前缀（DDP 保存）
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    key = "visual.positional_embedding"
    if key in state and hasattr(clip_model, "visual") and hasattr(clip_model.visual, "positional_embedding"):
        pe_ckpt = state[key]                              # [1+Hc*Wc, C]
        pe_model = clip_model.visual.positional_embedding # [1+Hm*Wm, C]

        if pe_ckpt.shape != pe_model.shape:
            # ckpt -> 模型尺寸的插值
            with torch.no_grad():
                cls_token  = pe_ckpt[:1, :]               # [1, C]
                grid_ckpt  = pe_ckpt[1:, :]               # [Hc*Wc, C]
                C          = grid_ckpt.shape[-1]
                old_side   = int((grid_ckpt.shape[0]) ** 0.5)
                new_side   = int((pe_model.shape[0] - 1) ** 0.5)

                grid_ckpt  = grid_ckpt.transpose(0, 1).reshape(1, C, old_side, old_side)
                grid_new   = F.interpolate(grid_ckpt, size=(new_side, new_side),
                                           mode="bilinear", align_corners=False)
                grid_new   = grid_new.reshape(1, C, new_side * new_side).transpose(1, 2).squeeze(0)

                pe_new     = torch.cat([cls_token, grid_new], dim=0)  # [1+Hm*Wm, C]
                # dtype/设备对齐
                pe_new     = pe_new.to(dtype=pe_model.dtype, device=pe_model.device)
                state[key] = pe_new
                print(f"[load_checkpoint] resized positional_embedding {tuple(pe_ckpt.shape)} -> {tuple(pe_new.shape)}")

    # 最终加载
    clip_model.load_state_dict(state, strict=strict)
    return checkpoint


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def print_args(logger, args):
    logger.info('--------args----------')
    for k in list(vars(args).keys()):
        logger.info('{}: {}'.format(k, vars(args)[k]))
    logger.info('--------args----------\n')   

def train(args):
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    logger = get_logger(os.path.join(args.log_dir, '{}_{}_s{}.txt'.format(args.dataset, args.fewshot, args.seed)))
    print_args(logger, args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_transform = load(name=args.model, jit = (not args.model in available_models()), device=device, download_root=args.clip_download_dir)

    clip_transform.transforms[0] = transforms.Resize(size=(args.img_size, args.img_size), interpolation=transforms.InterpolationMode.BICUBIC)
    clip_transform.transforms[1] = transforms.CenterCrop(size=(args.img_size, args.img_size))
    target_transform = transforms.Compose([
        transforms.Resize(size=clip_transform.transforms[0].size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])
    clip_model.eval()
    
    for param in clip_model.parameters():
        param.requires_grad_(False)
    
    clip_model = clip_model.to(device)
    clip_model.insert(args=args, tokenizer=tokenize, device=device)

    test_dataset_mvtec = MVTecDataset(root=args.data_dir, train=False, category=None, transform=clip_transform, gt_target_transform=target_transform)
    test_dataset_isic = ISICDataset(root=args.data_dir, train=False, category=None,transform=clip_transform, gt_target_transform=target_transform)
    test_dataset_clinic = ClinicDBDataset(root=args.data_dir, train=False, category=None,transform=clip_transform, gt_target_transform=target_transform)
    test_dataset_colon = ColonDBDataset(root=args.data_dir, train=False, category=None,transform=clip_transform, gt_target_transform=target_transform)
    test_dataset_visa = VisaDataset(root=args.data_dir, train=False, category=None,transform=clip_transform, gt_target_transform=target_transform)
    test_dataset_btad = BTADDataset(root=args.data_dir, train=False, category=None,transform=clip_transform, gt_target_transform=target_transform)
    test_dataset_dtd = DTDDataset(root=args.data_dir, train=False, category=None,transform=clip_transform, gt_target_transform=target_transform)
    test_dataset_brainmri = BrainMRIDataset(root=args.data_dir, train=False, category=None,transform=clip_transform, gt_target_transform=target_transform)
    test_dataset_br35h = Br35HDataset(root=args.data_dir, train=False, category=None,transform=clip_transform, gt_target_transform=target_transform)
    test_dataset_dagm = DAGMDataset(root=args.data_dir, train=False, category=None,transform=clip_transform, gt_target_transform=target_transform)
    test_dataset_kvasir = KvasirDataset(root=args.data_dir, train=False, category=None,transform=clip_transform, gt_target_transform=target_transform)
    
    all_test_dataset_dict = {
        "mvtec": test_dataset_mvtec,
        "visa": test_dataset_visa,
        "btad": test_dataset_btad,
        "dtd": test_dataset_dtd,
        'dagm': test_dataset_dagm,
        "isic": test_dataset_isic,
        "clinic": test_dataset_clinic,
        "colon": test_dataset_colon,
        "brainmri": test_dataset_brainmri,
        "br35h": test_dataset_br35h,
        'kvasir': test_dataset_kvasir,
    }
    if len(args.test_dataset) < 1:
        test_dataset_dict = all_test_dataset_dict
    else:
        test_dataset_dict = {}
        for ds_name in args.test_dataset:
            test_dataset_dict[ds_name] = all_test_dataset_dict[ds_name]
    if args.dataset in test_dataset_dict:
        del test_dataset_dict[args.dataset]
    if args.dataset == 'mvtec':
        train_dataset = test_dataset_mvtec
    else:
        train_dataset = test_dataset_visa
        
    # 新增：确定 best 判定集
    if args.best_ds and args.best_ds in all_test_dataset_dict:
        best_select_ds_name = args.best_ds
    else:
        # 回退策略：从 test_dataset_dict 任选一个（优先与训练集不同）
        best_select_ds_name = next(iter(test_dataset_dict.keys()))

    best_select_ds = all_test_dataset_dict[best_select_ds_name]
    logger.info(f"[Best-Selector] using single test set: {best_select_ds_name} for best.pt")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    _ensure_dir(args.weight)
    optimizer = torch.optim.AdamW(clip_model.get_trainable_parameters(), lr=2e-5, weight_decay=1e-2)
    start_epoch = 1

    # === 统一带数据集名的权重命名：last_{dataset}.pt / best_{dataset}.pt / final_{dataset}.pt ===
    def _ckpt_name(stem: str) -> str:
        return f"{stem}_{args.dataset}.pt"

    last_path   = os.path.join(args.weight, _ckpt_name("last"))
    legacy_last = os.path.join(args.weight, "last.pt")  # 兼容旧文件名

    if args.weight is not None and (os.path.exists(last_path) or os.path.exists(legacy_last)):
        ckpt_file = last_path if os.path.exists(last_path) else legacy_last
        ckpt = load_checkpoint(clip_model, ckpt_file, device=device, strict=args.load_strict)
        logger.info(f"Loaded checkpoint from: {ckpt_file}")
        ckpt_epoch = int(ckpt.get("epoch", 0))
        start_epoch = ckpt_epoch + 1
        logger.info(f"Resume training from epoch {start_epoch}")
        if args.resume and ckpt.get("optimizer_state_dict") is not None:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                logger.warning(f"Failed to load optimizer state, continue without resume: {e}")
    else:
        # 如果没有权重文件，加载预训练模型并开始训练
        logger.info("No checkpoint found, starting training from scratch with pretrained weights")
        clip_model.eval()  # Load pretrained model for training initialization
        optimizer = torch.optim.AdamW(clip_model.get_trainable_parameters(), lr=args.lr, weight_decay=1e-2)


        start_epoch = 1  # 从第1轮开始训练    

    best_maxf1 = -1.0 
    for epoch in range(start_epoch, args.epochs + 1):
        total_loss = []

        for items in tqdm(train_dataloader):
            imgs, labels, gts = items[:3]
            labels = labels.to(device)
            imgs = imgs.to(device)
            gts = gts.to(device)
            cls_labels, predict_masks, img_tokens, logits_img = clip_model.detect_forward_seg(imgs, args=args)

            # 调整真实标签与预测掩码尺寸一致
            gts = F.interpolate(gts, size=predict_masks[0].shape[-2:], mode='bilinear')
            gts[gts < 0.5] = 0
            gts[gts > 0.5] = 1

                
            # 计算分类损失
            
            ce_loss = F.binary_cross_entropy(cls_labels, labels.float())    #bce

            # 计算分割损失
            eps = 1e-6
            loss_focal = focal_loss(predict_masks, gts)  # 使用 sigmoid focal loss 计算预测掩码的损失
            loss_dice = dice_loss(predict_masks, gts, num_masks=gts.size(0))  # 计算 Dice 损失

            # 总损失
            loss = ce_loss + args.lambda1 * loss_focal + args.lambda2 * loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())  

        epoch_loss = float(np.mean(total_loss))
        logger.info("Epoch: {}/{}, Loss: {:.6f}".format(epoch, args.epochs, epoch_loss))
        # 打印最后一个 batch 的 loss

        if ce_loss is not None and torch.isfinite(ce_loss) and torch.isfinite(loss_focal) and torch.isfinite(loss_dice):
            print(f"ce={ce_loss.item():.4f}, focal={loss_focal.item():.4f}, dice={loss_dice.item():.4f}")
        # logger.info("Epoch: {}/{}, Loss: {:.6f}".format(epoch, args.epochs, np.mean(total_loss)))
        save_checkpoint(clip_model, optimizer, epoch, args, os.path.join(args.weight, _ckpt_name("last")))

        maxf1 = evaluate_maxf1_px(clip_model, best_select_ds, args, device)
        logger.info(f"[Eval@{best_select_ds_name}] max-F1={maxf1:.4f}")
        if maxf1 > best_maxf1:
            best_maxf1 = maxf1
            save_checkpoint(clip_model, optimizer, epoch, args, os.path.join(args.weight, _ckpt_name("best_maxf1")))
            logger.info(f" New best(max-F1): {best_maxf1:.4f} -> saved best_maxf1.pt")
  
    
      
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Pytorch implemention of AF-CLIP')
    
    parser.add_argument('--clip_download_dir', type=str, default='./download/clip/', help='training dataset')
    
    parser.add_argument('--data_dir', type=str, default='./data', help='training dataset')
    
    parser.add_argument('--dataset', type=str, default='mvtec', help='training dataset', choices=['mvtec', 'visa'])
    
    parser.add_argument('--model', type=str, default="ViT-L/14@336px", help='model')
    
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    
    parser.add_argument('--lr', type=float, default=1e-4, help='learning tate')
    
    parser.add_argument('--alpha', type=float, default=0.1, help='label combination')

    parser.add_argument('--epochs', type=int, default=60, help='training epoch')
    
    parser.add_argument('--prompt_len', type=int, default=12, help='prompt length')
    
    parser.add_argument('--category', type=str, default=None, help='normal class')
    
    parser.add_argument('--fewshot', type=int, default=0, help='few shot num')
    
    parser.add_argument('--seed', type=int, default=122, help='seed')
    
    parser.add_argument('--log_dir', type=str, default='./log/', help='log dir')
    
    parser.add_argument('--suffix', type=str, default='defect', help='prompt suffix')
    
    parser.add_argument('--img_size', type=int, default=518)
    
    parser.add_argument('--feature_layers', nargs='+', type=int, default=[6, 12, 18, 24], help='choose vit layers to extract features')
    
    parser.add_argument('--test_dataset', nargs='+', type=str, default=[], help='choose vit layers to extract features')
    
    parser.add_argument('--weight', type=str, default='./weight', help='load weight path')
    
    parser.add_argument('--vis', type=int, default=1, help='visualization results')
    
    parser.add_argument('--vis_dir', type=str, default='./vis_results/', help='visualization results dir')
    
    parser.add_argument('--memory_layers',  nargs='+', type=int, default=[6, 12, 18, 24], help='choose resnet layers to store and compare features')
    
    parser.add_argument('--lambda1', type=float, default=100, help='lambda1 for loss')
    
    parser.add_argument('--lambda2', type=float, default=1, help='lambda2 for loss')

    parser.add_argument('--f1_thresh', type=float, default=0.5,help='threshold on probability map for pixel-level F1')
    
    parser.add_argument('--best_ds', type=str, default='',help='the single test dataset name used to decide best.pt (e.g., mvtec/visa/...)')

    parser.add_argument('--load_strict', action='store_true', default=False, help='严格加载 checkpoint 参数')
    
    parser.add_argument('--resume', action='store_true', default=True, help='是否从 checkpoint 恢复训练')
    args = parser.parse_args()
    
    args.seed = setup_seed(args.seed)
    train(args)
    
    
    
