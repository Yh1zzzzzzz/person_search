import logging
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from datasets.sampler import RandomIdentitySampler
from datasets.sampler_ddp import RandomIdentitySampler_DDP
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms.functional as F
from PIL import Image
import random
from utils.comm import get_world_size

from .bases import ImageDataset, TextDataset, ImageTextDataset, ImageTextMLMDataset
from .hf_t5gemma2 import ImageDatasetT5Gemma2, TextDatasetT5Gemma2, ImageTextDatasetT5Gemma2

from .cuhkpedes import CUHKPEDES
from .icfgpedes import ICFGPEDES
from .rstpreid import RSTPReid

__factory = {'CUHK-PEDES': CUHKPEDES, 'ICFG-PEDES': ICFGPEDES, 'RSTPReid': RSTPReid}


class SiglipLetterboxAugment(object):
    """
    针对 SigLIP 优化的预处理管道：
    1. 保持长宽比缩放 (Bicubic)
    2. [训练时] 在纯净的图片上做几何/光度增强 (Flip, Color, Blur)
    3. 居中填充灰色背景 (Padding)
    """
    def __init__(self, target_size, fill_value=128, is_train=True, aug=False):
        # 能够处理 int 或 tuple
        self.target_size = (target_size, target_size) if isinstance(target_size, int) else target_size
        self.fill_value = fill_value
        self.is_train = is_train
        self.aug = aug
        
        # 定义增强参数
        # 1. 颜色抖动：SigLIP 对色彩敏感，Person Search 依赖颜色描述，所以 Hue 设为 0
        self.color_jitter = T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)
        
    def __call__(self, img):
        # -----------------------------------------------------------
        # Step 1: Resize (保持长宽比，使用 Bicubic)
        # -----------------------------------------------------------
        w, h = img.size
        target_h, target_w = self.target_size
        
        scale = min(target_w / w, target_h / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        
        # SigLIP 官方通常偏好 BICUBIC 插值
        img = img.resize((new_w, new_h), Image.BICUBIC)
        
        # -----------------------------------------------------------
        # Step 2: Augmentation (在 Resize 后，Padding 前)
        # -----------------------------------------------------------
        if self.is_train and self.aug:
            # A. 随机水平翻转 (Random Horizontal Flip)
            if random.random() < 0.5:
                img = F.hflip(img)
            
            # B. 颜色抖动 (Color Jitter) - 只在有效行人区域做
            if random.random() < 0.5:
                img = self.color_jitter(img)
            
            # C. 高斯模糊 (Gaussian Blur)
            # 448分辨率很高，模糊可以模拟低清摄像头，增强鲁棒性
            if random.random() < 0.1:
                img = F.gaussian_blur(img, kernel_size=5)

        # -----------------------------------------------------------
        # Step 3: Padding (居中补灰)
        # -----------------------------------------------------------
        # 创建画布，填充 fill_value (通常是 128)
        canvas = Image.new('RGB', (target_w, target_h), (self.fill_value, self.fill_value, self.fill_value))
        
        # 计算居中位置
        pad_left = (target_w - new_w) // 2
        pad_top = (target_h - new_h) // 2
        
        # 粘贴
        canvas.paste(img, (pad_left, pad_top))
        
        return canvas

def bulid_transforms_for_T5Gemma2(img_size=(448, 448), is_train=True, aug=False):
    """
    为使用 SigLIP 作为 Vision Tower 的 T5Gemma2 构建 Transform。
    特点：
    1. 输入尺寸 448x448
    2. 均值方差均为 0.5 (SigLIP 标准)
    3. 采用 Resize -> Augment -> Pad 流程
    """
    
    # --- 关键修改：SigLIP 的标准归一化参数 ---
    # 大多数 SigLIP 模型 (如 google/siglip-so400m) 训练时 rescale_factor=1/255 
    # 并且使用 mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    # 这相当于将像素值缩放到 [-1, 1] 区间
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    
    # 填充颜色：归一化前的 0.5 对应像素值 128
    fill_value = 128

    if not is_train:
        # 验证/推理阶段：无增强，仅 Resize + Pad
        transform = T.Compose([
            SiglipLetterboxAugment(target_size=img_size, fill_value=fill_value, is_train=False, aug=False),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return transform

    # 训练阶段
    if aug:
        # 强增强模式
        transform = T.Compose([
            # 1. 自定义管道：Resize -> Augment(Flip/Color/Blur) -> Pad
            SiglipLetterboxAugment(target_size=img_size, fill_value=fill_value, is_train=True, aug=True),
            
            # 2. 转 Tensor
            T.ToTensor(),
            
            # 3. 归一化 (到 -1 ~ 1)
            T.Normalize(mean=mean, std=std),
            
            # 4. 随机擦除 (Random Erasing) - 放在最后!
            # 对于 448x448，scale 设为 (0.02, 0.15) 比较合适 (也就是 126像素到 340像素的遮挡块)
            # value='random' 或 value=0 (因为归一化后灰色背景是0) 都可以，推荐 'random' 增加鲁棒性
            T.RandomErasing(p=0.5, scale=(0.02, 0.15), value='random'), 
        ])
    else:
        # 弱增强模式 (通常只开翻转)
        transform = T.Compose([
            # 这里的 aug=True 开启了 SiglipLetterboxAugment 里的基础增强逻辑，
            # 你可以在 SiglipLetterboxAugment 内部再细分，或者像下面这样简化：
            # 这里为了简单，假设 aug=False 时只做 Resize+Pad (类似 Baseline)，或者你可以手动加 Flip
            SiglipLetterboxAugment(target_size=img_size, fill_value=fill_value, is_train=True, aug=False),
            T.RandomHorizontalFlip(0.5), # 如果 SiglipLetterboxAugment 没开增强，在这里补一个翻转
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        
    return transform

def build_transforms(img_size=(384, 128), aug=False, is_train=True):
    """
    构建图片数据增强，传递到dataset中，供dataset读取图片时使用
    """
    height, width = img_size

    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    if not is_train:
        transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return transform

    # transform for training
    if aug:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(scale=(0.02, 0.4), value=mean),
        ])
    else:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    return transform









def collate(batch):
    """
    处理的数据类型
    从 ImageTextDataset 的 __getitem__ 方法返回的样本格式：
    {
        'img': PIL.Image.Image,
        'img_id': int,
        'caption': tensor,
        'pids': int,
    }
    处理后的数据类型,
    'pids': torch.tensor([1, 2, 3, ...]),                      # [batch_size]
    'image_ids': torch.tensor([1001, 1002, 1003, ...]),        # [batch_size]
    'images': torch.Tensor([batch_size, 3, 384, 128]),         # [batch_size, 3, 384, 128]
    'caption_ids': torch.Tensor([batch_size, 77]),             # [batch_size, 77]
    """
    keys = set([key for b in batch for key in b.keys()])
    # turn list of dicts data structure to dict of lists data structure
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

    batch_tensor_dict = {}
    for k, v in dict_batch.items():
        if isinstance(v[0], int):
            batch_tensor_dict.update({k: torch.tensor(v)})
        elif torch.is_tensor(v[0]):
             batch_tensor_dict.update({k: torch.stack(v)})
        else:
            raise TypeError(f"Unexpect data type: {type(v[0])} in a batch.")

    return batch_tensor_dict

def build_dataloader(args, tranforms=None):
    """
    根据命令行参数，构建数据加载器(DataLoader)
    :param args: 命令行参数
    :param tranforms: 数据增强
    :return: train_loader, val_img_loader, val_txt_loader, num_classes
    """
    logger = logging.getLogger("IRRA.dataset")

    num_workers = args.num_workers
    dataset = __factory[args.dataset_name](root=args.root_dir)
    num_classes = len(dataset.train_id_container)

    backbone = getattr(args, "backbone", "clip")
    is_t5 = backbone in ("t5gemma2", "t5gemma2_vion_tower")
    processor = None
    if is_t5:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(getattr(args, "hf_model_name_or_path", "T5_270M_Base"))
    if args.training:
        if is_t5:
            t5_img_size = int(getattr(args, "t5_image_size", 448))
            train_transforms = bulid_transforms_for_T5Gemma2(img_size=(t5_img_size, t5_img_size),
                                                          is_train=True,
                                                          aug=args.img_aug)
            
            val_transforms = bulid_transforms_for_T5Gemma2(img_size=(t5_img_size, t5_img_size),
                                                        is_train=False,
                                                        aug=False)
            loss_names = str(getattr(args, "loss_names", ""))
            train_set = ImageTextDatasetT5Gemma2(
                dataset.train,
                processor=processor,
                text_length=args.text_length,
                mm_max_length=getattr(args, "mm_max_length", 512),
                enable_mlm=("mlm" in args.loss_names),
                enable_gen=("gen" in loss_names),
                gen_prompt=getattr(args, "gen_prompt", "Caption"),
                gen_prompt_length=int(getattr(args, "gen_prompt_length", 32)),
                train_transforms=train_transforms,
            )
        else:
            train_transforms = build_transforms(img_size=args.img_size,
                                                aug=args.img_aug,
                                                is_train=True)
            val_transforms = build_transforms(img_size=args.img_size,
                                              is_train=False)

            if args.MLM:
                train_set = ImageTextMLMDataset(dataset.train,
                                         train_transforms,
                                         text_length=args.text_length)
            else:
                train_set = ImageTextDataset(dataset.train,
                                         train_transforms,
                                         text_length=args.text_length)

        if args.sampler == 'identity':
            if args.distributed:
                logger.info('using ddp random identity sampler')
                logger.info('DISTRIBUTED TRAIN START')
                mini_batch_size = args.batch_size // get_world_size()
                # TODO wait to fix bugs
                data_sampler = RandomIdentitySampler_DDP(
                    dataset.train, args.batch_size, args.num_instance)
                batch_sampler = torch.utils.data.sampler.BatchSampler(
                    data_sampler, mini_batch_size, True)

            else:
                logger.info(
                    f'using random identity sampler: batch_size: {args.batch_size}, id: {args.batch_size // args.num_instance}, instance: {args.num_instance}'
                )
                train_loader = DataLoader(train_set,
                                          batch_size=args.batch_size,
                                          sampler=RandomIdentitySampler(
                                              dataset.train, args.batch_size,
                                              args.num_instance),
                                          num_workers=num_workers,
                                          collate_fn=collate,
                                          pin_memory=True)
        elif args.sampler == 'random':
            # TODO add distributed condition
            logger.info('using random sampler')
            train_loader = DataLoader(train_set,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      collate_fn=collate,
                                      pin_memory=True)
        else:
            logger.error('unsupported sampler! expected softmax or triplet but got {}'.format(args.sampler))

        # use test set as validate set
        ds = dataset.val if args.val_dataset == 'val' else dataset.test
        if is_t5:
            val_img_set = ImageDatasetT5Gemma2(ds['image_pids'], ds['img_paths'], processor=processor, transforms=val_transforms)
            val_txt_set = TextDatasetT5Gemma2(
                ds['caption_pids'],
                ds['captions'],
                tokenizer=processor.tokenizer,
                text_length=args.text_length,
            )
        else:
            val_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],
                                       val_transforms)
            val_txt_set = TextDataset(ds['caption_pids'],
                                      ds['captions'],
                                      text_length=args.text_length)

        val_img_loader = DataLoader(val_img_set,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    pin_memory=True)
        val_txt_loader = DataLoader(val_txt_set,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    pin_memory=True)

        return train_loader, val_img_loader, val_txt_loader, num_classes

    else:
        # build dataloader for testing
        if is_t5:
            t5_img_size = int(getattr(args, "t5_image_size", 448))
            test_transforms = bulid_transforms_for_T5Gemma2(img_size=(t5_img_size, t5_img_size), is_train=False, aug=False)
        elif tranforms:
            test_transforms = tranforms
        else:
            test_transforms = build_transforms(img_size=args.img_size,
                                               is_train=False)

        ds = dataset.test
        if is_t5:
            test_img_set = ImageDatasetT5Gemma2(ds['image_pids'], ds['img_paths'], processor=processor, transforms=test_transforms)
            test_txt_set = TextDatasetT5Gemma2(
                ds['caption_pids'],
                ds['captions'],
                tokenizer=processor.tokenizer,
                text_length=args.text_length,
            )
        else:
            test_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],
                                        test_transforms)
            test_txt_set = TextDataset(ds['caption_pids'],
                                       ds['captions'],
                                       text_length=args.text_length)

        test_img_loader = DataLoader(test_img_set,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers,
                                     pin_memory=True)
        test_txt_loader = DataLoader(test_txt_set,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers,
                                     pin_memory=True)
        return test_img_loader, test_txt_loader, num_classes
