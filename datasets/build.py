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



def bulid_transforms_for_T5Gemma2(img_size=(896, 896), is_train=True, aug=False):
    
    # img_size is kept for API compatibility; final resizing is done by processor.
    _ = img_size

    if not is_train:
        # 验证/推理阶段：不做增强
        return T.Compose([
            T.ToTensor(),
        ])

    if aug:
        return T.Compose([
            T.RandomHorizontalFlip(0.5),
            T.RandomApply([T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)], p=0.5),
            T.RandomApply([T.GaussianBlur(kernel_size=5)], p=0.1),
            T.ToTensor()
        ])
        

    # 弱增强：仅翻转
    return T.Compose([
        T.RandomHorizontalFlip(0.5),
        T.ToTensor()
    ])

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

        hf_use_fast = getattr(args, "hf_use_fast", True)
        processor = AutoProcessor.from_pretrained(
                getattr(args, "hf_model_name_or_path", "T5_270M_Base"),
                use_fast=bool(hf_use_fast),
            )

    
    if args.training:
        if is_t5:
            train_transforms = bulid_transforms_for_T5Gemma2(is_train=True,aug=args.img_aug)
            
            val_transforms = bulid_transforms_for_T5Gemma2(is_train=True,aug=args.img_aug)
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
