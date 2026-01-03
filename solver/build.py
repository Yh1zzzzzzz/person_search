import torch

from .lr_scheduler import LRSchedulerWithWarmup


def build_optimizer(args, model):
    params = []

    print(f'Using {args.lr_factor} times learning rate for random init module ')
    
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = args.lr
        weight_decay = args.weight_decay

        if "cross" in key:
            # use large learning rate for random initialized cross modal module
            lr =  args.lr * args.lr_factor # default 5.0
        if "bias" in key:
            lr = args.lr * args.bias_lr_factor
            weight_decay = args.weight_decay_bias
        if "classifier" in key:
            lr = float(getattr(args, "classifier_lr", args.lr * args.lr_factor))
        if "mlm_head" in key:
            lr = args.lr * args.lr_factor

        # Projection heads: use the same (larger) learning rate for symmetric alignment.
        # - t5gemma2_vion_tower: vision_tower_align.*
        # - t5gemma2: vision_proj.* and text_proj.*
        # - (also apply to text_proj for symmetry)
        if ("vision_tower_align" in key) or ("vision_proj" in key) or ("text_proj" in key):
            lr = float(getattr(args, "projector_lr", 1e-3))

        # BNNeck affine params: keep in the same group as classifier for stability
        if ("bn_i" in key) or ("bn_t" in key):
            lr = float(getattr(args, "classifier_lr", args.lr * args.lr_factor))
        
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            params, lr=args.lr, momentum=args.momentum
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            params,
            lr=args.lr,
            betas=(args.alpha, args.beta),
            eps=1e-3,
        )
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            params,
            lr=args.lr,
            betas=(args.alpha, args.beta),
            eps=1e-8,
        )
    else:
        NotImplementedError

    return optimizer


def build_lr_scheduler(args, optimizer):
    return LRSchedulerWithWarmup(
        optimizer,
        milestones=args.milestones,
        gamma=args.gamma,
        warmup_factor=args.warmup_factor,
        warmup_epochs=args.warmup_epochs,
        warmup_method=args.warmup_method,
        total_epochs=args.num_epoch,
        mode=args.lrscheduler,
        target_lr=args.target_lr,
        power=args.power,
    )
