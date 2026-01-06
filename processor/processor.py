import logging
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
from contextlib import nullcontext


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("IRRA.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "sdm_loss": AverageMeter(),
        "itc_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "mlm_loss": AverageMeter(),
        "gen_loss": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "mlm_acc": AverageMeter()
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0

    def _as_loss_tensor(x, device: str):
        if torch.is_tensor(x):
            return x
        return torch.as_tensor(x, device=device, dtype=torch.float32)

    def _to_scalar(x):
        if torch.is_tensor(x):
            return x.detach()
        return x

    # train
    for epoch in range(start_epoch, num_epoch + 1):
        warmup_epochs = int(getattr(args, "warmup_epochs", 5))
        
        # Determine if we should freeze vision tower
        should_freeze = (epoch <= warmup_epochs)
        
        _model_inner = model.module if hasattr(model, "module") else model
        
        _vision_tower = None
        if hasattr(_model_inner, "encoder") and hasattr(_model_inner.encoder, "vision_tower"):
            if get_rank() == 0:
                print("✅ 成功找到 Vision Tower，准备冻结逻辑。")
            _vision_tower = _model_inner.encoder.vision_tower
        else:
            if get_rank() == 0:
                print("❌ 警告：未找到 model.encoder.vision_tower，请检查模型定义！")
            raise ValueError("Vision Tower not found for freezing logic")
        
        if _vision_tower is not None:
            n_frozen = 0
            n_total = 0
            for param in _vision_tower.parameters():
                n_total += 1
                param.requires_grad = not should_freeze
                if should_freeze:
                    n_frozen += 1
            
            if get_rank() == 0:
                status = "FROZEN" if should_freeze else "UNFROZEN"
                logger.info(f"Epoch {epoch}: Vision Tower is {status} ({n_frozen}/{n_total} params affected).")
        
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()

        use_amp = bool(getattr(args, "amp", False))
        amp_dtype_str = str(getattr(args, "amp_dtype", "bf16")).lower()
        if amp_dtype_str == "fp16":
            amp_dtype = torch.float16
        else:
            amp_dtype = torch.bfloat16

        grad_accum_steps = int(getattr(args, "grad_accum_steps", 1))
        if grad_accum_steps <= 0:
            raise ValueError(f"grad_accum_steps must be >= 1, got {grad_accum_steps}")

        max_grad_norm = float(getattr(args, "max_grad_norm", 0.0))
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if use_amp
            else nullcontext()
        )

        optimizer.zero_grad(set_to_none=True)

        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            if bool(getattr(args, "reuse_mm_encoder_for_image", False)):
                # Let the model reuse gen encoder outputs for image features (when available)
                batch["reuse_mm_encoder_for_image"] = True

            with autocast_ctx:
                ret = model(batch)

            if "total_loss" in ret:
                total_loss = _as_loss_tensor(ret["total_loss"], device)
            else:
                loss_terms = []
                for k, v in ret.items():
                    if k.endswith("_loss"):
                        loss_terms.append(_as_loss_tensor(v, device))
                total_loss = torch.stack(loss_terms).sum() if len(loss_terms) > 0 else torch.zeros((), device=device)

            # gradient accumulation
            loss_for_backward = total_loss / grad_accum_steps

            if "images" in batch:
                batch_size = batch["images"].shape[0]
            else:
                batch_size = batch["pixel_values"].shape[0]

            meters['loss'].update(_to_scalar(total_loss), batch_size)
            meters['sdm_loss'].update(_to_scalar(ret.get('sdm_loss', 0)), batch_size)
            meters['itc_loss'].update(_to_scalar(ret.get('itc_loss', 0)), batch_size)
            meters['id_loss'].update(_to_scalar(ret.get('id_loss', 0)), batch_size)
            meters['mlm_loss'].update(_to_scalar(ret.get('mlm_loss', 0)), batch_size)
            meters['gen_loss'].update(_to_scalar(ret.get('gen_loss', 0)), batch_size)

            meters['img_acc'].update(_to_scalar(ret.get('img_acc', 0)), batch_size)
            meters['txt_acc'].update(_to_scalar(ret.get('txt_acc', 0)), batch_size)
            meters['mlm_acc'].update(_to_scalar(ret.get('mlm_acc', 0)), 1)

            if scaler.is_enabled():
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            do_step = ((n_iter + 1) % grad_accum_steps == 0)
            if do_step:
                if max_grad_norm > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    val = v.avg
                    if torch.is_tensor(val):
                        val = val.item()
                    if val > 0:
                        info_str += f", {k}: {val:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
        
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        for k, v in meters.items():
            val = v.avg
            if torch.is_tensor(val):
                val = val.item()
            if val > 0:
                tb_writer.add_scalar(k, val, epoch)


        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    top1 = evaluator.eval(model.module.eval())
                else:
                    top1 = evaluator.eval(model.eval())

                torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)
    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")


def do_inference(model, test_img_loader, test_txt_loader):

    logger = logging.getLogger("IRRA.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1 = evaluator.eval(model.eval())
