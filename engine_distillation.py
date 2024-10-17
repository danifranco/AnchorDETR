# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-student_model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher
import datasets.transforms as T

def train_one_epoch_distillation(teacher_model: torch.nn.Module, student_model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    
    teacher_model.eval()
    student_model.train()

    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()
    unnorm = T.UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):

        # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
        with torch.no_grad():
            # Unnormalize the data as cellSAM expects to be in [0,1] range 
            imgs, masks = samples.decompose()
            for img in imgs:
                unnorm(img)

            processed_imgs, _ = teacher_model.sam_bbox_preprocessing(imgs, device=device)
            teacher_outputs = teacher_model.cellfinder.decode_head(processed_imgs)

            # Generate the bboxes from cellFinder
            # boxes_per_heatmap = teacher_model.generate_bounding_boxes(imgs, device=device)
            # boxes_per_heatmap = [bbox/2 for bbox in boxes_per_heatmap] # Seems that the bboxes are rescaled expecting a 1024x1204 input
            # # comprobar la salida de esto pintando la imagen antes de empezar a machear. quizas requiere normalizacion
            
            # for j, img in enumerate(imgs):
            #     utils.save_image(
            #         img, 
            #         boxes_per_heatmap[j], 
            #         "/scratch/dfranco/thesis/data2/dfranco/exp_results/anchorDETR_cellSAM_13/teacher_results", 
            #         f"image_{j}.png"
            #     )
        outputs = student_model(samples)

        loss = criterion(outputs, teacher_outputs)
        loss_value = loss.item()
        loss_value_reduce = utils.all_reduce_mean(loss_value)
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_value_reduce)
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(student_model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
        samples, targets = prefetcher.next()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
@torch.no_grad()
def evaluate_distillation(teacher_model, student_model, criterion, postprocessors, data_loader, device, output_dir, save_json=False):
    student_model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:' 

    unnorm = T.UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    for samples, targets in metric_logger.log_every(data_loader, 100, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
        with torch.no_grad():
            # Unnormalize the data as cellSAM expects to be in [0,1] range 
            imgs, masks = samples.decompose()
            for img in imgs:
                unnorm(img)

            processed_imgs, _ = teacher_model.sam_bbox_preprocessing(imgs, device=device)
            teacher_outputs = teacher_model.cellfinder.decode_head(processed_imgs)


        outputs = student_model(samples)
        loss = criterion(outputs, teacher_outputs)
        loss_value = loss.item()
        loss_value_reduce = utils.all_reduce_mean(loss_value)
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_value_reduce)
            sys.exit(1)
        metric_logger.update(loss=loss_value)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
        
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
