from detectron2.data.transforms import augmentation
import os
import torch
import logging
import weakref

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch, create_ddp_model, AMPTrainer
from detectron2.solver import get_default_optimizer_params, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

from joint_panoptic_forecast.config import add_forecast_config
from joint_panoptic_forecast.data.cityscapes import InstanceFullSeqDatasetMapper, InstanceSavedFeatMapper, InstanceSeqDatasetMapper,  SaveFeatsEvaluator
from joint_panoptic_forecast.data.cityscapes.evaluators import CityscapesForecastInstsegEvaluator, CityscapesForecastPansegEvaluator, CityscapesForecastSemsegEvaluator
from joint_panoptic_forecast.hooks import LossEvalHook

from joint_panoptic_forecast.my_simple_trainer import MySimpleTrainer
from joint_panoptic_forecast.utils import build_detection_val_loader



class Trainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)
        self.grad_accumulate_steps = cfg.SOLVER.GRAD_ACCUMULATE_STEPS
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else MySimpleTrainer)(
            model, data_loader, optimizer, grad_accumulate_steps=self.grad_accumulate_steps,
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference_%s"%(dataset_name))
        save_seq_feats = cfg.MODEL.SAVE_SEQ_FEATS
        eval_ids = cfg.EVAL_IDS
        eval_pq = cfg.EVAL_PQ
        eval_instseg = cfg.EVAL_INSTANCE_SEGMENTATION
        eval_panseg = cfg.EVAL_PANOPTIC_SEGMENTATION
        eval_semseg = cfg.EVAL_SEMANTIC_SEGMENTATION
        use_depth_sorting = cfg.EVAL_WITH_DEPTH_SORTING
        viz = cfg.EVAL_VIZ
        bg_dir = cfg.PANOPTIC_SEGMENTATION_BG_DIR
        if 'val' in dataset_name:
            split = 'val'
        elif 'test' in dataset_name:
            split = 'test'
        else:
            split = 'train'


        if save_seq_feats:
            return SaveFeatsEvaluator(dataset_name, cfg.OUTPUT_DIR)
        elif eval_instseg:
            return CityscapesForecastInstsegEvaluator(dataset_name, output_folder, use_depth_sorting=use_depth_sorting,
                                                      viz=viz)
        elif eval_semseg:
            bg_dir = cfg.PANOPTIC_SEGMENTATION_BG_DIR
            return CityscapesForecastSemsegEvaluator(dataset_name, output_folder, use_depth_sorting=use_depth_sorting,
                                                     viz=viz, bg_dir=bg_dir, split=split)
        elif eval_panseg:
            if 'val' in dataset_name:
                split = 'val'
            else:
                split = 'train'
            return CityscapesForecastPansegEvaluator(dataset_name, output_folder, use_depth_sorting=use_depth_sorting,
                                                     bg_dir=bg_dir, viz=viz, split=split)
        else:
            raise ValueError('Evaluation not selected!')

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = Trainer._get_dataset_mapper(cfg, train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def _get_dataset_mapper(cls, cfg, train=False):
        use_saved_feat_mode = cfg.MODEL.USE_SAVED_FEAT_MODE
        seq_len = cfg.INPUT.SEQ_LEN
        if train:
            dataset = cfg.DATASETS.TRAIN[0]
        else:
            dataset = cfg.DATASETS.TEST[0]
        save_seq_feats = cfg.MODEL.SAVE_SEQ_FEATS
        if train:
            target_frame_range = cfg.INPUT.TARGET_FRAME_RANGE
            use_reverse_aug = cfg.INPUT.USE_REVERSE_AUG
        else:
            target_frame_range = None
            use_reverse_aug = False
        
        if use_saved_feat_mode:
            if 'train' in dataset:
                name = 'train'
            elif 'test' in dataset:
                name = 'test'
            else:
                name = 'val'
            bg_file = cfg.INPUT.SAVED_BG_LOGIT_PATH
            if bg_file is not None:
                bg_file = bg_file % name
            bg_depth_file = cfg.INPUT.SAVED_BG_DEPTH_PATH
            if bg_depth_file is not None:
                bg_depth_file = bg_depth_file % name
            bg_depth_dir = cfg.INPUT.SAVED_BG_DEPTH_DIR
            if bg_depth_dir is not None:
                bg_depth_dir = os.path.join(bg_depth_dir, name)
            bg_feat_file = cfg.INPUT.SAVED_BG_FEAT_PATH
            if bg_feat_file is not None:
                bg_feat_file = bg_feat_file % name
            pred_odom_file = cfg.INPUT.SAVED_PRED_ODOM_PATH
            if pred_odom_file is not None:
                pred_odom_file = pred_odom_file % name
            mapper = InstanceSavedFeatMapper(cfg, augmentations=[], seq_len=seq_len, target_frame_range=target_frame_range,
                                             use_reverse_aug=use_reverse_aug,
                                             load_images=not train,
                                             bg_h5_path=bg_file, bg_depth_h5_path=bg_depth_file, bg_depth_dir=bg_depth_dir,
                                             bg_feats_h5_path=bg_feat_file, odom_h5_path=pred_odom_file)
        elif save_seq_feats:
            mapper = InstanceFullSeqDatasetMapper(cfg)
        else:
            mapper = InstanceSeqDatasetMapper(cfg, augmentations=[], seq_len=seq_len)
        return mapper

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = Trainer._get_dataset_mapper(cfg, train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        name = cfg.SOLVER.LR_SCHEDULER_NAME
        if name == 'CosineAnnealingWarmRestarts':
            min_lr = cfg.SOLVER.MIN_LR
            T0 = cfg.SOLVER.T0
            T_mult = cfg.SOLVER.T_MULT
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T0,
                T_mult,
                eta_min=min_lr,
            )
        return build_lr_scheduler(cfg, optimizer)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(
            self.cfg.TEST.LOSS_EVAL_PERIOD,
            self.model,
            build_detection_val_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                Trainer._get_dataset_mapper(self.cfg)
            )
        ))
        return hooks

    @classmethod
    def build_optimizer(cls, cfg, model):
        params = get_default_optimizer_params(
            model,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
        )
        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            return maybe_add_gradient_clipping(cfg, torch.optim.SGD)(
                params,
                cfg.SOLVER.BASE_LR,
                momentum=cfg.SOLVER.MOMENTUM,
                nesterov=cfg.SOLVER.NESTEROV,
            )
        elif optimizer_type == "ADAM":
            return maybe_add_gradient_clipping(cfg, torch.optim.Adam)(params, cfg.SOLVER.BASE_LR)
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration
            self.start_iter = self.iter + 1
        

def setup(args):
    cfg = get_cfg()
    add_forecast_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    
    return cfg


def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        model.eval()
        res = Trainer.test(cfg, model)
        return res
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )