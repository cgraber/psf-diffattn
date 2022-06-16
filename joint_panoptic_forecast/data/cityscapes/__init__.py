import os
from .datasets import register_cityscapes_saved_feats
from .evaluators import *
from .dataset_mapper import *

_saved_root = os.getenv("DETECTRON2_SAVEDFEAT_FOLDER", 'saved_feats')
_cityscapes_root = os.getenv("DETECTRON2_DATASETS", 'data')
_with_id = bool(os.getenv("DETECTRON2_SAVEDFEAT_WITHID",0))
_with_depths = bool(os.getenv("DETECTRON2_SAVEDFEAT_WITHDEPTHS",0))
_debug_mode = bool(os.getenv("DETECTRON2_DEBUG_MODE",0))
_panseg_dir = os.getenv("DETECTRON2_CITYSCAPES_PANSEG_DIR", None)
new_approach = True
register_cityscapes_saved_feats(_cityscapes_root, _saved_root, _with_id, _with_depths, debug_mode=_debug_mode,
                                panseg_dir=_panseg_dir, new_approach=new_approach)
register_cityscapes_saved_feats(_cityscapes_root, _saved_root, _with_id, _with_depths, target_frame_range=[15, 30], debug_mode=_debug_mode, new_approach=new_approach)
