import copy
import glob
import os
import json
import logging
import pandas as pd
import multiprocessing as mp
import numpy as np
from itertools import chain
from collections import defaultdict
import configparser
import pycocotools.mask as mask_util
import functools
from PIL import Image
from tqdm import tqdm
import pickle

import torch
#torch.multiprocessing.set_sharing_strategy('file_system')

from detectron2.data import DatasetCatalog
from detectron2.utils.comm import get_world_size
from detectron2.structures import Boxes, Instances, BoxMode
from detectron2.utils.file_io import PathManager

try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    pass

logger = logging.getLogger(__name__)


_RAW_CITYSCAPES_SPLITS = {
    "cityscapes_fine_{task}_train": ("cityscapes/leftImg8bit/train/", "cityscapes/gtFine/train/"),
    "cityscapes_fine_{task}_val": ("cityscapes/leftImg8bit/val/", "cityscapes/gtFine/val/"),
    "cityscapes_fine_{task}_test": ("cityscapes/leftImg8bit/test/", "cityscapes/gtFine/test/"),
}

def _get_cityscapes_files(image_dir, gt_dir):
    files = []
    # scan through the directory
    cities = PathManager.ls(image_dir)
    logger.info(f"{len(cities)} cities found in '{image_dir}'.")
    use_largest_seq = bool(os.getenv('CITYSCAPES_LARGEST_SEQ', 0))
    use_single_val = bool(os.getenv('CITYSCAPES_SINGLE_VAL', 0))
    if use_largest_seq:
        print("USING LARGEST SEQ")
    for city in cities:
        city_img_dir = os.path.join(image_dir, city)
        city_gt_dir = os.path.join(gt_dir, city)
        for basename in PathManager.ls(city_img_dir):
            if use_largest_seq:
                img_id = '_'.join(basename.split('_')[:3])
                if img_id != 'zurich_000068_000019':
                    continue
            if use_single_val:
                img_id = '_'.join(basename.split('_')[:3])
                if img_id != 'frankfurt_000000_000294':
                    continue
            image_file = os.path.join(city_img_dir, basename)

            suffix = "leftImg8bit.png"
            assert basename.endswith(suffix), basename
            basename = basename[: -len(suffix)]

            instance_file = os.path.join(city_gt_dir, basename + "gtFine_instanceIds.png")
            label_file = os.path.join(city_gt_dir, basename + "gtFine_labelIds.png")
            json_file = os.path.join(city_gt_dir, basename + "gtFine_polygons.json")

            files.append((image_file, instance_file, label_file, json_file))
    assert len(files), "No images found in {}".format(image_dir)
    for f in files[0]:
        assert PathManager.isfile(f), f
    return files


def _cityscapes_files_to_dict(files, from_json, to_polygons, meta, feat_file,
                              new_approach=False):
    """
    Parse cityscapes annotation files to a instance segmentation dataset dict.
    Args:
        files (tuple): consists of (image_file, instance_id_file, label_id_file, json_file)
        from_json (bool): whether to read annotations from the raw json file or the png files.
        to_polygons (bool): whether to represent the segmentation as polygons
            (COCO's format) instead of masks (cityscapes's format).
    Returns:
        A dict in Detectron2 Dataset format.
    """
    from cityscapesscripts.helpers.labels import id2label, name2label

    image_file, instance_id_file, _, json_file = files

    annos = []

    if from_json:
        from shapely.geometry import MultiPolygon, Polygon

        with PathManager.open(json_file, "r") as f:
            jsonobj = json.load(f)
        ret = {
            "file_name": image_file,
            "image_id": os.path.basename(image_file),
            "height": jsonobj["imgHeight"],
            "width": jsonobj["imgWidth"],
        }
        height = jsonobj["imgHeight"]
        width = jsonobj["imgWidth"]

        # `polygons_union` contains the union of all valid polygons.
        polygons_union = Polygon()

        # CityscapesScripts draw the polygons in sequential order
        # and each polygon *overwrites* existing ones. See
        # (https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/json2instanceImg.py) # noqa
        # We use reverse order, and each polygon *avoids* early ones.
        # This will resolve the ploygon overlaps in the same way as CityscapesScripts.
        for obj in jsonobj["objects"][::-1]:
            if "deleted" in obj:  # cityscapes data format specific
                continue
            label_name = obj["label"]

            try:
                label = name2label[label_name]
            except KeyError:
                if label_name.endswith("group"):  # crowd area
                    label = name2label[label_name[: -len("group")]]
                else:
                    raise
            if label.id < 0:  # cityscapes data format
                continue

            # Cityscapes's raw annotations uses integer coordinates
            # Therefore +0.5 here
            poly_coord = np.asarray(obj["polygon"], dtype="f4") + 0.5
            # CityscapesScript uses PIL.ImageDraw.polygon to rasterize
            # polygons for evaluation. This function operates in integer space
            # and draws each pixel whose center falls into the polygon.
            # Therefore it draws a polygon which is 0.5 "fatter" in expectation.
            # We therefore dilate the input polygon by 0.5 as our input.
            poly = Polygon(poly_coord).buffer(0.5, resolution=4)

            if not label.hasInstances or label.ignoreInEval:
                # even if we won't store the polygon it still contributes to overlaps resolution
                polygons_union = polygons_union.union(poly)
                continue

            # Take non-overlapping part of the polygon
            poly_wo_overlaps = poly.difference(polygons_union)
            if poly_wo_overlaps.is_empty:
                continue
            polygons_union = polygons_union.union(poly)

            anno = {}
            anno["iscrowd"] = label_name.endswith("group")
            anno["category_id"] = label.id

            if isinstance(poly_wo_overlaps, Polygon):
                poly_list = [poly_wo_overlaps]
            elif isinstance(poly_wo_overlaps, MultiPolygon):
                poly_list = poly_wo_overlaps.geoms
            else:
                raise NotImplementedError("Unknown geometric structure {}".format(poly_wo_overlaps))

            poly_coord = []
            for poly_el in poly_list:
                # COCO API can work only with exterior boundaries now, hence we store only them.
                # TODO: store both exterior and interior boundaries once other parts of the
                # codebase support holes in polygons.
                poly_coord.append(list(chain(*poly_el.exterior.coords)))
            anno["segmentation"] = poly_coord
            (xmin, ymin, xmax, ymax) = poly_wo_overlaps.bounds

            anno["bbox"] = (xmin, ymin, xmax, ymax)
            anno["bbox_mode"] = BoxMode.XYXY_ABS

            annos.append(anno)
    else:
        # See also the official annotation parsing scripts at
        # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/instances2dict.py  # noqa
        with PathManager.open(instance_id_file, "rb") as f:
            inst_image = np.asarray(Image.open(f), order="F")
        # ids < 24 are stuff labels (filtering them first is about 5% faster)
        flattened_ids = np.unique(inst_image[inst_image >= 24])

        ret = {
            "file_name": image_file,
            "image_id": os.path.basename(image_file),
            "height": inst_image.shape[0],
            "width": inst_image.shape[1],
        }

        for instance_id in flattened_ids:
            # For non-crowd annotations, instance_id // 1000 is the label_id
            # Crowd annotations have <1000 instance ids
            label_id = instance_id // 1000 if instance_id >= 1000 else instance_id
            label = id2label[label_id]
            if not label.hasInstances or label.ignoreInEval:
                continue

            anno = {}
            anno["iscrowd"] = instance_id < 1000
            anno["category_id"] = label.id

            mask = np.asarray(inst_image == instance_id, dtype=np.uint8, order="F")

            inds = np.nonzero(mask)
            ymin, ymax = inds[0].min(), inds[0].max()
            xmin, xmax = inds[1].min(), inds[1].max()
            anno["bbox"] = (xmin, ymin, xmax, ymax)
            if xmax <= xmin or ymax <= ymin:
                continue
            anno["bbox_mode"] = BoxMode.XYXY_ABS
            if to_polygons:
                # This conversion comes from D4809743 and D5171122,
                # when Mask-RCNN was first developed.
                contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[
                    -2
                ]
                polygons = [c.reshape(-1).tolist() for c in contours if len(c) >= 3]
                # opencv's can produce invalid polygons
                if len(polygons) == 0:
                    continue
                anno["segmentation"] = polygons
            else:
                anno["segmentation"] = mask_util.encode(mask[:, :, None])[0]
            annos.append(anno)
    ret["annotations"] = annos
    image_id = os.path.basename(image_file)
    city, seq, frame = image_id.split('_')[:3]
    datum = meta[(meta['city'] == city)&\
                    (meta['seq'] == seq)&\
                    (meta['frame'] == frame)].iloc[0]
    if new_approach:
        ret['sequence_annotations'] = datum
        ret["feat_file"] = feat_file
        return ret
    frame_offsets = datum['frame_offsets']
    num_instances = list(frame_offsets[1:] - frame_offsets[:-1])
    instances = []
    bboxes = torch.split(torch.FloatTensor(datum['pred_boxes']), num_instances)
    scores = torch.split(torch.FloatTensor(datum['scores']), num_instances)
    pred_classes = torch.split(torch.LongTensor(datum['pred_classes']), num_instances)
        
    if 'ids' in datum.keys():
        try:
            ids = torch.split(torch.LongTensor(datum['ids']), num_instances)
        except:
            print("IDS: ", len(datum['ids']), sum(num_instances))
            raise ValueError()
        use_ids = True
    else:
        ids = [None for _ in range(len(bboxes))]
        use_ids = False
    if 'depths' in datum.keys():
        depths = torch.FloatTensor(datum['depths'])
        depth_masks = depths < 1000000
        depths = torch.split(depths, num_instances)
        depth_masks = torch.split(depth_masks, num_instances)
        use_depths = True
    else:
        depths = depth_masks = [None for _ in range(len(bboxes))]
        use_depths = False
    img_size = (height, width)
    for box, score, cls, _id, depth, depth_mask in zip(bboxes, scores, pred_classes, ids, depths, depth_masks):
        inst = Instances(img_size)
        boxes = Boxes(box)
        inst.gt_boxes = boxes
        inst.pred_boxes = boxes
        inst.pred_classes = cls
        inst.scores = score
        if use_depths:
            inst.depths = depth
            inst.depth_masks = depth_mask
        if use_ids:
            inst.ids = _id
        instances.append(inst)
    #ret = {
    #"height": height,
    #"width": width,
    ret["feat_file"] = feat_file
    ret["instances"] = instances
    ret["frame_offsets"] = frame_offsets
    
    return ret

def load_cityscapes_saved_feats(image_dir, gt_dir, feat_file, meta_file, from_json, to_polygons, target_frame_range=None, debug_mode=False,
                                pan_seg_img_dir=None, pan_seg_anno_file=None,
                                new_approach=False):
    files = _get_cityscapes_files(image_dir, gt_dir)
    if debug_mode:
        files = files[:16]
    all_ret = []
    meta = pd.read_pickle(meta_file)
    logger.info("Preprocessing cityscapes annotations ...")
    logger.info("meta file: ",meta_file)
    logger.info("feat file: ",feat_file)
    print("META FILE: ",meta_file)
    # This is still not fast: all workers will execute duplicate works and will
    # take up to 10m on a 8GPU server.
    pool = mp.Pool(processes=max(mp.cpu_count() // get_world_size() // 2, 4))

    ret = pool.map(
        functools.partial(_cityscapes_files_to_dict, from_json=from_json, to_polygons=to_polygons, meta=meta,feat_file=feat_file,
         new_approach=new_approach),
        files,
    )
    logger.info("Loaded {} images from {}".format(len(ret), image_dir))

    # Map cityscape ids to contiguous ids
    from cityscapesscripts.helpers.labels import labels

    if pan_seg_anno_file is not None:
        with open(pan_seg_anno_file) as fin:
            gt_info = json.load(fin)
        id2idx = {anno['image_id']:idx for idx, anno in enumerate(gt_info['annotations'])}
        final_ret = []
        for dict_per_image in ret:
            img_id = '_'.join(dict_per_image['image_id'].split('_')[:3])
            pan_idx = id2idx[img_id]
            anno = gt_info['annotations'][pan_idx]['segments_info']
            #anno = [a for a in anno if a['category_id'] > 10 and a['iscrowd'] == 0]
            anno = [a for a in anno if a['iscrowd'] == 0]

            dict_per_image['panoptic_annotations'] = anno

            dict_per_image['pan_seg_file_name'] = os.path.join(pan_seg_img_dir, gt_info['annotations'][pan_idx]['file_name'])
            

    labels = [l for l in labels if l.hasInstances and not l.ignoreInEval]
    dataset_id_to_contiguous_id = {l.id: idx for idx, l in enumerate(labels)}
    for dict_per_image in ret:
        for anno in dict_per_image["annotations"]:
            anno["category_id"] = dataset_id_to_contiguous_id[anno["category_id"]]
    if target_frame_range is not None:
        assert(pan_seg_anno_file is None)
        final_ret = []
        for dict_per_image in ret:
            for target_frame in range(target_frame_range[0], target_frame_range[1]):
                new_dict = copy.deepcopy(dict_per_image)
                new_dict['target_frame'] = target_frame
                final_ret.append(new_dict)
        ret = final_ret

    return ret



def _get_builtin_metadata(dataset_name):
    
    if dataset_name == "cityscapes":
        # fmt: off
        CITYSCAPES_THING_CLASSES = [
            "person", "rider", "car", "truck",
            "bus", "train", "motorcycle", "bicycle",
        ]
        CITYSCAPES_STUFF_CLASSES = [
            "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
            "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
            "truck", "bus", "train", "motorcycle", "bicycle",
        ]
        # fmt: on
        return {
            "thing_classes": CITYSCAPES_THING_CLASSES,
            "stuff_classes": CITYSCAPES_STUFF_CLASSES,
        }
    raise KeyError("No built-in metadata for dataset {}".format(dataset_name))

def register_cityscapes_saved_feats(cityscapes_root, saved_root, use_id, use_depths, target_frame_range=None, debug_mode=False,
                                    panseg_dir=None, bg_file=None, new_approach=False):
    for key, (image_dir, gt_dir) in _RAW_CITYSCAPES_SPLITS.items():
        meta = _get_builtin_metadata("cityscapes")
        image_dir = os.path.join(cityscapes_root, image_dir)
        gt_dir = os.path.join(cityscapes_root, gt_dir)
        if target_frame_range is not None:
            saved_key = key.format(task="saved_feats_range%d_%d"%(target_frame_range[0],target_frame_range[1]))
        else:
            saved_key = key.format(task="saved_feats")

        if use_id and use_depths:
            task = 'instance_seg_withid_withdepth'
        elif use_depths:
            task = 'instance_seg_withdepth'
        elif use_id:
            task = 'instance_seg_withid'
        else:
            task = 'instance_seg'
        inst_key = key.format(task=task)
        feats_key = key.format(task='instance_seg')
        feat_file = os.path.join(saved_root, '%s_feats.h5'%feats_key)
        meta_file = os.path.join(saved_root, '%s_meta.pkl'%inst_key)
        if panseg_dir is not None:
            split = key.split('_')[-1]
            current_panseg_dir = os.path.join(panseg_dir, 'cityscapes_panoptic_%s_trainId'%split)
            current_panseg_anno_file = os.path.join(panseg_dir, 'cityscapes_panoptic_%s_trainId.json'%split)
        else:
            current_panseg_dir = None
            current_panseg_anno_file = None
        DatasetCatalog.register(
            saved_key,
            lambda w=image_dir, x=gt_dir, y=feat_file, z=meta_file, q=current_panseg_dir,r=current_panseg_anno_file,s=new_approach: load_cityscapes_saved_feats(
                w, x, y, z, True, True, target_frame_range, debug_mode,
                pan_seg_img_dir=q, 
                pan_seg_anno_file=r, new_approach=s,
            )
        )


