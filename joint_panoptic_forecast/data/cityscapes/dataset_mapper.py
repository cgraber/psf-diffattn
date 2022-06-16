import copy
import json
import logging
import os
import h5py
from typing import Callable, Dict, List, Optional, Union, Tuple
import pycocotools.mask as mask_util

import numpy as np
import torch

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import Boxes, Instances, PolygonMasks

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')




class InstanceSeqDatasetMapper:


    @configurable
    def __init__(
        self,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        seq_len: int,
        seq_interval: int,
        instance_mask_format: str = "polygon",
        recompute_boxes: bool = False,
        target_frame_range = None,
    ):
        self.augmentations = T.AugmentationList(augmentations)
        self.image_format = image_format
        logger = logging.getLogger(__name__)
        logger.info("Augmentations used in training: " + str(augmentations))
        self.seq_len = seq_len
        self.seq_interval = seq_interval
        self.instance_mask_format = instance_mask_format
        self.recompute_boxes = recompute_boxes
        self.target_frame_range = target_frame_range
        print("TARGET FRAME RANGE: ", self.target_frame_range)

    @classmethod
    def from_config(cls, cfg):
        augs = [] #for now, assume no data augmentation
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        recompute_boxes = False #TODO: if we ever add cropping, might need to change
        ret = {
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "seq_len": cfg.INPUT.SEQ_LEN,
            "seq_interval": cfg.INPUT.SEQ_INTERVAL,
            "recompute_boxes": recompute_boxes,
            "target_frame_range": cfg.INPUT.TARGET_FRAME_RANGE,
        }
        return ret

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        final_image = dataset_dict["file_name"]
        orig_image_dir = os.path.dirname(final_image)
        image_dir = orig_image_dir.replace('leftImg8bit', 'leftImg8bit_sequence')
        city, seq, frame = os.path.basename(final_image).split('_')[:3]
        frame = int(frame)
        if 'target_frame' in dataset_dict:
            target_frame = frame - 19 + dataset_dict['target_frame']
        elif self.target_frame_range is not None:
            target_idx = np.random.randint(*self.target_frame_range)
            target_frame = frame - 19 + target_idx
        else:
            target_frame = frame
        imgs = []
        for fr in range(target_frame - (self.seq_len-1)*self.seq_interval, target_frame+1, self.seq_interval):
            im_path = os.path.join(image_dir, '%s_%s_%06d_leftImg8bit.png'%(city, seq, fr))
            img = utils.read_image(im_path, format=self.image_format)
            utils.check_image_size(dataset_dict, img)
            imgs.append(img)
        image_shape = imgs[0].shape[:2]

        # load egomotion stuff
        timestamp_dir = orig_image_dir.replace('leftImg8bit', 'timestamp_sequence')
        odom_dir = orig_image_dir.replace('leftImg8bit', 'vehicle_sequence')
        times = []
        odometries = []
        for fr in range(target_frame - (self.seq_len-1)*self.seq_interval, target_frame+1):
            time_path = os.path.join(timestamp_dir, '%s_%s_%06d_timestamp.txt'%(city, seq, fr))
            with open(time_path, 'r') as fin:
                time = float(fin.read())
            times.append(time/1e9)
            odom_path = os.path.join(odom_dir, '%s_%s_%06d_vehicle.json'%(city, seq, fr))
            with open(odom_path, "r") as f:
                odom_dict = json.load(f)
            speed = odom_dict.get('speed')
            yaw_rate = odom_dict.get('yawRate')
            odom = torch.zeros(5)
            odom[0] = speed
            odom[1] = yaw_rate
            if fr > target_frame - (self.seq_len-1)*self.seq_interval:
                delta_t = times[-1] - times[-2]
                dx, dy, dtheta = get_vehicle_now_T_prev(speed, yaw_rate, delta_t)
                odom[2] = dx
                odom[3] = dy
                odom[4] = dtheta
            
            odometries.append(odom)
        odometries = torch.stack(odometries)
        if self.seq_interval > 1:
            odometries = odometries[::self.seq_interval]
        dataset_dict["odometries"] = odometries


                

        

        # eventually, might want to do augmentations here
        transforms = T.TransformList([])

        dataset_dict["images"] = [torch.as_tensor(
            np.ascontiguousarray(img.transpose(2, 0, 1))
        ) for img in imgs]
        #TODO: do we need anything else for gt processing?
        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                anno.pop("keypoints", None)
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances) #TODO: maybe not for forecasting? 

        
        return dataset_dict


class InstanceFullSeqDatasetMapper:


    @configurable
    def __init__(
        self,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        instance_mask_format: str = "polygon",
        recompute_boxes: bool = False,
    ):
        self.augmentations = T.AugmentationList(augmentations)
        self.image_format = image_format
        logger = logging.getLogger(__name__)
        logger.info("Augmentations used in training: " + str(augmentations))
        self.instance_mask_format = instance_mask_format
        self.recompute_boxes = recompute_boxes

    @classmethod
    def from_config(cls, cfg):
        augs = [] #for now, assume no data augmentation
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        recompute_boxes = False #TODO: if we ever add cropping, might need to change
        ret = {
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "recompute_boxes": recompute_boxes,
        }
        return ret

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        final_image = dataset_dict["file_name"]
        orig_image_dir = os.path.dirname(final_image)
        image_dir = orig_image_dir.replace('leftImg8bit', 'leftImg8bit_sequence')
        city, seq, frame = os.path.basename(final_image).split('_')[:3]
        frame = int(frame)
        imgs = []
        for fr in range(frame - 19, frame+11):
            im_path = os.path.join(image_dir, '%s_%s_%06d_leftImg8bit.png'%(city, seq, fr))
            img = utils.read_image(im_path, format=self.image_format)
            utils.check_image_size(dataset_dict, img)
            imgs.append(img)
        image_shape = imgs[0].shape[:2]

        # load egomotion stuff
        timestamp_dir = orig_image_dir.replace('leftImg8bit', 'timestamp_sequence')
        odom_dir = orig_image_dir.replace('leftImg8bit', 'vehicle_sequence')
        times = []
        odometries = []
        for fr in range(frame - 19, frame+11):
            time_path = os.path.join(timestamp_dir, '%s_%s_%06d_timestamp.txt'%(city, seq, fr))
            with open(time_path, 'r') as fin:
                time = float(fin.read())
            times.append(time/1e9)
            odom_path = os.path.join(odom_dir, '%s_%s_%06d_vehicle.json'%(city, seq, fr))
            with open(odom_path, "r") as f:
                odom_dict = json.load(f)
            speed = odom_dict.get('speed')
            yaw_rate = odom_dict.get('yawRate')
            odom = torch.zeros(5)
            odom[0] = speed
            odom[1] = yaw_rate
            if fr > frame - 19:
                delta_t = times[-1] - times[-2]
                dx, dy, dtheta = get_vehicle_now_T_prev(speed, yaw_rate, delta_t)
                odom[2] = dx
                odom[3] = dy
                odom[4] = dtheta
            
            odometries.append(odom)
        odometries = torch.stack(odometries)
        dataset_dict["odometries"] = odometries


                

        

        # eventually, might want to do augmentations here
        transforms = T.TransformList([])

        dataset_dict["images"] = [torch.as_tensor(
            np.ascontiguousarray(img.transpose(2, 0, 1))
        ) for img in imgs]

        #TODO: do we need anything else for gt processing?
        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                anno.pop("keypoints", None)
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances) #TODO: maybe not for forecasting? 

        return dataset_dict


class InstanceSavedFeatMapper:


    @configurable
    def __init__(
        self,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        seq_len: int,
        seq_interval: int,
        instance_mask_format: str = "polygon",
        recompute_boxes: bool = False,
        use_id: bool = False,
        target_frame_range = None,
        use_reverse_aug: bool = False,
        vizemb_mapper = None,
        load_images = False,
        bg_h5_path = None,
        bg_depth_h5_path = None,
        bg_depth_dir = None,
        new_approach = False,
        bg_feats_h5_path = None,
        odom_h5_path = None,
        target_frame = None,
        orbslam_odom_meta_path = None,
    ):
        self.augmentations = T.AugmentationList(augmentations)
        self.image_format = image_format
        logger = logging.getLogger(__name__)
        logger.info("Augmentations used in training: " + str(augmentations))
        self.instance_mask_format = instance_mask_format
        self.recompute_boxes = recompute_boxes
        self.seq_len = seq_len
        self.seq_interval = seq_interval
        self.use_id = use_id
        self.target_frame_range = target_frame_range
        self.use_reverse_aug = use_reverse_aug
        print("TARGET FRAME RANGE: ",self.target_frame_range)
        if self.use_reverse_aug:
            print("USING REVERSE AUG")
        self.vizemb_mapper = vizemb_mapper
        self.load_images = load_images
        self.bg_h5_path = bg_h5_path
        self.bg_h5 = None
        self.bg_depth_h5_path = bg_depth_h5_path
        self.bg_depth_h5 = None
        self.new_approach = new_approach
        self.bg_depth_dir = bg_depth_dir
        self.bg_feats_h5_path = bg_feats_h5_path
        self.bg_feats_h5 = None
        self.odom_h5_path = odom_h5_path
        self.odom_h5 = None
        if target_frame is None:
            self.target_frame = 19
        else:
            self.target_frame = target_frame
        if orbslam_odom_meta_path is not None:
            import pandas as pd
            self.orbslam_odom_meta = pd.read_pickle(orbslam_odom_meta_path)
        else:
            self.orbslam_odom_meta = None

    @classmethod
    def from_config(cls, cfg):
        augs = [] #for now, assume no data augmentation
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        recompute_boxes = False #TODO: if we ever add cropping, might need to change
        #new_approach = bool(os.getenv("DETECTRON2_NEW_APPROACH", 0))
        ret = {
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "recompute_boxes": recompute_boxes,
            "seq_len": cfg.INPUT.SEQ_LEN,
            "seq_interval": cfg.INPUT.SEQ_INTERVAL,
            "use_id": cfg.INPUT.USE_ID,
            "target_frame_range": cfg.INPUT.TARGET_FRAME_RANGE,
            "new_approach": True,
            "target_frame": cfg.INPUT.TARGET_FRAME,
            "orbslam_odom_meta_path": cfg.INPUT.ORBSLAM_ODOM_META_PATH,
        }
        return ret

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        if self.vizemb_mapper is not None:
            # Load data augmentation
            vizemb_dataset_dict = self.vizemb_mapper(dataset_dict)
            dataset_dict['vizemb_input'] = vizemb_dataset_dict
        if self.use_reverse_aug and np.random.random() > 0.5:
            flip = True
        else:
            flip = False
        final_image = dataset_dict["file_name"]
        orig_image_dir = os.path.dirname(final_image)
        
        city, seq, frame = os.path.basename(final_image).split('_')[:3]

        img = utils.read_image(final_image, format=self.image_format)
        utils.check_image_size(dataset_dict, img)
        image_shape = img.shape[:2]

        if self.new_approach:
            all_feats = []
            name = '%s/%s/%s'%(city, seq, frame)
            final_instances = []
            transforms = T.TransformList([])
            all_ids = set()
            if self.target_frame_range is None:
                target_frame = self.target_frame
            else:
                target_frame = np.random.randint(*self.target_frame_range)
            seq_annos = dataset_dict['sequence_annotations']
            frame_offsets = seq_annos['frame_offsets']
            num_instances = list(frame_offsets[1:] - frame_offsets[:-1])
            bboxes = torch.split(torch.FloatTensor(seq_annos['pred_boxes']), num_instances)
            scores = torch.split(torch.FloatTensor(seq_annos['scores']), num_instances)
            pred_classes = torch.split(torch.LongTensor(seq_annos['pred_classes']), num_instances)
            if 'ids' in seq_annos:
                ids = torch.split(torch.LongTensor(seq_annos['ids']), num_instances)
            else:
                ids = None
            if 'depths' in seq_annos:
                depths = torch.FloatTensor(seq_annos['depths'])
                depth_masks = depths < 1000000
                depths = torch.split(depths, num_instances)
                depth_masks = torch.split(depth_masks, num_instances)
                use_depths = True
            else:
                use_depths = False
            with h5py.File(dataset_dict['feat_file'], 'r') as fin:
                dset = fin[name]

                for fr in range(target_frame - (self.seq_len-1)*self.seq_interval, target_frame+1, self.seq_interval):
                    o0 = frame_offsets[fr]
                    o1 = frame_offsets[fr+1]
                    new_feats = torch.FloatTensor(dset[o0:o1])
                    new_inst = Instances(image_shape)
                    new_inst.gt_boxes = Boxes(bboxes[fr])
                    new_inst.pred_boxes = new_inst.gt_boxes
                    new_inst.pred_classes = pred_classes[fr]
                    new_inst.scores = scores[fr]
                    if use_depths:
                        new_inst.depths = depths[fr]
                        new_inst.depth_masks = depth_masks[fr]
                    new_inst.features = new_feats
                    if ids is not None:

                        new_inst.ids = ids[fr]
                        for id in new_inst.ids:
                            if id != -1:
                                all_ids.add(id.item())
                        valid = new_inst.ids != -1
                        new_inst = new_inst[torch.arange(len(valid))[valid]]
                    final_instances.append(new_inst)
        else:
            # properly create feats, bounding boxes
            all_feats = []
            name = '%s/%s/%s'%(city, seq, frame)
            final_instances = []
            transforms = T.TransformList([])
            all_instances = dataset_dict['instances']
            all_ids = set()
            if self.target_frame_range is None:
                target_frame = self.target_frame
            else:
                target_frame = np.random.randint(*self.target_frame_range)
            with h5py.File(dataset_dict['feat_file'], 'r', swmr=True) as fin:
                dset = fin[name]
                for fr in range(target_frame - (self.seq_len-1)*self.seq_interval, target_frame+1, self.seq_interval):
                    o0 = dataset_dict['frame_offsets'][fr]
                    o1 = dataset_dict['frame_offsets'][fr+1]
                    #all_feats.append(torch.FloatTensor(dset[o0:o1]))
                    #final_instances.append(all_instances[fr])
                    new_feats = torch.FloatTensor(dset[o0:o1])
                    new_instances = all_instances[fr]
                    new_instances.features = new_feats
                    if self.use_id:
                        for id in new_instances.ids:
                            if id != -1:
                                all_ids.add(id.item())
                        valid = new_instances.ids != -1
                        new_feats = new_feats[valid]
                        new_instances = new_instances[torch.arange(len(valid))[valid]]
                    all_feats.append(new_feats)
                    final_instances.append(new_instances)
        if self.vizemb_mapper is not None or self.load_images:
            imgs = []
            image_dir = orig_image_dir.replace('leftImg8bit', 'leftImg8bit_sequence')
            for fr in range(target_frame - (self.seq_len-1)*self.seq_interval, target_frame+1, self.seq_interval):
                im_path = os.path.join(image_dir, '%s_%s_%06d_leftImg8bit.png'%(city, seq, int(frame)-19+fr))
                img = utils.read_image(im_path, format=self.image_format)
                utils.check_image_size(dataset_dict, img)
                imgs.append(img)
            image_shape = imgs[0].shape[:2]
            dataset_dict["images"] = [torch.as_tensor(
                np.ascontiguousarray(img.transpose(2, 0, 1))
            ) for img in imgs]

        if self.use_id:
            convert_id = {}
            for id_idx, id in enumerate(all_ids):
                convert_id[id] = id_idx
            for inst in final_instances:
                inst.ids = torch.LongTensor([convert_id[id.item()] for id in inst.ids])
        if flip:
            final_instances = final_instances[::-1]
            #all_feats = all_feats[::-1]
            
        #all_feats = torch.cat(all_feats)
        #dataset_dict['features'] = all_feats
        dataset_dict['proposals'] = final_instances


        frame = int(frame)


        # load egomotion stuff
        timestamp_dir = orig_image_dir.replace('leftImg8bit', 'timestamp_sequence')
        odom_dir = orig_image_dir.replace('leftImg8bit', 'vehicle_sequence')
        times = []
        odometries = []
        if self.orbslam_odom_meta is not None:
            datum3d = self.orbslam_odom_meta[(self.orbslam_odom_meta['city'] == city)&
                                              (self.orbslam_odom_meta['seq'] == seq)&
                                              (self.orbslam_odom_meta['frame']==frame)].iloc[0]
            odometry = np.stack([
                datum3d['speed'],
                datum3d['yaw_rate'],
                datum3d['dx'],
                datum3d['dy'],
                datum3d['dtheta']
            ], axis=-1)[range(target_frame - (self.seq_len-1)*self.seq_interval, target_frame+1)]
            odometries = torch.from_numpy(odometry).float()
        else:
            for fr in range(frame-19+target_frame - (self.seq_len-1)*self.seq_interval, frame-19+target_frame+1):
                time_path = os.path.join(timestamp_dir, '%s_%s_%06d_timestamp.txt'%(city, seq, fr))
                with open(time_path, 'r') as fin:
                    time = float(fin.read())
                times.append(time/1e9)
                odom_path = os.path.join(odom_dir, '%s_%s_%06d_vehicle.json'%(city, seq, fr))
                with open(odom_path, "r") as f:
                    odom_dict = json.load(f)
                speed = odom_dict.get('speed')
                yaw_rate = odom_dict.get('yawRate')
                odom = torch.zeros(5)
                odom[0] = speed
                odom[1] = yaw_rate
                if fr > frame -19+target_frame - (self.seq_len-1)*self.seq_interval:
                    delta_t = times[-1] - times[-2]
                    dx, dy, dtheta = get_vehicle_now_T_prev(speed, yaw_rate, delta_t)
                    odom[2] = dx
                    odom[3] = dy
                    odom[4] = dtheta
                
                odometries.append(odom)
            if flip:
                odometries = odometries[::-1]
            odometries = torch.stack(odometries)
        if self.seq_interval > 1:
            odometries = odometries[::self.seq_interval]
        if flip:
            odometries *= -1
        dataset_dict["odometries"] = odometries
        if self.odom_h5_path is not None:
            if self.odom_h5 is None:
                self.odom_h5 = h5py.File(self.odom_h5_path, 'r')
            start_fr = target_frame - (self.seq_len//2)*self.seq_interval
            name = '%s/%s/%d/%d'%(city, seq, int(frame), start_fr)
            try:
                odom_preds = self.odom_h5[name][:]
            except Exception as e:
                print("FAILING NAME: ",name)
                raise e
            final_pred_odom = []
            inp_times = np.array(times[:len(times)//2])
            avg_delta_t = np.mean(inp_times[1:]-inp_times[:-1])
            for odom_idx in range(len(odom_preds)):
                
                speed, yaw_rate = odom_preds[odom_idx]
                dx, dy, dtheta = get_vehicle_now_T_prev(speed, yaw_rate, avg_delta_t)
                final_pred_odom.append(torch.FloatTensor([speed, yaw_rate, dx, dy, dtheta]))
            final_pred_odom = torch.stack(final_pred_odom)
            if self.seq_interval > 1:
                final_pred_odom = final_pred_odom[self.seq_interval-1::self.seq_interval]
            dataset_dict['pred_future_odometries'] = final_pred_odom
            
            


        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                anno.pop("keypoints", None)
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances) #TODO: maybe not for forecasting? 

        if self.bg_h5_path is not None:
            if self.target_frame == 19:
                sf =0
            elif self.target_frame == 25:
                sf = 2
            import panopticapi.utils
            base_name = '%s/%s/%d/%d'%(city, seq, int(frame), sf)
            if self.bg_h5 is None:
                self.bg_h5 = h5py.File(self.bg_h5_path, 'r')
            bg_logits = torch.from_numpy(self.bg_h5[base_name][:])[:11]
            dataset_dict['bg_logits'] = bg_logits

            
            gt_img = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")
            gt_pan_seg = torch.from_numpy(panopticapi.utils.rgb2id(gt_img)).long()
            panoptic_annos = dataset_dict.pop('panoptic_annotations')
            final_gt_pan_seg = gt_pan_seg.clone()
            for a in panoptic_annos:
                if a['category_id'] <= 10:
                    final_gt_pan_seg[gt_pan_seg == a['id']] = a['category_id']
            dataset_dict['gt_pan_seg'] = final_gt_pan_seg
            panoptic_annos = [a for a in panoptic_annos if a['category_id'] > 10]
            bboxes = torch.tensor([a['bbox'] for a in panoptic_annos], dtype=float)
            if len(bboxes.shape) == 1:
                bboxes = bboxes.unsqueeze(0)
            bboxes[:, 2:] += bboxes[:, :2]
            classes = torch.tensor([a['category_id']-11 for a in panoptic_annos], dtype=torch.long)
            ids = torch.tensor([a['id'] for a in panoptic_annos], dtype=torch.long)
            p_inst = Instances(image_shape)
            p_inst.gt_boxes = Boxes(bboxes)
            p_inst.gt_classes = classes
            p_inst.ids = ids
            dataset_dict['gt_pan_instances'] = p_inst
        if self.bg_feats_h5_path is not None:
            if self.target_frame == 19:
                sf =0
            elif self.target_frame == 25:
                sf = 2
            else:
                raise ValueError('BG depth loading not implemented for target frame: ', self.target_frame)
            base_name = '%s/%s/%d/%d'%(city, seq, int(frame), sf)
            if self.bg_feats_h5 is None:
                self.bg_feats_h5 = h5py.File(self.bg_feats_h5_path, 'r')
            bg_feats = torch.from_numpy(self.bg_feats_h5[base_name][:])
            dataset_dict['bg_feats'] = bg_feats


        if self.bg_depth_h5_path is not None or self.bg_depth_dir is not None:
            import cv2
            if self.bg_depth_h5_path is not None:
                if self.bg_depth_h5 is None:
                    self.bg_depth_h5 = h5py.File(self.bg_depth_h5_path, 'r')
                if self.target_frame == 19:
                    sf =0
                elif self.target_frame == 25:
                    sf = 2
                else:
                    raise ValueError('BG depth loading not implemented for target frame: ', self.target_frame)
                name = '%s/%s/%06d/%d'%(city, seq, int(frame), sf)
                bg_depths = self.bg_depth_h5[name][:]
            else:
                if self.target_frame not in [19, 25]:
                    raise ValueError('need to check for short term:', self.target_frame)
                name = "%s_%s_%06d_depths.png"%(city, seq, int(frame))
                base_path = os.path.join(self.bg_depth_dir, city, name)
                all_depths = []
                for i in range(3):
                    depth_path = base_path % i
                    depths = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
                    all_depths.append(depths)
                bg_depths = np.stack(all_depths, -1)

            #dsize = (int(bg_depths.shape[1]/4), int(bg_depths.shape[0]/4))
            #bg_depths = cv2.resize(bg_depths, dsize=dsize, interpolation=cv2.INTER_NEAREST)
            bg_depths = torch.from_numpy(bg_depths.astype(np.float32)).float()
            bg_depths = bg_depths.permute(2, 0, 1)
            bg_depths = bg_depths / 256.0 - 1
            bg_depth_masks = (bg_depths > 0) * (bg_depths != 254)
            bg_depths[~bg_depth_masks] = -1
            dataset_dict['bg_depths'] = bg_depths
            dataset_dict['bg_depth_masks'] = bg_depth_masks
            

        return dataset_dict


class InstanceSavedFeatFullSeqMapper:


    @configurable
    def __init__(
        self,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        seq_len: int,
        seq_interval: int,
        instance_mask_format: str = "polygon",
        recompute_boxes: bool = False,
        use_id: bool = False,
        target_frame_range = None,
        use_reverse_aug: bool = False,
        vizemb_mapper = None,
        load_images = False,
    ):
        self.augmentations = T.AugmentationList(augmentations)
        self.image_format = image_format
        logger = logging.getLogger(__name__)
        logger.info("Augmentations used in training: " + str(augmentations))
        self.instance_mask_format = instance_mask_format
        self.recompute_boxes = recompute_boxes
        self.seq_len = seq_len
        self.seq_interval = seq_interval
        self.use_id = use_id
        self.target_frame_range = target_frame_range
        self.use_reverse_aug = use_reverse_aug
        print("TARGET FRAME RANGE: ",self.target_frame_range)
        if self.use_reverse_aug:
            print("USING REVERSE AUG")
        self.vizemb_mapper = vizemb_mapper
        self.load_images = load_images

    @classmethod
    def from_config(cls, cfg):
        augs = [] #for now, assume no data augmentation
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        recompute_boxes = False #TODO: if we ever add cropping, might need to change
        ret = {
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "recompute_boxes": recompute_boxes,
            "seq_len": cfg.INPUT.SEQ_LEN,
            "seq_interval": cfg.INPUT.SEQ_INTERVAL,
            "use_id": cfg.INPUT.USE_ID,
            "target_frame_range": cfg.INPUT.TARGET_FRAME_RANGE,
        }
        return ret

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        final_image = dataset_dict["file_name"]
        orig_image_dir = os.path.dirname(final_image)
        
        city, seq, frame = os.path.basename(final_image).split('_')[:3]

        img = utils.read_image(final_image, format=self.image_format)
        utils.check_image_size(dataset_dict, img)
        image_shape = img.shape[:2]

        # properly create feats, bounding boxes
        all_feats = []
        name = '%s/%s/%s'%(city, seq, frame)
        final_instances = []
        transforms = T.TransformList([])
        all_ids = set()
        target_frame = 19
        seq_annos = dataset_dict['sequence_annotations']
        frame_offsets = seq_annos['frame_offsets']
        num_instances = list(frame_offsets[1:] - frame_offsets[:-1])
        bboxes = torch.split(torch.FloatTensor(seq_annos['pred_boxes']), num_instances)
        scores = torch.split(torch.FloatTensor(seq_annos['scores']), num_instances)
        pred_classes = torch.split(torch.LongTensor(seq_annos['pred_classes']), num_instances)
        if 'ids' in seq_annos:
            ids = torch.split(torch.LongTensor(seq_annos['ids']), num_instances)
        else:
            ids = None
        if 'depths' in seq_annos:
            depths = torch.FloatTensor(seq_annos['depths'])
            depth_masks = depths < 1000000
            depths = torch.split(depths, num_instances)
            depth_masks = torch.split(depth_masks, num_instances)
            use_depths = True
        else:
            use_depths = False
        with h5py.File(dataset_dict['feat_file'], 'r', swmr=True) as fin:
            dset = fin[name]
            for fr in range(30):
                o0 = frame_offsets[fr]
                o1 = frame_offsets[fr+1]
                new_feats = torch.FloatTensor(dset[o0:o1])
                new_inst = Instances(image_shape)
                new_inst.gt_boxes = Boxes(bboxes[fr])
                new_inst.pred_boxes = new_inst.gt_boxes
                new_inst.pred_classes = pred_classes[fr]
                new_inst.scores = scores[fr]
                if use_depths:
                    new_inst.depths = depths[fr]
                    new_inst.depth_masks = depth_masks[fr]
                new_inst.features = new_feats
                if ids is not None:
                    new_inst.ids = ids[fr]
                    for id in new_inst.ids:
                        if id != -1:
                            all_ids.add(id.item())
                    valid = new_inst.ids != -1
                    new_inst = new_inst[torch.arange(len(valid))[valid]]
                
                #all_feats.appendnew_feats)
                final_instances.append(new_inst)
        if self.vizemb_mapper is not None or self.load_images:
            imgs = []
            image_dir = orig_image_dir.replace('leftImg8bit', 'leftImg8bit_sequence')
            for fr in range(30):
                im_path = os.path.join(image_dir, '%s_%s_%06d_leftImg8bit.png'%(city, seq, int(frame)-19+fr))
                img = utils.read_image(im_path, format=self.image_format)
                utils.check_image_size(dataset_dict, img)
                imgs.append(img)
            image_shape = imgs[0].shape[:2]
            dataset_dict["images"] = [torch.as_tensor(
                np.ascontiguousarray(img.transpose(2, 0, 1))
            ) for img in imgs]

        if self.use_id:
            convert_id = {}
            for id_idx, id in enumerate(all_ids):
                convert_id[id] = id_idx
            for inst in final_instances:
                inst.ids = torch.LongTensor([convert_id[id.item()] for id in inst.ids])
            
        #all_feats = torch.cat(all_feats)
        #dataset_dict['features'] = all_feats
        dataset_dict['proposals'] = final_instances


        frame = int(frame)


        # load egomotion stuff
        timestamp_dir = orig_image_dir.replace('leftImg8bit', 'timestamp_sequence')
        odom_dir = orig_image_dir.replace('leftImg8bit', 'vehicle_sequence')
        times = []
        odometries = []
        for fr in range(frame-19+target_frame - (self.seq_len-1)*self.seq_interval, frame-19+target_frame+1):
            time_path = os.path.join(timestamp_dir, '%s_%s_%06d_timestamp.txt'%(city, seq, fr))
            with open(time_path, 'r') as fin:
                time = float(fin.read())
            times.append(time/1e9)
            odom_path = os.path.join(odom_dir, '%s_%s_%06d_vehicle.json'%(city, seq, fr))
            with open(odom_path, "r") as f:
                odom_dict = json.load(f)
            speed = odom_dict.get('speed')
            yaw_rate = odom_dict.get('yawRate')
            odom = torch.zeros(5)
            odom[0] = speed
            odom[1] = yaw_rate
            if fr > frame -19+target_frame - (self.seq_len-1)*self.seq_interval:
                delta_t = times[-1] - times[-2]
                dx, dy, dtheta = get_vehicle_now_T_prev(speed, yaw_rate, delta_t)
                odom[2] = dx
                odom[3] = dy
                odom[4] = dtheta
            
            odometries.append(odom)
        odometries = torch.stack(odometries)
        if self.seq_interval > 1:
            odometries = odometries[::self.seq_interval]
        dataset_dict["odometries"] = odometries

        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                anno.pop("keypoints", None)
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances) #TODO: maybe not for forecasting? 
        
        return dataset_dict


class InstanceSavedFeatSingleFrameMapper:


    @configurable
    def __init__(
        self,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        seq_len: int,
        seq_interval: int,
        instance_mask_format: str = "polygon",
        recompute_boxes: bool = False,
        target_frame_range = None,
    ):
        self.augmentations = T.AugmentationList(augmentations)
        self.image_format = image_format
        logger = logging.getLogger(__name__)
        logger.info("Augmentations used in training: " + str(augmentations))
        self.instance_mask_format = instance_mask_format
        self.recompute_boxes = recompute_boxes
        self.seq_len = seq_len
        self.seq_interval = seq_interval
        self.target_frame_range = target_frame_range
        print("TARGER FRAME RANGE: ",self.target_frame_range)

    @classmethod
    def from_config(cls, cfg):
        augs = [] #for now, assume no data augmentation
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        recompute_boxes = False #TODO: if we ever add cropping, might need to change
        ret = {
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "recompute_boxes": recompute_boxes,
            "seq_len": cfg.INPUT.SEQ_LEN,
            "seq_interval": cfg.INPUT.SEQ_INTERVAL,
            "target_frame_range": cfg.INPUT.TARGET_FRAME_RANGE,
        }
        return ret

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        final_image = dataset_dict["file_name"]
        orig_image_dir = os.path.dirname(final_image)
        
        city, seq, frame = os.path.basename(final_image).split('_')[:3]

        img = utils.read_image(final_image, format=self.image_format)
        utils.check_image_size(dataset_dict, img)
        image_shape = img.shape[:2]

        # properly create feats, bounding boxes
        all_feats = []
        name = '%s/%s/%s'%(city, seq, frame)
        final_instances = []
        transforms = T.TransformList([])
        all_instances = dataset_dict['instances']
        all_ids = set()
        if self.target_frame_range is None:
            fr = 19
        else:
            fr = np.random.randint(*self.target_frame_range)
        with h5py.File(dataset_dict['feat_file'], 'r') as fin:
            dset = fin[name]
            o0 = dataset_dict['frame_offsets'][fr]
            o1 = dataset_dict['frame_offsets'][fr+1]
            #all_feats.append(torch.FloatTensor(dset[o0:o1]))
            #final_instances.append(all_instances[fr])
            new_feats = torch.FloatTensor(dset[o0:o1])
            new_instances = all_instances[fr]
            new_instances.features = new_feats

            
        #all_feats = torch.cat(all_feats)
        #dataset_dict['features'] = new_feats
        dataset_dict['proposals'] = new_instances

        frame = int(frame)


        

        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                anno.pop("keypoints", None)
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances) #TODO: maybe not for forecasting? 

        
        return dataset_dict


def get_vehicle_now_T_prev(speed, yaw_rate, delta_t):
    # The vehicle motion is stored in a 2d velocity model. The vehicle is assumed to move on a horizontal plan.
    # Thus, if the vehicle bumps or moves on a hill, this motion model can be inaccurate.

    # 2d pose representation:
    # The pose of the vehicle in a 2d top-down view coordinate is represented as [x, y, theta], where (x, y) specifies
    # the location, and theta specifies the heading direction. The coordinate is defined as z-up, and the vehicle moves
    # on the x-y plane.
    #
    # World coordinate:
    # In this function, define the global coordinate to be the vehicle's coordinate at the previous frame.
    # Thus, (x_prev = 0, y_prev = 0, theta_prev = 0).
    #
    # Steps:
    # 1. Calculate the current vehicle pose in the world coordinate.
    #    (Good reference for velocity model: Section 5.3 in https://ccc.inaoep.mx/~mdprl/documentos/CH5.pdf)
    # 2. Derive the 3x3 (3-dof) the relative poses between two frames in the world coordinate.
    # 3. Expand the 3x3 3-dof transform to a 4x4 6-dof transform assuming the vehicle travels on the x-y plane.

    # Handle special case where the angular velocity is close to 0. The vehicle is moving front.
    angle_rad_eps = 0.000175  # ~0.01 deg
    if abs(yaw_rate) < angle_rad_eps:
        x = delta_t * speed
        y = 0.0
        theta = 0.0
    else:
        # Follow Equation (5.9) in the reference by setting the previous pose to (0, 0, 0).
        r = speed / yaw_rate
        wt = yaw_rate * delta_t
        x = r * np.sin(wt)
        y = r - r * np.cos(wt)
        theta = wt

    return x, y, theta


class DatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.
    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.
    The callable currently does the following:
    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            precomputed_proposal_topk: if given, will load pre-computed
                proposals from dataset_dict and keep the top k proposals for each image.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.use_keypoint           = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk          = precomputed_proposal_topk
        self.recompute_boxes        = recompute_boxes
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = utils.build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
        }

        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)

        if cfg.MODEL.LOAD_PROPOSALS:
            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        return ret

    def _transform_annotations(self, dataset_dict, transforms, image_shape):
        # USER: Modify this if you want to keep them for some reason.
        for anno in dataset_dict["annotations"]:
            if not self.use_instance_mask:
                anno.pop("segmentation", None)
            if not self.use_keypoint:
                anno.pop("keypoints", None)

        # USER: Implement additional transformations if you have other types of data
        annos = [
            utils.transform_instance_annotations(
                obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
            )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(
            annos, image_shape, mask_format=self.instance_mask_format
        )

        # After transforms such as cropping are applied, the bounding box may no longer
        # tightly bound the object. As an example, imagine a triangle object
        # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
        # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
        # the intersection of original bounding box and the cropping box.
        if self.recompute_boxes:
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        dataset_dict["instances"] = utils.filter_empty_instances(instances)

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )
        '''
        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        '''
        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        return dataset_dict