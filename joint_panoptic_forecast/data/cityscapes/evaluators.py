import os
import numpy as np
import torch
import logging
import pickle
import h5py
import pandas as pd
from collections import OrderedDict, defaultdict
from typing import Optional
from scipy.optimize import linear_sum_assignment
from PIL import Image

from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.structures import pairwise_iou
from detectron2.structures.boxes import matched_boxlist_iou
from detectron2.utils import comm


class SaveFeatsEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, output_dir):
        self._metadata = MetadataCatalog.get(dataset_name)
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.out_path = os.path.join(output_dir, '%s_feats.h5'%dataset_name)

    def reset(self):
        self.meta = defaultdict(list)
        self._out_h5 = h5py.File(self.out_path, 'w')

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            file_name = input["file_name"]
            info = os.path.basename(file_name).split('_')
            city, seq, frame = info[:3]
            feats = output['feats']
            proposals = output['proposals']
            name = '%s/%s/%s'%(city, seq, frame)
            self._out_h5.create_dataset(name, data=feats.cpu().numpy(), compression='gzip')
            num_instances = [len(p) for p in proposals]
            assert(sum(num_instances) == feats.size(0))
            frame_offsets = np.insert(np.cumsum(num_instances), 0, 0)
            scores = np.concatenate([p.scores.cpu().numpy() for p in proposals])
            pred_classes = np.concatenate([p.pred_classes.cpu().numpy() for p in proposals])
            pred_boxes = np.concatenate([p.pred_boxes.tensor.cpu().numpy() for p in proposals])
            self.meta['city'].append(city)
            self.meta['seq'].append(seq)
            self.meta['frame'].append(frame)
            self.meta['frame_offsets'].append(frame_offsets)
            self.meta['pred_boxes'].append(pred_boxes)
            self.meta['scores'].append(scores)
            self.meta['pred_classes'].append(pred_classes)


    def evaluate(self):
        df = pd.DataFrame(self.meta)
        out_meta_path = os.path.join(self.output_dir, '%s_meta.pkl'%self.dataset_name)
        df.to_pickle(out_meta_path)
        self._out_h5.close()
        return {}



class CityscapesForecastInstsegEvaluator(DatasetEvaluator):
    """
    This class attempts to evaluate bounding box FDE/IoU, based on matching
    the prediction to the target frame.
    """

    def __init__(self, dataset_name, output_dir: Optional[str] = None, use_depth_sorting=False,
                 viz=False):
        self._metadata = MetadataCatalog.get(dataset_name)
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self.base_output_dir = output_dir
        self.out_seg_dir = os.path.join(output_dir, 'instseg')
        self.out_viz_dir = os.path.join(output_dir, 'instseg_viz')
        self.use_depth_sorting = use_depth_sorting
        self.viz = viz
        self.dataset_name = dataset_name
        if not os.path.exists(self.out_seg_dir):
            os.makedirs(self.out_seg_dir)

        if viz and not os.path.exists(self.out_viz_dir):
            os.makedirs(self.out_viz_dir)

    def reset(self):
        self.entries = defaultdict(lambda: defaultdict(int))
        self.score_entries = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    def process(self, inputs, outputs):
        #from detectron2.utils.visualizer import Visualizer
        from ..utils import MyVisualizer as Visualizer
        from PIL import Image
        import cityscapesscripts.helpers.labels as cslabels
        for input, output in zip(inputs, outputs):
            image_id = input['image_id'].split('_')
            city, seq, frame = image_id[:3]
            inst = output['instances']
            cls = inst.pred_classes
            if 'panoptic_segmentation' in output:
                self.output_panseg = True
                seg = output['panoptic_segmentation']
                #scene_seg = torch.zeros(1024, 2048, device=inst_segs.device)
                for val in torch.unique(seg):
                    val = val.item()
                    if val > 10:
                        inst_ind = val - 11
                        inst_seg = seg == val
                        #scene_seg[inst_seg] = inst_ind+1

                        inst_cl = cls[inst_ind].item()
                        inst_cl = cslabels.trainId2label[inst_cl+11].id
                        inst_mask = (seg == val)*255
                        final_inst_ind = self.entries[(city, seq, frame)][inst_cl]
                        self.entries[(city, seq, frame)][inst_cl] += 1
                        score = 1.0
                        self.score_entries[(city, seq, frame)][inst_cl][final_inst_ind] = score
                        out_path = os.path.join(self.out_seg_dir, '%s_%s_%s_%d_%d.png'%(city, seq, frame, inst_cl, final_inst_ind))
                        im = Image.fromarray(inst_mask.cpu().numpy().astype(np.uint8))
                        im.save(out_path)
            else:
                inst_segs = inst.pred_masks
                self.output_panseg = False
                if self.use_depth_sorting:
                    depths = inst.depths
                    inst_depths, inst_order = depths.sort(descending=True)
                    scene_seg = torch.zeros(1024, 2048, device=depths.device)
                    for inst_ind in inst_order:
                        inst_seg = (inst_segs[inst_ind] >= 0.5).long()*(inst_ind+1)
                        scene_seg = (inst_seg > 0)*inst_seg + \
                            (~(inst_seg > 0))*scene_seg
                    for inst_ind in range(len(inst_segs)):
                        inst_cl = cls[inst_ind].item()
                        inst_cl = cslabels.trainId2label[inst_cl+11].id
                        inst_mask = (scene_seg == (inst_ind+1))*255
                        final_inst_ind = self.entries[(city, seq, frame)][inst_cl]
                        self.entries[(city, seq, frame)][inst_cl] += 1
                        score = 1.0
                        self.score_entries[(city, seq, frame)][inst_cl][final_inst_ind] = score
                        out_path = os.path.join(self.out_seg_dir, '%s_%s_%s_%d_%d.png'%(city, seq, frame, inst_cl, final_inst_ind))
                        im = Image.fromarray(inst_mask.cpu().numpy().astype(np.uint8))
                        im.save(out_path)

                else:
                    for inst_ind in range(len(inst_segs)):
                        inst_cl = cls[inst_ind].item()
                        inst_cl = cslabels.trainId2label[inst_cl+11].id
                        inst_mask = (inst_segs[inst_ind].float() >= 0.5)*255
                        final_inst_ind = self.entries[(city, seq, frame)][inst_cl]
                        self.entries[(city, seq, frame)][inst_cl] += 1
                        score = 1.0
                        self.score_entries[(city, seq, frame)][inst_cl][final_inst_ind] = score
                        out_path = os.path.join(self.out_seg_dir, '%s_%s_%s_%d_%d.png'%(city, seq, frame, inst_cl, final_inst_ind))
                        im = Image.fromarray(inst_mask.cpu().numpy().astype(np.uint8))
                        im.save(out_path)
            if self.viz:
                image_path = input['file_name']
                img = np.array(Image.open(image_path))
                visualizer = Visualizer(img)
                vis_output = visualizer.draw_instance_predictions(predictions=inst.to('cpu'))
                out_path = os.path.join(self.out_viz_dir, '%s_%s_%s.png'%(city, seq, frame))
                vis_output.save(out_path)

    def evaluate(self):
        import cityscapesscripts.helpers.labels as cslabels
        comm.synchronize()
        if comm.get_rank() > 0:
            raise NotImplementedError()
        for name, cl_dict in self.entries.items():
            out_txt_path = os.path.join(self.out_seg_dir, '%s_%s_%s.txt'%name)
            with open(out_txt_path, 'w') as fout:
                for cl, count in cl_dict.items():
                    for i in range(count):
                        score = self.score_entries[name][cl][i]
                        tmp_name = '%s_%s_%s_%d_%d.png'%(*name, cl, i)
                        fout.write('%s %d %f\n'%(tmp_name, cl, score))
        # write empty files for any sequences that were excluded
        if 'test' in self.dataset_name:
            split = 'test'
        else:
            split = 'val'
        base_cityscapes_dir = os.path.join(os.environ["CITYSCAPES_DATASET"], 'leftImg8bit', split)
        missing_count = 0
        for city in os.listdir(base_cityscapes_dir):
            city_path = os.path.join(base_cityscapes_dir, city)
            for fname in os.listdir(city_path):
                name = tuple(fname.split('_')[:3])
                if name not in self.score_entries:
                    missing_count += 1
                    out_txt_path = os.path.join(self.out_seg_dir, '%s_%s_%s.txt'%name)
                    with open(out_txt_path, 'w') as fout:
                        pass
        print("MISSING COUNT: ",missing_count)
        import subprocess
        os.environ["CITYSCAPES_EXPORT_DIR"] = self.base_output_dir
        os.environ["CITYSCAPES_RESULTS"] = self.out_seg_dir
        subprocess.call(["python", "-m", "cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling"])
        orig_result_file = os.path.join(os.environ['CITYSCAPES_DATASET'], 'evaluationResults', 'resultInstanceLevelSemanticLabeling.json')
        subprocess.call(["mv", orig_result_file, self.base_output_dir])
        return {}


class CityscapesForecastSemsegEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, output_dir: Optional[str] = None, use_depth_sorting=False,
                 viz=False, bg_dir=None, split=None):
        self._metadata = MetadataCatalog.get(dataset_name)
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self.base_output_dir = output_dir
        self.out_seg_dir = os.path.join(output_dir, 'semseg')
        self.out_viz_dir = os.path.join(output_dir, 'semseg_viz')
        self.use_depth_sorting = use_depth_sorting
        self.viz = viz
        self.bg_dir = bg_dir
        self.split = split
        if not os.path.exists(self.out_seg_dir):
            os.makedirs(self.out_seg_dir)

        if viz and not os.path.exists(self.out_viz_dir):
            os.makedirs(self.out_viz_dir)

    def reset(self):
        self.entries = defaultdict(lambda: defaultdict(int))
        self.score_entries = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.final_annotations = []

    def process(self, inputs, outputs):
        #from detectron2.utils.visualizer import Visualizer
        from ..utils import MyVisualizer as Visualiser
        from PIL import Image
        import cityscapesscripts.helpers.labels as cslabels
        for input, output in zip(inputs, outputs):
            image_id = input['image_id'].split('_')
            city, seq, frame = image_id[:3]
            if 'panoptic_segmentation' in output:
                self.output_panseg = True
                seg = output['panoptic_segmentation']
                inst = output['instances']
                scene_seg = torch.zeros_like(seg)
                for val in torch.unique(seg):
                    val = val.item()
                    if val <= 10:
                        final_val = val
                    else:
                        inst_ind = val - 11
                        final_val = inst.pred_classes[inst_ind].item() + 11
                    new_cl = cslabels.trainId2label[final_val].id
                    scene_seg[seg == val] = new_cl
            else:
                self.output_panseg = False
                raise NotImplementedError()
            out_path = os.path.join(self.out_seg_dir, '%s_%s_%s_gtFine_labelIds.png'%(city, seq, frame))
            im = Image.fromarray(scene_seg.cpu().numpy().astype(np.uint8))
            im.save(out_path)

    def evaluate(self):
        import cityscapesscripts.helpers.labels as cslabels
        comm.synchronize()
        if comm.get_rank() > 0:
            raise NotImplementedError()
        
        # write empty files for any sequences that were excluded

        base_cityscapes_dir = os.path.join(os.environ["CITYSCAPES_DATASET"], 'leftImg8bit', self.split)
        if not self.output_panseg:
            raise NotImplementedError()

        import subprocess
        cityscapes_dir = os.environ['CITYSCAPES_DATASET']
        os.environ["CITYSCAPES_EXPORT_DIR"] = self.base_output_dir
        os.environ["CITYSCAPES_RESULTS"] = self.out_seg_dir
        subprocess.call([
            "python", "-m", "cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling",
            ])
        return {}

class CityscapesForecastPansegEvaluator(DatasetEvaluator):
    """
    This class attempts to evaluate bounding box FDE/IoU, based on matching
    the prediction to the target frame.
    """

    def __init__(self, dataset_name, output_dir: Optional[str] = None, use_depth_sorting=False,
                 viz=False, bg_dir=None, split=None):
        self._metadata = MetadataCatalog.get(dataset_name)
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self.base_output_dir = output_dir
        self.out_seg_dir = os.path.join(output_dir, 'panseg')
        self.out_viz_dir = os.path.join(output_dir, 'panseg_viz')
        self.use_depth_sorting = use_depth_sorting
        self.viz = viz
        self.bg_dir = bg_dir
        self.split = split
        if not os.path.exists(self.out_seg_dir):
            os.makedirs(self.out_seg_dir)

        if viz and not os.path.exists(self.out_viz_dir):
            os.makedirs(self.out_viz_dir)

    def reset(self):
        self.entries = defaultdict(lambda: defaultdict(int))
        self.score_entries = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.final_annotations = []

    def create_pan_img(self, seg):
        seg = seg.cpu().numpy()
        pan_img = np.zeros(
            (seg.shape[0], seg.shape[1], 3), dtype=np.uint8
        )
        for segmentId in np.unique(seg):
            mask = seg == segmentId
            color = [segmentId % 256, segmentId // 256, segmentId // 256 // 256]
            pan_img[mask] = color
        pan_img = Image.fromarray(pan_img)
        return pan_img

    def get_segments_info(self, seg, convert_to_train=False):
        import cityscapesscripts.helpers.labels as cslabels
        segments_info = []
        seg = seg.cpu().numpy()
        seg_vals = np.unique(seg)
        for seg_val in seg_vals:
            if seg_val == 0:
                continue
            elif seg_val > 100:
                category_id = int(seg_val / 1000)
            else:
                category_id = seg_val
            if convert_to_train:
                category_id = cslabels.labels[category_id].trainId
            segments_info.append({
                "category_id": int(category_id),
                "id": int(seg_val),
                "isthing": int(seg_val > 100),
            })
        return segments_info

    def process(self, inputs, outputs):
        #from detectron2.utils.visualizer import Visualizer
        from ..utils import MyVisualizer as Visualizer
        from PIL import Image
        import cityscapesscripts.helpers.labels as cslabels
        for input, output in zip(inputs, outputs):
            image_id = input['image_id'].split('_')
            city, seq, frame = image_id[:3]
            if 'panoptic_segmentation' in output:
                self.output_panseg = True
                # here, we directly predict the panoptic segmentation. need to convert
                inst = output['instances']
                seg = output['panoptic_segmentation']
                scene_seg = torch.clone(seg)
                for val in torch.unique(scene_seg):
                    val = val.item()
                    if val <= 10: #static
                        new_val = cslabels.trainId2label[val].id
                        scene_seg[seg == val] = new_val
                    else: #dynamic instance
                        inst_ind = val - 11
                        inst_cl = inst.pred_classes[inst_ind].item()
                        inst_cl = cslabels.trainId2label[inst_cl+11].id
                        final_inst_id = self.entries[(city, seq, frame)][inst_cl]
                        self.entries[(city, seq, frame)][inst_cl] += 1
                        seg_val = inst_cl*1000 + final_inst_id
                        scene_seg[seg == val] = seg_val
            else:
                self.output_panseg = False
                inst = output['instances']
                inst_segs = inst.pred_masks
                cls = inst.pred_classes
                if self.use_depth_sorting:
                    depths = inst.depths
                    inst_depths, inst_order = depths.sort(descending=True)
                else:
                    inst_order = range(len(inst))
                if self.bg_dir is not None:
                    bg_path = os.path.join(self.bg_dir, city, '%s_%s_%s_gtFine_labelIds.png'%(city, seq, frame))
                    bg = Image.open(bg_path)
                    scene_seg = torch.from_numpy(
                        np.array(bg, dtype=np.int32)
                    ).long()
                    final_scene_seg = torch.zeros_like(scene_seg).fill_(-1)
                    for val in torch.unique(scene_seg):
                        val = val.item()
                        new_val = cslabels.trainId2label[val].id
                        final_scene_seg[scene_seg == val] = new_val
                    scene_seg = final_scene_seg.to(inst_segs.device)
                else:
                    scene_seg = torch.zeros(1024, 2048, device=depths.device).fill_(-1)
                for inst_ind in inst_order:
                    inst_cl = cls[inst_ind].item()
                    inst_cl = cslabels.trainId2label[inst_cl+11].id
                    final_inst_id = self.entries[(city, seq, frame)][inst_cl]
                    self.entries[(city, seq, frame)][inst_cl] += 1
                    seg_val = inst_cl*1000 + final_inst_id
                    inst_seg = (inst_segs[inst_ind] >= 0.5).long()*seg_val
                    scene_seg = (inst_seg > 0)*inst_seg + \
                        (~(inst_seg > 0))*scene_seg
            pan_img = self.create_pan_img(scene_seg)
            segments_info = self.get_segments_info(scene_seg)
            new_annotations = {
                "file_name": '%s_%s_%s_pred_panoptic.png'%(city, seq, frame),
                "image_id": "%s_%s_%s"%(city, seq, frame),
                "segments_info": segments_info,
            }
            self.final_annotations.append(new_annotations)
            out_path = os.path.join(self.out_seg_dir, '%s_%s_%s_pred_panoptic.png'%(city, seq, frame))
            pan_img.save(out_path)
            

            if self.viz:
                image_path = input['file_name']
                img = np.array(Image.open(image_path))
                visualizer = Visualizer(img, metadata=MetadataCatalog.get('cityscapes_fine_panoptic_val'))
                train_segments_info = self.get_segments_info(scene_seg, convert_to_train=True)
                vis_output = visualizer.draw_panoptic_seg(scene_seg.cpu(), train_segments_info)
                out_path = os.path.join(self.out_viz_dir, '%s_%s_%s.png'%(city, seq, frame))
                vis_output.save(out_path)

    def evaluate(self):
        import cityscapesscripts.helpers.labels as cslabels
        comm.synchronize()
        if comm.get_rank() > 0:
            raise NotImplementedError()
        
        # write empty files for any sequences that were excluded

        base_cityscapes_dir = os.path.join(os.environ["CITYSCAPES_DATASET"], 'leftImg8bit', self.split)
        if not self.output_panseg:
            for city in os.listdir(base_cityscapes_dir):
                city_path = os.path.join(base_cityscapes_dir, city)
                for fname in os.listdir(city_path):
                    name = tuple(fname.split('_')[:3])
                    if name not in self.entries:
                        city, seq, frame = name
                        bg_path = os.path.join(self.bg_dir, city, '%s_%s_%s_gtFine_labelIds.png'%(city, seq, frame))
                        bg = Image.open(bg_path)
                        scene_seg = torch.from_numpy(
                            np.array(bg, dtype=np.int32)
                        ).long()
                        final_scene_seg = torch.zeros_like(scene_seg).fill_(-1)
                        for val in torch.unique(scene_seg):
                            val = val.item()
                            new_val = cslabels.trainId2label[val].id
                            final_scene_seg[scene_seg == val] = new_val
                        scene_seg = final_scene_seg
                        pan_img = self.create_pan_img(scene_seg)
                        out_path = os.path.join(self.out_seg_dir, '%s_%s_%s_pred_panoptic.png'%(city, seq, frame))
                        pan_img.save(out_path)
                        segments_info = self.get_segments_info(scene_seg)
                        new_annotations = {
                            'file_name': '%s_%s_%s_pred_panoptic.png' % (city, seq, frame),
                            'image_id': '%s_%s_%s'%(city, seq, frame),
                            'segments_info': segments_info,
                        }
                        self.final_annotations.append(new_annotations)

        final_annotations = {'annotations': self.final_annotations}
        annotation_file = os.path.join(self.base_output_dir, 'panseg.json')
        import json
        with open(annotation_file, 'w', encoding='utf-8') as f:
            json.dump(final_annotations, f, ensure_ascii=False, indent=4)

        import subprocess
        cityscapes_dir = os.environ['CITYSCAPES_DATASET']
        subprocess.call([
            "python", "-m", "cityscapesscripts.evaluation.evalPanopticSemanticLabeling",
            "--gt-json-file", os.path.join(cityscapes_dir, 'gtFine', 'cityscapes_panoptic_%s.json'%self.split),
            "--gt-folder", os.path.join(cityscapes_dir, 'gtFine', 'cityscapes_panoptic_%s'%self.split),
            "--prediction-json-file", annotation_file,
            "--prediction-folder", self.out_seg_dir,
            "--results_file", os.path.join(self.base_output_dir, 'resultPanopticSemanticLabeling.json')
            ])
        return {}