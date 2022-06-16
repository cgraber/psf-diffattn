from detectron2.structures.boxes import pairwise_intersection
import torch
import torch.utils.data as torchdata
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.structures import Boxes, Instances
from detectron2.data import DatasetFromList, MapDataset, get_detection_dataset_dicts, DatasetMapper
from detectron2.data.samplers import InferenceSampler


def pairwise_giou(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    area1 = boxes1.area()
    area2 = boxes2.area()
    inter = pairwise_intersection(boxes1, boxes2)
    u = area1[:, None] + area2 - inter
    iou = torch.where(
        inter > 0,
        inter / u,
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )

    # compute area of smallest enclosing boxes
    boxes1, boxes2 = boxes1.tensor, boxes2.tensor
    width_height = torch.max(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.min(
        boxes1[:, None, :2], boxes2[:, :2]
    )
    width_height.clamp_(min=0)
    g_area = width_height.prod(dim=2)
    giou = iou - (g_area - u)/(g_area + 1e-8)
    return giou



def focal_loss(inputs, targets, gamma):
    BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    pt = torch.exp(-BCE_loss) # prevents nans when probability 0
    focal_loss = (1-pt)**gamma * BCE_loss
    return focal_loss

def mc_focal_loss(inputs, targets, gamma, ignore_index=None):
    ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=ignore_index)
    pt = torch.exp(-ce_loss)
    focal_loss = ((1-pt)**gamma * ce_loss).mean()
    return focal_loss

def bbox_ulbr2cwh(boxes: torch.Tensor):
    old_shape = boxes.shape
    boxes = boxes.reshape(-1, 4)
    cx = (boxes[:, 2] + boxes[:, 0])/2
    cy = (boxes[:, 3] + boxes[:, 1])/2
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    boxes = torch.stack([cx, cy, w, h], dim=-1)
    return boxes.reshape(old_shape)

def bbox_cwh2ulbr(boxes: torch.Tensor):
    old_shape = boxes.shape
    boxes = boxes.reshape(-1, 4)
    x0 = boxes[:, 0] - boxes[:, 2]/2
    x1 = boxes[:, 0] + boxes[:, 2]/2
    y0 = boxes[:, 1] - boxes[:, 3]/2
    y1 = boxes[:, 1] + boxes[:, 3]/2
    boxes = torch.stack([x0, y0, x1, y1], dim=-1)
    return boxes.reshape(old_shape)

def normalize_box_tensor(boxes: torch.Tensor, img_w, img_h, normalize_factor=1):
    boxes = boxes.clone()
    shape = boxes.shape
    boxes = boxes.reshape(-1, 4)
    boxes[:, 0::2] *= 2.0*normalize_factor/img_w
    boxes[:, 1::2] *= 2.0*normalize_factor/img_h
    boxes -= 1
    return boxes.reshape(shape)

def unnormalize_box_tensor(boxes: torch.Tensor, img_w, img_h, normalize_factor=1):
    boxes = boxes.clone()
    shape = boxes.shape
    boxes = boxes.reshape(-1, 4)
    boxes += 1
    boxes[:, 0::2] *= img_w/(2.0*normalize_factor)
    boxes[:, 1::2] *= img_h/(2.0*normalize_factor)
    return boxes.reshape(shape)

def normalize_box_tensor_by_stats(boxes: torch.Tensor, mean, std):
    boxes = boxes.clone()
    shape = boxes.shape
    boxes = boxes.reshape(-1, 4)
    boxes = (boxes - mean[None]) / std[None]
    return boxes.reshape(shape)

def unnormalize_box_tensor_by_stats(boxes: torch.Tensor, mean, std):
    boxes = boxes.clone()
    shape = boxes.shape
    boxes = boxes.reshape(-1, 4)
    boxes = boxes*std[None] + mean[None]
    return boxes.reshape(shape)

def normalize_depths_by_stats(depths, mean, std, depth_mask=None):
    shape = depths.shape
    depths = depths.reshape(-1)
    depths = (depths - mean) / std[None]
    if depth_mask is not None:
        depth_mask = depth_mask.reshape(-1)
        depths[~depth_mask] = 0
    return depths.reshape(shape)

def unnormalize_depths_by_stats(depths, mean, std):
    return depths*std + mean


def normalize_boxes(boxes: Boxes, img_w, img_h):
    """
    given Boxes instance, scale such that all values lie in [-1, 1]
    """
    boxes = boxes.clone()
    boxes.scale(2.0/img_w, 2.0/img_h)
    boxes.tensor = boxes.tensor - 1 
    return boxes


def unnormalize_boxes(boxes: Boxes, img_w, img_h):
    """
    given normalized Boxes instance, scale such that values lie in image plane
    """
    boxes = boxes.clone()
    boxes.tensor = boxes.tensor + 1
    boxes.scale(img_w/2.0, img_h/2.0)
    return boxes

def box2one_hot(box_tensor, size_per_dim, img_w, img_h):
    orig_shape = box_tensor.shape
    box_tensor = box_tensor.reshape(-1, 4)
    lb = torch.tensor([[0,0,0,0]], device=box_tensor.device)
    ub = torch.tensor([[img_w-1, img_h-1, img_w-1, img_h-1]], device=box_tensor.device)
    box_tensor = torch.max(torch.min(box_tensor, ub), lb)
    w_size = img_w // size_per_dim
    h_size = img_h // size_per_dim
    size_tensor = torch.tensor([w_size, h_size, w_size, h_size], device=box_tensor.device).reshape(1, 4)
    box_inds = box_tensor // size_tensor
    box_inds = box_inds.reshape(-1).long()
    result = torch.zeros(len(box_inds), size_per_dim, device=box_inds.device)
    result[range(len(box_inds)), box_inds] = 1
    final_shape = orig_shape[:-1] + torch.Size([4*size_per_dim])
    return result.reshape(final_shape)


def index_instances(instances: Instances, indices):
    if len(indices) == 0:
        return None
    else:
        return Instances.cat([instances[i] for i in indices])

def clone_instances(instances: Instances, ignore_fields=[]):
    new_inst = Instances(instances.image_size)
    for k, v in instances.get_fields().items():
        if k not in ignore_fields:
            new_inst.set(k, v)
    return new_inst

def get_batch_time_encoding(times, emb_dim):
    b, max_len = times.shape
    d_ind = torch.arange(0, int(emb_dim/2), dtype=torch.float, device=times.device)
    omega = 1 / 10000**(2*d_ind/emb_dim)
    tmp = times.unsqueeze(-1)*omega.reshape(1, 1, -1)
    time_emb = torch.stack([
        torch.sin(tmp),
        torch.cos(tmp)
    ], -1).reshape(b, max_len, emb_dim)
    return time_emb


def get_time_encoding(emb_dim, num_instances, time_offset, device):
    d_ind = torch.arange(0, int(emb_dim/2), dtype=torch.float, 
                            device=device)
    t_ind = torch.cat([
        torch.zeros(n, device=device).fill_(t+time_offset) for t,n in enumerate(num_instances)
    ])
    omega = 1 / 10000**(2*d_ind/emb_dim)
    tmp = t_ind.unsqueeze(-1)*omega.unsqueeze(0)
    time_emb = torch.stack([
        torch.sin(tmp),
        torch.cos(tmp)
    ], -1).reshape(-1, emb_dim)
    return time_emb


def get_position_encoding(boxes, emb_size):
    orig_shape = boxes.shape
    boxes = boxes.reshape(-1, 4, 1)
    d = emb_size // 4
    d_ind = torch.arange(0, int(d/2), dtype=torch.float,
                         device=boxes.device)
    omega = 1 / 10000 ** (2*d_ind/d)
    tmp = boxes * omega.reshape(1, 1, int(d/2))
    result = torch.stack([
        torch.sin(tmp),
        torch.cos(tmp),
    ], -1)
    final_shape = orig_shape[:-1] + (4*d,)
    result = result.reshape(*final_shape)
    return result


def _val_loader_from_config(cfg, dataset_name, mapper=None):
    """
    Uses the given `dataset_name` argument (instead of the names in cfg), because the
    standard practice is to evaluate each test set individually (not combining them).
    """
    if isinstance(dataset_name, str):
        dataset_name = [dataset_name]

    dataset = get_detection_dataset_dicts(
        dataset_name,
        filter_empty=False,
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(x)] for x in dataset_name
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )
    if mapper is None:
        mapper = DatasetMapper(cfg, False)
    return {"dataset": dataset, "mapper": mapper, "total_batch_size": 1, "num_workers": cfg.DATALOADER.NUM_WORKERS}

@configurable(from_config=_val_loader_from_config)
def build_detection_val_loader(dataset, *, mapper, sampler=None, total_batch_size, num_workers=0):
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    if sampler is None:
        sampler = InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torchdata.sampler.BatchSampler(sampler, total_batch_size, drop_last=False)
    data_loader = torchdata.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader

def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def matched_boxlist_iou(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Compute pairwise intersection over union (IOU) of two sets of matched
    boxes. The box order must be (xmin, ymin, xmax, ymax).
    Similar to boxlist_iou, but computes only diagonal elements of the matrix
    Args:
        boxes1: (Boxes) bounding boxes, sized [N,4].
        boxes2: (Boxes) bounding boxes, sized [N,4].
    Returns:
        Tensor: iou, sized [N].
    """
    assert len(boxes1) == len(
        boxes2
    ), "boxlists should have the same" "number of entries, got {}, {}".format(
        len(boxes1), len(boxes2)
    )
    area1 = boxes1.area()  # [N]
    area2 = boxes2.area()  # [N]
    box1, box2 = boxes1.tensor, boxes2.tensor
    lt = torch.max(box1[:, :2], box2[:, :2])  # [N,2]
    rb = torch.min(box1[:, 2:], box2[:, 2:])  # [N,2]
    wh = (rb - lt).clamp(min=0)  # [N,2]
    inter = wh[:, 0] * wh[:, 1]  # [N]
    iou = inter / (area1 + area2 - inter)  # [N]
    return iou