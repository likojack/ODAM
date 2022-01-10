"""
DETR model and criterion classes.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.utils import box_utils
from src.utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer

class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """

        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        
        # object embedding
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.offset_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.angle_embed = MLP(hidden_dim, hidden_dim, 30, 3)
        self.size_embed = MLP(hidden_dim, hidden_dim, 3, 3)
        self.depth_embed = MLP(hidden_dim, hidden_dim, 1, 3)

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs_objs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        # object outputs
        outputs_class = self.class_embed(hs_objs)
        outputs_coord = self.bbox_embed(hs_objs).sigmoid()
        outputs_angle = self.angle_embed(hs_objs)
        outputs_offset = self.offset_embed(hs_objs)
        outputs_size = self.size_embed(hs_objs)
        outputs_depth = self.depth_embed(hs_objs)

        out = {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_coord[-1],
            'pred_angle': outputs_angle[-1],
            'pred_offset': outputs_offset[-1],
            'pred_size': outputs_size[-1],
            'pred_depth': outputs_depth[-1],
            'pred_obj_features': hs_objs[-1],
        }
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord, outputs_angle, outputs_offset,
                outputs_size, outputs_depth)

        return out

    def postprocess(self, out, img_size, threshold, intr_mat, nms_2d=True):
        """ postprocess the network prediction

        argmax to get rotation, non maximum suppresion in 3D

        Args:
            threshold: the threshold to keep objects
            img_size: original image size
            intr_mat: [3, 3] intrinsic matrix
            nms_2d: whether to use NMS on 2D bounding box
        Returns:
            bbox_2d: [b, 4], bbox_aabb

        """

        img_w, img_h = img_size
        b_size = len(out['pred_logits'])
        f = torch.tensor([intr_mat[0, 0], intr_mat[1, 1]]).float().cuda()
        c = torch.tensor([intr_mat[0, 2], intr_mat[1, 2]]).float().cuda()

        bboxes_2d = []
        scores = []
        dimensions = []
        angles = []
        t_cos = []
        classes = []
        for i in range(b_size):
            n_objs = len(out['pred_boxes'][i])
            probas = out['pred_logits'].softmax(-1)[i, :, :-1].cpu()
            keep_obj = probas.max(-1).values > threshold
            score = probas.max(-1).values[keep_obj].detach().cpu().numpy()
            class_ = probas.max(-1).indices[keep_obj].cpu().numpy()
            boxes = box_utils.box_cxcywh_to_xyxy_pytorch(out['pred_boxes'][i])
            rescale_boxes = boxes * \
                torch.tensor([img_w, img_h, img_w, img_h]).float().cuda()
            rescale_boxes = rescale_boxes.reshape(n_objs, 2, 2)
            rescaled_offset = out['pred_offset'][i] * \
                torch.tensor([img_w, img_h]).float().cuda()
            shape_center = rescaled_offset + torch.mean(rescale_boxes, dim=1)
            center_3d = (shape_center - c.unsqueeze(0)) / f.unsqueeze(0)
            center_3d *= out['pred_depth'][i]
            center_3d = torch.cat([center_3d, out['pred_depth'][i]], dim=1)

            _, n_angle_bins = out['pred_angle'][i, keep_obj].shape
            angle = out['pred_angle'][i].max(-1).indices * (180/n_angle_bins)
            angle = angle[keep_obj].cpu().numpy()
            dimension = out['pred_size'][i, keep_obj].detach().cpu().numpy()
            rescale_boxes = rescale_boxes[keep_obj].detach().cpu().numpy()
            t_co = center_3d[keep_obj].detach().cpu().numpy()
            keep_objs = self.nms_3d(class_, score, t_co, dimension, rescale_boxes, nms_2d)
            dimensions.append(dimension[keep_objs])
            bboxes_2d.append(rescale_boxes[keep_objs])
            t_cos.append(t_co[keep_objs])
            classes.append(class_[keep_objs])
            scores.append(score[keep_objs])
            angles.append(angle[keep_objs])

        return {
            "bboxes": bboxes_2d,
            "dimensions": dimensions,
            "angles": angles,
            "translates": t_cos,
            "classes": classes,
            "scores": scores}

    def nms_3d(self, class_, scores, t_cos, dimensions, bbox_2d, nms_2d):
        orders = np.argsort(scores)[::-1]  # order by descending

        exist_objects = {}
        n_objs = len(class_)
        keep_objs = []
        suppressed_objs = []

        for i, source_id in enumerate(orders):  # start from the detection with highest confidence
            # skip if this object has been suppressed
            if source_id in suppressed_objs:
                continue
            else:
                keep_objs.append(source_id)

            t_co = t_cos[source_id]
            dimension = dimensions[source_id]
            bbox = np.array([
                [-dimension[0], -dimension[1], -dimension[2]],
                [dimension[0], dimension[1], dimension[2]],
            ])
            source_bbox = bbox / 2.
            source_bbox += t_co[None, :]
            # keep this object

            for target_id in orders[i+1:]:
                if target_id in suppressed_objs:
                    continue
                t_co = t_cos[target_id]
                dimension = dimensions[target_id]
                bbox = np.array([
                    [-dimension[0], -dimension[1], -dimension[2]],
                    [dimension[0], dimension[1], dimension[2]],
                ])
                target_bbox = bbox / 2.
                target_bbox += t_co[None, :]
                iou = box_utils.iou_3d(source_bbox, target_bbox)
                if class_[target_id] == class_[source_id] and iou > 0.25:
                    suppressed_objs.append(target_id)
                    continue
                if nms_2d:
                    if box_utils.iou_2d(bbox_2d[source_id], bbox_2d[target_id]) > 0.5:
                        suppressed_objs.append(target_id)
                        continue
        return keep_objs



        #     if i in suppressed_objs:
        #         continue
        #     if i in keep_objs:
        #         continue
        #     max_iou = -1
        #     if class_[i] not in exist_objects:
        #         exist_objects[class_[i]] = [[i, cur_bbox, bbox_2d[i]]]
        #         keep_objs.append(i)
        #     else:
        #         for id_, bbox in enumerate(exist_objects[class_[i]]):                
        #             obj_id, bbox_3d, bbox_2d_area = bbox
        #             iou = box_utils.iou_3d(cur_bbox, bbox_3d)
        #             if iou > max_iou:
        #                 max_iou = iou
        #             if max_iou > 0.25:  # pick the one with larger 2D bbox
        #                 if bbox_2d_area > cur_bbox_area:
        #                     pass
        #                 else:
        #                     del exist_objects[class_[i]][id_]
        #                     keep_objs.remove(obj_id)
        #                     exist_objects[class_[i]].append([i, cur_bbox, cur_bbox_area])
        #                     keep_objs.append(i)
        #                 break
        #         if max_iou < 0.25:
        #             exist_objects[class_[i]].append([i, cur_bbox, cur_bbox_area])
        #             keep_objs.append(i)
        # return keep_objs

    @torch.jit.unused
    def _set_aux_loss(
        self, outputs_class, outputs_coord, outputs_angle, outputs_offset,
        outputs_size, outputs_depth
    ):
        # TODO: include plane and other object property for aux loss
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_angle': c,
                 'pred_offset': d, 'pred_size': e, 'pred_depth': f,
                }
                for a, b, c, d, e, f in zip(
                    outputs_class[:-1], outputs_coord[:-1],
                    outputs_angle[:-1], outputs_offset[:-1],
                    outputs_size[:-1], outputs_depth[:-1],
                    )]
        # return [{'pred_logits': a, 'pred_boxes': b}
        #         for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        empty_weight_plane = torch.ones(5)
        empty_weight_plane[-1] = 0.1
        self.register_buffer('empty_weight_plane', empty_weight_plane)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["objects"][J, 0] for t, (_, J) in zip(targets, indices)]).long()
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["objects"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_pred = torch.sum(pred_logits.softmax(-1)[:, :, :-1].max(-1).values > 0.7, dim=1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """

        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['objects'][i, 1: 5] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_utils.generalized_box_iou(
            box_utils.box_cxcywh_to_xyxy_pytorch(src_boxes),
            box_utils.box_cxcywh_to_xyxy_pytorch(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_size(self, outputs, targets, indices, num_boxes):
        assert 'pred_size' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_sizes = outputs['pred_size'][idx]
        target_sizes = torch.cat([t['objects'][i, 5: 8] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {}
        loss_size = F.l1_loss(src_sizes, target_sizes, reduction='none')
        losses['loss_size'] = loss_size.sum() / num_boxes
        return losses

    def loss_depth(self, outputs, targets, indices, num_boxes):
        assert 'pred_depth' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_depths = outputs['pred_depth'][idx]
        target_depths = torch.cat([t['objects'][i, -2] for t, (_, i) in zip(targets, indices)], dim=0)
        target_depths = target_depths.unsqueeze(-1)
        losses = {}
        loss_depth = F.l1_loss(src_depths, target_depths, reduction='none')
        losses['loss_depth'] = loss_depth.sum() / num_boxes
        return losses

    def loss_angle(self, outputs, targets, indices, num_boxes):
        assert 'pred_angle' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_angle = outputs['pred_angle'][idx]
        target_angle = torch.cat([t['objects'][i, -1] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {}
        loss_ce = F.cross_entropy(src_angle, target_angle.long(), reduction='none')
        loss_ce = loss_ce.sum() / num_boxes
        losses['loss_angle'] = loss_ce
        return losses

    def loss_offset(self, outputs, targets, indices, num_boxes):
        assert 'pred_offset' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_offset = outputs['pred_offset'][idx]
        target_offset = torch.cat([t['objects'][i, 8: 10] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {}
        loss_offset = F.l1_loss(src_offset, target_offset, reduction='none')
        loss_offset = loss_offset.sum() / num_boxes
        losses['loss_offset'] = loss_offset
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'size': self.loss_size,
            'angle': self.loss_angle,
            'depth': self.loss_depth,
            'offset': self.loss_offset,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        obj_indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["objects"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, obj_indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        # TODO: set aux decoding loss at every level
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                obj_indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, obj_indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)                
        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_utils.box_cxcywh_to_xyxy_pytorch(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    num_classes = 20 if args.dataset_file != 'coco' else 91
    num_classes = 18 if args.dataset_file == "scan_net" else num_classes

    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    matcher = build_matcher(args)

    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )

    weight_dict = {
        'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef, 'loss_angle': 1,
        'loss_offset': 3, 'loss_size': 1, 'loss_depth': 1,
    }
    weight_dict['loss_giou'] = args.giou_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality', 'size', 'angle', 'offset', 'depth']
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors