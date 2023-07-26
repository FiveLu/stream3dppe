import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import Linear, bias_init_with_prob

from mmcv.runner import force_fp32
from mmdet.core import (build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models.utils import build_transformer
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
import torch.nn.functional as F
from mmdet.models.utils import NormedLinear
from projects.mmdet3d_plugin.models.dense_heads.streampetr_head import StreamPETRHead
from projects.mmdet3d_plugin.models.utils.depthnet import SELayer, build_depthnet
from projects.mmdet3d_plugin.models.utils.positional_encoding import pos2posemb3d, pos2posemb1d, nerf_positional_encoding
from projects.mmdet3d_plugin.models.utils.misc import MLN, topk_gather, transform_reference_points, memory_refresh, SELayer_Linear
from mmcv.cnn import Conv2d, Linear
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding


@HEADS.register_module()
class StreamPETRHead3DPPE(StreamPETRHead):
    _version = 1

    def __init__(self,
                 # 3dppe part
                 with_depth_supervision=True,
                 depthnet=dict(
                     type='CameraAwareDepthNet',
                     in_channels=256,
                     context_channels=256,
                     depth_channels=16,
                     mid_channels=256,
                     with_depth_correction=True,
                     with_pgd=True,
                     with_context_encoder=False,
                     fix_alpha=0.41,
                 ),
                 positional_encoding=dict(
                     type='SinePositionalEncoding3D', num_feats=128, normalize=True),
                 with_filter=False,     
                 num_keep=5,       
                 with_dpe=False,    # Distribution-guide position encoder
                 use_sigmoid=True,
                 use_detach=False,
                 share_pe_encoder=False,
                 with_2dpe_only=False,
                 use_prob_depth=True,
                 loss_depth=dict(type='SmoothL1Loss',
                                 beta=1.0 / 9.0, reduction='mean', loss_weight=0.1),
                 use_dfl=True,
                 loss_dfl=dict(type='DistributionFocalLoss',
                               reduction='mean', loss_weight=0.25),
                 with_pos_info=False,
                 with_position=True,
                 with_multiview=True,
                 init_query=None,
                 **kwargs):
        self.with_dpe = with_dpe
        self.with_depth_supervision = with_depth_supervision
        self.with_filter = with_filter
        self.use_sigmoid = use_sigmoid
        self.use_detach = use_detach  # detach depth score
        self.share_pe_encoder = share_pe_encoder
        self.with_2dpe_only = with_2dpe_only
        self.with_pos_info = with_pos_info
        self.use_dfl = use_dfl
        self.with_position = with_position
        self.with_multiview = with_multiview
        self.init_query = init_query
        
        if self.with_depth_supervision:
            kwargs['in_channels'] = depthnet['context_channels']
            if with_filter:
                kwargs['depth_num'] = num_keep
                self.num_keep = num_keep

        super(StreamPETRHead3DPPE, self).__init__(**kwargs)

        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        if self.with_depth_supervision:
            self.depth_net = build_depthnet(depthnet)
            self.depth_num = self.depth_net.depth_channels

            self.loss_depth = build_loss(loss_depth)
            self.with_pgd = getattr(self.depth_net, 'with_pgd', False)
            if self.use_dfl:
                self.loss_dfl = build_loss(loss_dfl)

            self.use_prob_depth = use_prob_depth
            if self.use_prob_depth:
                index = torch.arange(
                    start=0, end=self.depth_num, step=1).float()  # (D, )
                bin_size = (
                    self.position_range[3] - self.depth_start) / (self.depth_num - 1)
                depth_bin = self.depth_start + bin_size * index  # (D, )
                self.register_buffer('project', depth_bin)  # (D, )
            if not self.use_prob_depth:
                assert self.depth_num == 1, 'depth_num setting is wrong'
                assert self.with_pgd is False, 'direct depth prediction cannot be combined with pgd'
                assert self.use_dfl is False, 'direct depth prediction cannot be combined with dfl'

        if self.with_dpe:
            self.dpe = SELayer(self.depth_num, self.embed_dims)

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        if self.with_position:
            self.input_proj = nn.Sequential(
                Conv2d(self.in_channels, self.embed_dims, kernel_size=1),
                nn.ReLU(),
                Conv2d(self.in_channels, self.embed_dims, kernel_size=1)
            )
            # self.input_proj = Conv2d(
            #     self.in_channels, self.embed_dims, kernel_size=1)
        else:
            self.input_proj = nn.Sequential(
                Conv2d(self.in_channels, self.embed_dims, kernel_size=1),
                nn.ReLU(),
                Conv2d(self.in_channels, self.embed_dims, kernel_size=1)
            )
            # self.input_proj = Conv2d(
            #     self.in_channels, self.embed_dims, kernel_size=1)

        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        if self.normedlinear:
            cls_branch.append(NormedLinear(
                self.embed_dims, self.cls_out_channels))
        else:
            cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        self.cls_branches = nn.ModuleList(
            [fc_cls for _ in range(self.num_pred)])
        self.reg_branches = nn.ModuleList(
            [reg_branch for _ in range(self.num_pred)])

        # self.position_encoder = nn.Sequential(
        #     nn.Linear(self.position_dim, self.embed_dims*4),
        #     nn.ReLU(),
        #     nn.Linear(self.embed_dims*4, self.embed_dims),
        # )

        if self.share_pe_encoder:
            position_encoder = nn.Sequential(
                nn.Linear(self.embed_dims*3//2, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.embed_dims),
            )
            if self.with_position:
                self.position_encoder = position_encoder
            self.query_embedding = position_encoder
        else:
            if self.with_position:
                # self.position_dim = 3 * self.depth_num      # D*3 3:(x, y, z)
                self.position_encoder = nn.Sequential(
                    nn.Linear(self.embed_dims*3//2, self.embed_dims),
                    nn.ReLU(),
                    nn.Linear(self.embed_dims, self.embed_dims),
                )
            self.query_embedding = nn.Sequential(
                nn.Linear(self.embed_dims*3//2, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.embed_dims),
            )
        if self.init_query is not None:
            ref_points = torch.from_numpy(np.load(self.init_query)) 
        else:
            ref_points = None
        self.reference_points = nn.Embedding(self.num_query, 3, _weight=ref_points)
        if self.num_propagated > 0:
            self.pseudo_reference_points = nn.Embedding(self.num_propagated, 3)

        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims*3//2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims)
        )

        if self.with_ego_pos:
            self.ego_pose_pe = MLN(180)
            self.ego_pose_memory = MLN(180)

        if self.with_pos_info:
            self.extra_position_encoder = nn.Sequential(
                nn.Linear(3, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(inplace=True),
            )
        if self.with_multiview:
            self.adapt_pos3d = nn.Sequential(
                nn.Conv2d(self.embed_dims*3//2, self.embed_dims *
                          4, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims*4, self.embed_dims,
                          kernel_size=1, stride=1, padding=0),
            )
        if self.with_2dpe_only:
            self.adapt_pos3d = nn.Sequential(
                nn.Conv2d(self.embed_dims, self.embed_dims,
                          kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims, self.embed_dims,
                          kernel_size=1, stride=1, padding=0),
            )

    def integral(self, depth_pred):
        """
        Args:
            depth_pred: (N, D)
        Returns:
            depth_val: (N, )
        """
        depth_score = F.softmax(depth_pred, dim=-1)     # (N, D)
        depth_val = F.linear(depth_score, self.project.type_as(
            depth_score))  # (N, D) * (D, )  --> (N, )
        return depth_val

    def forward(self, memory_center, img_metas, topk_indexes=None,  **data):
        # zero init the memory bank
        self.pre_update_memory(data)

        x = data['img_feats']
        B, N, C, H, W = x.shape
        x = x.flatten(0, 1)
        # depth_supervision
        if self.with_depth_supervision:
            intrinsics = data['intrinsics'][..., :3, :3].contiguous()
            extrinsics = data['extrinsics'].contiguous()
            if self.with_pgd:
                depth, x, depth_direct = self.depth_net(
                    x, intrinsics, extrinsics)
            else:
                # (B * N, D/1, H, W),  (B*N, C, H, W)
                depth, x = self.depth_net(x, intrinsics, extrinsics)
            if self.use_prob_depth:
                self.depth_score = depth  
                # (B*N_view, H, W, D) --> (B*N_view*H*W, D)
                depth_prob = depth.permute(
                    0, 2, 3, 1).contiguous().view(-1, self.depth_num)
                # softmax to prob  , then integral
                depth_prob_val = self.integral(depth_prob)
                depth_map_pred = depth_prob_val

                if self.with_pgd:
                    if self.depth_net.fix_alpha is not None:
                        sig_alpha = self.depth_net.fuse_lambda
                    else:
                        sig_alpha = torch.sigmoid(self.depth_net.fuse_lambda)
                    depth_direct_val = depth_direct.view(-1)      # (B*N*H*W, )
                    depth_pgd_fuse = sig_alpha * depth_direct_val + \
                        (1 - sig_alpha) * depth_prob_val
                    depth_map_pred = depth_pgd_fuse
            else:
                # direct depth
                depth_map_pred = depth.exp().view(-1)     # (B*N*H*W, )
        else:
            depth_map_pred = data['depth_map']  # for gt depth test(Upper limit of 3dppe) 

        x = self.input_proj(x)
        x = x.view(B, N, C, H, W)

        if self.with_position:
            # 3D PE: (B, N_view, embed_dims, H, W)
            if self.use_detach:
                depth_map = depth_map_pred.detach()
            else:
                depth_map = depth_map_pred
            depth_map = depth_map.view(B, N, H, W)

            # for vis
            if False:
                import cv2
                mean = np.array([123.675, 116.28, 103.53]).reshape(3, 1, 1)
                std = np.array([58.395, 57.12, 57.375]).reshape(3, 1, 1)
                for imgid in range(N):
                    cur_depth_map = depth_map[0][imgid]
                    cur_depth_map = cur_depth_map.detach().cpu().numpy()
                    cur_depth_map = cur_depth_map*255/60
                    cur_depth_map = cur_depth_map.astype(np.uint8)
                    depth_color_map = cv2.applyColorMap(
                        cur_depth_map, cv2.COLORMAP_RAINBOW)

                    ori_img = data['img'][0][imgid].cpu().numpy()
                    img_for_show = ((ori_img*std+mean))

                    img_for_show = np.transpose(
                        img_for_show, (1, 2, 0)).astype(np.uint8)[..., ::-1]
                    depth_color_map = cv2.resize(
                        depth_color_map, (img_for_show.shape[1], img_for_show.shape[0]), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(
                        f"./debug_save/ori_img_{imgid}.png", img_for_show)
                    cv2.imwrite(
                        f"./debug_save/depth_map_{imgid}.png", depth_color_map)

                    img_with_color = img_for_show*0.6+0.4*depth_color_map
                    cv2.imwrite(
                        f"./debug_save/depth_map_withimg_{imgid}.png", img_with_color)

            masks = x.new_zeros((B, N, H, W))
            coords_position_embeding = self.position_embeding_3dppe(
                data, x, img_metas, depth_map)
            pos_embed = coords_position_embeding
            if self.with_multiview:
                # (B, N_view, num_feats*3=embed_dims*3/2, H, W)
                sin_embed = self.positional_encoding(masks)
                # (B, N_view, num_feats*3=embed_dims*3/2, H, W) --> (B*N_view, num_feats*3=embed_dims*3/2, H, W)
                # --> (B*N_view, embed_dims, H, W) --> (B, N_view, embed_dims, H, W)
                sin_embed = self.adapt_pos3d(
                    sin_embed.flatten(0, 1)).view(x.size())
                # (B, N_view, embed_dims, H, W)
                pos_embed = pos_embed + sin_embed
            elif self.with_2dpe_only:
                pos_embeds = []
                for i in range(N):
                    xy_embed = self.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(xy_embed.unsqueeze(1))
                sin_embed = torch.cat(pos_embeds, 1)
                sin_embed = self.adapt_pos3d(
                    sin_embed.flatten(0, 1)).view(x.size())
                pos_embed = pos_embed + sin_embed
            else:
                pos_embed = pos_embed
        else:
            if self.with_multiview:
                pos_embed = self.positional_encoding(masks)
                pos_embed = self.adapt_pos3d(
                    pos_embed.flatten(0, 1)).view(x.size())
            elif self.with_2dpe_only:
                pos_embeds = []
                for i in range(N):
                    pos_embed = self.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(pos_embed.unsqueeze(1))
                pos_embed = torch.cat(pos_embeds, 1)
            else:
                pos_embed = x.new_zeros(x.size())

        num_tokens = N * H * W
        memory = x.permute(0, 1, 3, 4, 2).reshape(B, num_tokens, C)
        memory = topk_gather(memory, topk_indexes)

        pos_embed = pos_embed.permute(0, 1, 3, 4, 2).reshape(B, num_tokens, C)
        pos_embed = topk_gather(pos_embed, topk_indexes)

        # first 256 tokens ->  memory   , last 644 query -> current
        reference_points = self.reference_points.weight
        reference_points, attn_mask, mask_dict = self.prepare_for_dn(
            B, reference_points, img_metas)
        # dim = 128*3 384 -> query_embed 256
        query_pos = self.query_embedding(
            pos2posemb3d(inverse_sigmoid(reference_points)))
        tgt = torch.zeros_like(query_pos)

        # prepare for the tgt and query_pos using mln.
        tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose = self.temporal_alignment(
            query_pos, tgt, reference_points)

        outs_dec, _ = self.transformer(
            memory, tgt, query_pos, pos_embed, attn_mask, temp_memory, temp_pos)

        outs_dec = torch.nan_to_num(outs_dec)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(reference_points.clone())
            assert reference.shape[-1] == 3
            outputs_class = self.cls_branches[lvl](outs_dec[lvl])
            tmp = self.reg_branches[lvl](outs_dec[lvl])

            tmp[..., 0:3] += reference[..., 0:3]
            tmp[..., 0:3] = tmp[..., 0:3].sigmoid()

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)
        all_bbox_preds[..., 0:3] = (
            all_bbox_preds[..., 0:3] * (self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3])

        # update the memory bank
        self.post_update_memory(
            data, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec, mask_dict)

        if mask_dict and mask_dict['pad_size'] > 0:
            output_known_class = all_cls_scores[:,
                                                :, :mask_dict['pad_size'], :]
            output_known_coord = all_bbox_preds[:,
                                                :, :mask_dict['pad_size'], :]
            outputs_class = all_cls_scores[:, :, mask_dict['pad_size']:, :]
            outputs_coord = all_bbox_preds[:, :, mask_dict['pad_size']:, :]
            mask_dict['output_known_lbs_bboxes'] = (
                output_known_class, output_known_coord)
            outs = {
                'all_cls_scores': outputs_class,
                'all_bbox_preds': outputs_coord,
                'dn_mask_dict': mask_dict,

            }
        else:
            outs = {
                'all_cls_scores': all_cls_scores,
                'all_bbox_preds': all_bbox_preds,
                'dn_mask_dict': None,
            }

        if self.with_depth_supervision and self.depth_net is not None:
            outs['depth_map_pred'] = depth_map_pred    # (B*N_view*H*W, )
            if self.training:
                outs['depth_map'] = data['depth_map']
                outs['depth_map_mask'] = data['depth_map_mask']
            if self.use_prob_depth:
                outs['depth_prob'] = depth_prob     # (B*N_view*H*W, D)
                if self.with_pgd:
                    outs['depth_prob_val'] = depth_prob_val  # (B*N_view*H*W, )
                    # (B*N_view*H*W, )
                    outs['depth_direct_val'] = depth_direct_val

        return outs

    def position_embeding_3dppe(self, data, img_feats, img_metas, depth_map=None):
        eps = 1e-5
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        B, N, C, H, W = img_feats.shape

        # Map to the original scale to get the corresponding pixel coordinates.
        coords_h = torch.arange(
            H, device=img_feats[0].device).float() * pad_h / H      # (H, )
        coords_w = torch.arange(
            W, device=img_feats[0].device).float() * pad_w / W      # (W, )

        # (2, W, H)  --> (W, H, 2)    2: (u, v)
        coords = torch.stack(torch.meshgrid(
            [coords_w, coords_h])).permute(1, 2, 0).contiguous()
        coords = coords.view(1, 1, W, H, 2).repeat(
            B, N, 1, 1, 1)       # (B, N_view, W, H, 2)

        depth_map = depth_map.permute(
            0, 1, 3, 2).contiguous()      # (B, N_view, W, H)

        depth_map = depth_map.unsqueeze(dim=-1)     # (B, N_view, W, H, 1)
        # (B, N_view, W, H, 2)    (du, dv)
        coords = coords * \
            torch.maximum(depth_map, torch.ones_like(depth_map) * eps)
        # (B, N_view, W, H, 3)   (du, dv, d)
        coords = torch.cat([coords, depth_map], dim=-1)
        # (B, N_view, W, H, 4)   (du, dv, d, 1)
        coords = torch.cat([coords, torch.ones_like(coords[..., :1])], dim=-1)

        lidar2imgs = data['lidar2img']
        img2lidars = lidar2imgs.inverse()

        coords = coords.unsqueeze(dim=-1)       # (B, N_view, W, H, 4, 1)
        # (B, N_view, 1, 1, 4, 4) --> (B, N_view, W, H, 4, 4)
        img2lidars = img2lidars.view(B, N, 1, 1, 4, 4).repeat(1, 1, W, H, 1, 1)

        # The frustum points corresponding to each pixel in the image are projected into the lidar system with img2lidars..
        # (B, N_view, W, H, D, 4, 4) @ (B, N_view, W, H, D, 4, 1) --> (B, N_view, W, H, D, 4, 1)
        # --> (B, N_view, W, H, D, 3)   3: (x, y, z)
        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        # With the help of position range, the 3D coordinates are normalized.
        coords3d[..., 0:3] = (coords3d[..., 0:3] - self.position_range[0:3]) / (
            self.position_range[3:6] - self.position_range[0:3])  # norm 0~1

        # (B, N_view, W, H, D, 3) --> (B, N_view, D, 3, H, W) --> (B*N_view, D*3, H, W)
        coords3d = coords3d.permute(0, 1, 3, 2, 4).contiguous().view(
            B*N, H, W, 3)      # (B*N_view, H, W, 3)
        coords3d = inverse_sigmoid(coords3d)    # (B*N_view, H, W, 3)
        # 3D position embedding(PE)
        coords_position_embeding = self.position_encoder(
            pos2posemb3d(coords3d))  # (B*N_view, H, W, embed_dims)
        coords_position_embeding = coords_position_embeding.permute(
            0, 3, 1, 2).contiguous()    # (B*N_view, embed_dims, H, W)

        return coords_position_embeding.view(B, N, self.embed_dims, H, W)

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None):
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()

        # loss_dict['size_loss'] = size_loss
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1

        if preds_dicts['dn_mask_dict'] is not None:
            known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt = self.prepare_for_loss(
                preds_dicts['dn_mask_dict'])
            all_known_bboxs_list = [known_bboxs for _ in range(num_dec_layers)]
            all_known_labels_list = [
                known_labels for _ in range(num_dec_layers)]
            all_num_tgts_list = [
                num_tgt for _ in range(num_dec_layers)
            ]

            dn_losses_cls, dn_losses_bbox = multi_apply(
                self.dn_loss_single, output_known_class, output_known_coord,
                all_known_bboxs_list, all_known_labels_list,
                all_num_tgts_list)
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i in zip(dn_losses_cls[:-1],
                                               dn_losses_bbox[:-1]):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                num_dec_layer += 1
        elif self.with_dn:
            dn_losses_cls, dn_losses_bbox = multi_apply(
                self.loss_single, all_cls_scores, all_bbox_preds,
                all_gt_bboxes_list, all_gt_labels_list,
                all_gt_bboxes_ignore_list)
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1].detach()
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1].detach()
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i in zip(dn_losses_cls[:-1],
                                               dn_losses_bbox[:-1]):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i.detach()
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i.detach()
                num_dec_layer += 1

        if self.with_depth_supervision:
            depth_map_pred = preds_dicts['depth_map_pred']  # (B*N_view*H*W, )
            depth_prob = preds_dicts.get(
                'depth_prob', None)    # (B*N_view*H*W, D)
            depth_map = preds_dicts['depth_map']
            depth_map_mask = preds_dicts['depth_map_mask']
            depth_loss_dict = self.depth_loss(
                depth_map_pred, depth_map, depth_map_mask, depth_prob)
            loss_dict.update(depth_loss_dict)

        return loss_dict

    def mask_points_by_dist(self, depth_map, depth_map_mask, min_dist, max_dist):
        mask = depth_map.new_ones(depth_map.shape, dtype=torch.bool)
        mask = torch.logical_and(mask, depth_map >= min_dist)
        mask = torch.logical_and(mask, depth_map < max_dist)
        depth_map_mask[~mask] = 0
        depth_map[~mask] = 0
        return depth_map, depth_map_mask

    def depth_loss(self, depth_map_pred, depth_map_tgt, depth_map_mask, depth_prob=None):
        depth_map_tgt = depth_map_tgt.view(-1)
        depth_map_mask = depth_map_mask.view(-1)
        depth_map_pred = depth_map_pred.view(-1)

        min_dist = self.depth_start
        depth_map_tgt, depth_map_mask = self.mask_points_by_dist(depth_map_tgt, depth_map_mask,
                                                                 min_dist=min_dist,
                                                                 max_dist=self.position_range[3])

        valid = depth_map_mask > 0
        valid_depth_pred = depth_map_pred[valid]      # (N_valid, )
        loss_dict = {}
        loss_depth = self.loss_depth(pred=valid_depth_pred, target=depth_map_tgt[valid],
                                     avg_factor=max(depth_map_mask.sum().float(), 1.0))

        loss_dict['loss_depth'] = loss_depth

        if self.use_dfl and depth_prob is not None:
            bin_size = (self.position_range[3] -
                        min_dist) / (self.depth_num - 1)
            depth_label_clip = (depth_map_tgt - self.depth_start) / bin_size
            depth_map_clip, depth_map_mask = self.mask_points_by_dist(depth_label_clip, depth_map_mask, 0,
                                                                      self.depth_num - 1)      # (B*N_view*H*W, )
            valid = depth_map_mask > 0      # (B*N_view*H*W, )
            valid_depth_prob = depth_prob[valid]    # (N_valid, )
            loss_dfl = self.loss_dfl(pred=valid_depth_prob, target=depth_map_clip[valid],
                                     avg_factor=max(depth_map_mask.sum().float(), 1.0))
            # DFL Args:
            # pred (torch.Tensor): Predicted general distribution of bounding boxes
            #     (before softmax) with shape (N, n+1), n is the max value of the
            #     integral set `{0, ..., n}` in paper.
            # label (torch.Tensor): Target distance label for bounding boxes with
            #     shape (N,).
            loss_dict['loss_dfl'] = loss_dfl

        return loss_dict

    def temporal_alignment(self, query_pos, tgt, reference_points):
        B = query_pos.size(0)

        temp_reference_point = (self.memory_reference_point -
                                self.pc_range[:3]) / (self.pc_range[3:6] - self.pc_range[0:3])
        temp_pos = self.query_embedding(pos2posemb3d(
            inverse_sigmoid(temp_reference_point)))
        temp_memory = self.memory_embedding
        rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(
            0).unsqueeze(0).repeat(B, query_pos.size(1), 1, 1)

        if self.with_ego_pos:
            rec_ego_motion = torch.cat([torch.zeros_like(
                reference_points[..., :3]), rec_ego_pose[..., :3, :].flatten(-2)], dim=-1)
            rec_ego_motion = nerf_positional_encoding(rec_ego_motion)
            tgt = self.ego_pose_memory(tgt, rec_ego_motion)
            query_pos = self.ego_pose_pe(query_pos, rec_ego_motion)
            memory_ego_motion = torch.cat(
                [self.memory_velo, self.memory_timestamp, self.memory_egopose[..., :3, :].flatten(-2)], dim=-1).float()
            memory_ego_motion = nerf_positional_encoding(memory_ego_motion)
            temp_pos = self.ego_pose_pe(temp_pos, memory_ego_motion)
            temp_memory = self.ego_pose_memory(temp_memory, memory_ego_motion)

        query_pos += self.time_embedding(pos2posemb1d(
            torch.zeros_like(reference_points[..., :1])))
        temp_pos += self.time_embedding(
            pos2posemb1d(self.memory_timestamp).float())

        # TODO:
        if self.num_propagated > 0:
            tgt = torch.cat([tgt, temp_memory[:, :self.num_propagated]], dim=1)
            query_pos = torch.cat(
                [query_pos, temp_pos[:, :self.num_propagated]], dim=1)
            reference_points = torch.cat(
                [reference_points, temp_reference_point[:, :self.num_propagated]], dim=1)
            rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(
                0).unsqueeze(0).repeat(B, query_pos.shape[1]+self.num_propagated, 1, 1)
            temp_memory = temp_memory[:, self.num_propagated:]
            temp_pos = temp_pos[:, self.num_propagated:]

        return tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose
