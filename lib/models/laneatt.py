import timm
import math
import torch.nn.functional as F
import cv2
import torch
import numpy as np
import torch.nn as nn
from torchvision.models import resnet18, resnet34
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from nms import nms
from lib.lane import Lane
from lib.focal_loss import FocalLoss
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from .resnet import resnet122 as resnet122_cifar
from .matching import match_proposals_with_targets
from .conv import CSPStage
from .mixer import MlpMixer
import torch
from bottleneck_transformer_pytorch import BottleStack

def line_iou(pred, target, img_w, length=15, aligned=True):
    '''
    Calculate the line iou value between predictions and targets
    Args:
        pred: lane predictions, shape: (num_pred, 72)
        target: ground truth, shape: (num_target, 72)
        img_w: image width
        length: extended radius
        aligned: True for iou loss calculation, False for pair-wise ious in assign
    '''
    px1 = pred - length
    px2 = pred + length
    tx1 = target - length
    tx2 = target + length
    if aligned:
        invalid_mask = target
        ovr = torch.min(px2, tx2) - torch.max(px1, tx1)
        union = torch.max(px2, tx2) - torch.min(px1, tx1)
    else:
        num_pred = pred.shape[0]
        invalid_mask = target.repeat(num_pred, 1, 1)
        ovr = (torch.min(px2[:, None, :], tx2[None, ...]) -
               torch.max(px1[:, None, :], tx1[None, ...]))
        union = (torch.max(px2[:, None, :], tx2[None, ...]) -
                 torch.min(px1[:, None, :], tx1[None, ...]))

    invalid_masks = (invalid_mask < 0) | (invalid_mask >= img_w)
    ovr[invalid_masks] = 0.
    union[invalid_masks] = 0.
    iou = ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)
    return iou


def liou_loss(pred, target, img_w, length=15):
    return (1 - line_iou(pred, target, img_w, length)).mean()





class EdgeDetection(nn.Module):
    def __init__(self):
        super(EdgeDetection, self).__init__()
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        x = x.mean(dim=1, keepdim=True)  # Convert to grayscale
        edges_x = F.conv2d(x, self.sobel_x.to(x.device), padding=1)
        edges_y = F.conv2d(x, self.sobel_y.to(x.device), padding=1)
        edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)  # Magnitude of gradient
        return edges


# Cross-Attention Module
class CrossAttention2D(nn.Module):
    def __init__(self, query_channels, key_value_channels, out_channels):
        super(CrossAttention2D, self).__init__()
        self.query_conv = nn.Conv2d(query_channels, out_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(key_value_channels, out_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(key_value_channels, out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        B, C, H, W = query.size()  # [B, C, H, W]
        query = self.query_conv(query).view(B, -1, H * W).permute(0, 2, 1)  # [B, HW, C_out]
        key = self.key_conv(key).view(B, -1, H * W)  # [B, C_out, HW]
        value = self.value_conv(value).view(B, -1, H * W).permute(0, 2, 1)  # [B, HW, C_out]

        attention = torch.bmm(query, key)  # [B, HW, HW]
        attention = self.softmax(attention)  # Normalize attention scores
        out = torch.bmm(attention, value)  # [B, HW, C_out]
        out = out.permute(0, 2, 1).view(B, C, H, W)  # [B, C_out, H, W]
        return out
import torch.nn.functional as F
class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        assert self.head_dim * num_heads == input_dim, "input_dim must be divisible by num_heads"

        self.W_q = nn.Linear(input_dim, input_dim)  # Query projection
        self.W_k = nn.Linear(input_dim, input_dim)  # Key projection
        self.W_v = nn.Linear(input_dim, input_dim)  # Value projection
        
    def forward(self, x):
        batch_size, seq_length, _ = x.size()

        # Step 1: Linear projections
        Q = self.W_q(x)  # (batch_size, seq_length, input_dim)
        K = self.W_k(x)
        V = self.W_v(x)

        # Step 2: Reshape for multi-head attention
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_length, head_dim)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_length, head_dim)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_length, head_dim)

        # Step 3: Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch_size, num_heads, seq_length, seq_length)

        # Step 4: Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, num_heads, seq_length, seq_length)
        
        # Step 5: Weighted sum of values
        context = torch.matmul(attention_weights, V)  # (batch_size, num_heads, seq_length, head_dim)

        # Step 6: Concatenate heads and pass through final linear layer
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)  # (batch_size, seq_length, input_dim)

        return context, attention_weights


class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))  # (batch_size, seq_length, input_dim)


class TransformerBlock(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(input_dim, num_heads)
        self.feed_forward = FeedForward(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        
    def forward(self, x):
        # Attention
        attn_output, attn_weights = self.attention(x)
        x = self.norm1(x + attn_output)  # Residual connection + Layer normalization

        # Feedforward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)  # Residual connection + Layer normalization

        return x, attn_weights  # Return both the output and attention weights


class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, num_layers):
        super(TransformerModel, self).__init__()
        self.layers = nn.ModuleList([TransformerBlock(input_dim, num_heads, hidden_dim) for _ in range(num_layers)])

    def forward(self, x):
        # Store attention weights for each layer
        attention_weights_list = []

        for layer in self.layers:
            x, attn_weights = layer(x)
            attention_weights_list.append(attn_weights)

        # Aggregate attention weights across heads (average or sum) and combine across layers
        # Example: Use the attention weights from the last layer
        final_attention_weights = attention_weights_list[-1]  # This has shape (batch_size, num_heads, seq_length, seq_length)

        # Average across heads to get to (batch_size, seq_length, seq_length)
        final_attention_weights = final_attention_weights.mean(dim=1)  # Shape: (batch_size, seq_length, seq_length)
        batch_size, seq_length, _ = final_attention_weights.shape

        # 1. Set diagonal elements to 0
        for i in range(batch_size):
            final_attention_weights[i].fill_diagonal_(0)

        # 2. Normalize each row to ensure the sum is 1
        # Calculate row sums excluding the diagonal (set diagonal to 0 before)
        row_sums = final_attention_weights.sum(dim=2, keepdim=True)  # Shape: (batch_size, seq_length, 1)

        # Avoid division by zero: add a small epsilon to the sum to prevent NaNs if the row is all zero
        row_sums = row_sums + 1e-8  # Add a small value to avoid division by zero

        # Normalize the rows: divide by the sum of each row
        final_attention_weights = final_attention_weights / row_sums
        return final_attention_weights
class LaneATT(nn.Module):
    def __init__(self,
                 backbone='resnet34',
                 pretrained_backbone=True,
                 S=72,
                 img_w=640,
                 img_h=360,
                 anchors_freq_path=None,
                 topk_anchors=None,
                 anchor_feat_channels=64):
        super(LaneATT, self).__init__()
        # Some definitions
        self.feature_extractor, backbone_nb_channels, self.stride = get_backbone(backbone, pretrained_backbone)
        self.img_w = img_w
        self.n_strips = S - 1
        self.n_offsets = S
        self.fmap_h = img_h // self.stride
        fmap_w = img_w // self.stride
        #fmap_w = img_w // self.stride
        self.fmap_w = fmap_w
        self.anchor_ys = torch.linspace(1, 0, steps=self.n_offsets, dtype=torch.float32)
        self.anchor_cut_ys = torch.linspace(1, 0, steps=self.fmap_h, dtype=torch.float32)
        self.anchor_feat_channels = anchor_feat_channels

        # Anchor angles, same ones used in Line-CNN
        self.left_angles = [72., 60., 49., 39., 30., 22.]
        self.right_angles = [108., 120., 131., 141., 150., 158.]
        self.bottom_angles = [165., 150., 141., 131., 120., 108., 100., 90., 80., 72., 60., 49., 39., 30., 15.]

        # Generate anchors
        self.anchors, self.anchors_cut = self.generate_anchors(lateral_n=72, bottom_n=128)
        self.anchorsc,self.anchorsc_cut= self.generate_anchors1(lateral_n=72, bottom_n=128)
        #print('This is self.anchors',self.anchors)
        #print('This is self.anchors_cut',self.anchors_cut)
        # Filter masks if `anchors_freq_path` is provided
        #print('This is anchors_freq_path',anchors_freq_path)
        #print('This is topk_anchors',topk_anchors)
        if anchors_freq_path is not None:
            anchors_mask = torch.load('/kaggle/working/LaneATT/data/tusimple_anchors_freq.pt').cpu()
            assert topk_anchors is not None
            ind = torch.argsort(anchors_mask, descending=True)[:topk_anchors]
            self.anchors = self.anchors[ind]
            self.anchors_cut = self.anchors_cut[ind]
        self.cut_zs, self.cut_ys, self.cut_xs, self.invalid_mask = self.compute_anchor_cut_indices(
            self.anchor_feat_channels, fmap_w, self.fmap_h)

        # Setup and initialize layers
        self.conv2 = nn.Conv2d(1024, self.anchor_feat_channels, kernel_size=1)
        self.conv1 = nn.Conv2d(512, self.anchor_feat_channels, kernel_size=1)
        self.cls_layer = nn.Linear(2 * self.anchor_feat_channels * self.fmap_h, 2)
        self.reg_layer = nn.Linear(2 * self.anchor_feat_channels * self.fmap_h, self.n_offsets + 1)
        self.attention_layer = nn.Linear(self.anchor_feat_channels * self.fmap_h, len(self.anchors) - 1)
        self.initialize_layer(self.conv2)
        self.initialize_layer(self.attention_layer)
        self.initialize_layer(self.conv1)
        self.initialize_layer(self.cls_layer)
        self.initialize_layer(self.reg_layer)
        self.theta = nn.Conv2d(512, 512 // 2, kernel_size=1)
        self.phi = nn.Conv2d(512, 512 // 2, kernel_size=1)
        self.g = nn.Conv2d(512, 512 // 2, kernel_size=1)
        self.out_conv = nn.Conv2d(512 // 2, 512, kernel_size=1)
        self.initialize_layer(self.theta)
        self.initialize_layer(self.phi)
        self.initialize_layer(self.g)
        self.initialize_layer(self.out_conv)
        self.dropout = nn.Dropout(0.2) 
        self.B1= BottleStack(
    dim = 512,              # channels in
    fmap_size = (12, 20),         # feature map size
    dim_out = 512,         # channels out
    proj_factor = 4,        # projection factor
    downsample = False,      # downsample on first layer or not
    heads = 12,              # number of heads
    dim_head = 512,         # dimension per head, defaults to 128
    rel_pos_emb = False,    # use relative positional embedding - uses absolute if False
    activation = nn.ReLU() )
        self.initialize_layer(self.B1)
        self.num_groups = 4
        self.channels_per_group = 512 // 4
        self.group_convs = nn.ModuleList([
            nn.Conv2d(self.channels_per_group, self.channels_per_group // 4, kernel_size=1, stride=1)
            for _ in range(4)
        ])
        self.final_conv = nn.Conv2d((self.channels_per_group // 4) * 4, 512, kernel_size=1)
        self.initialize_layer(self.final_conv)
    def forward(self, x, conf_threshold=None, nms_thres=0, nms_topk=3000):
        
        batch_features1 = self.feature_extractor(x)
        batch_features = self.B1(batch_features1)
    
        # Use cross-attention with feature map as query and edge map as key/value

       # A=CSPStage(512,64,4,spp=True).to('cuda')
       # batch_features=A(batch_features)
        #model = SwinFeatureExtractor()
       # batch_features=model(x)

       # bach=CBAM(512,512).to('cuda')
       # batch_size, _, height, width = batch_features.size()
       # theta = self.theta(batch_features).view(batch_size, -1, height * width)  # (B, C/2, H*W)
       # phi = self.phi(batch_features).view(batch_size, -1, height * width)      # (B, C/2, H*W)
      #  g = self.g(batch_features).view(batch_size, -1, height * width)          # (B, C/2, H*W)
      #  theta_phi = torch.bmm(theta.permute(0, 2, 1), phi)  # (B, H*W, H*W)
      #  attention = F.softmax(theta_phi, dim=-1)  # (B, H*W, H*W)
      #  weighted_g = torch.bmm(g, attention.permute(0, 2, 1))  # (B, C/2, H*W)
      #  weighted_g = weighted_g.view(batch_size, -1, height, width)  # (B, C/2, H, W)
       # batch_features = self.out_conv(weighted_g) + batch_features
        #batch_features=self.conv1(batch_features)
        groups = torch.split(batch_features1, self.channels_per_group, dim=1)
        processed_groups = [conv(group) for group, conv in zip(groups, self.group_convs)]
        aggregated = torch.cat(processed_groups, dim=1)
        output = self.final_conv(aggregated)
        batch_features = torch.cat([output,batch_features] ,dim=1)
        batch_features=self.conv2(batch_features)
        batch_anchor_features = self.cut_anchor_features(batch_features)

        # Join proposals from all images into a single proposals features batch
        batch_anchor_features = batch_anchor_features.view(-1, self.anchor_feat_channels * self.fmap_h)
        
        # Add attention features
        batch_size = x.shape[0]
        num_proposals = len(self.anchors)
        d_k = batch_anchor_features.shape[1]
        batch_size = x.shape[0]
        seq_length = len(self.anchors)
        input_dim = batch_anchor_features.shape[1]
        num_heads = 4
        hidden_dim = 512
        num_layers = 6


        #anchor_features = batch_anchor_features.view(batch_size, num_proposals, d_k)


       # transformer_model = TransformerModel(input_dim, num_heads, hidden_dim, num_layers).to('cuda')
       # attention_matrix = transformer_model(anchor_features)
        softmax = nn.Softmax(dim=1)
        scores = self.attention_layer(batch_anchor_features)
        attention = softmax(scores).reshape(x.shape[0], len(self.anchors), -1)
        attention_matrix = torch.eye(attention.shape[1], device=x.device).repeat(x.shape[0], 1, 1)
        non_diag_inds = torch.nonzero(attention_matrix == 0., as_tuple=False)
        attention_matrix[:] = 0
        attention_matrix[non_diag_inds[:, 0], non_diag_inds[:, 1], non_diag_inds[:, 2]] = attention.flatten()
        #attention_matrix=attention_matrix1*attention_matrix
       # attention_matrix=attention_matrix/2
        batch_anchor_features = batch_anchor_features.reshape(x.shape[0], len(self.anchors), -1)
        attention_features = torch.bmm(torch.transpose(batch_anchor_features, 1, 2),
                                       torch.transpose(attention_matrix, 1, 2)).transpose(1, 2)
        attention_features = attention_features.reshape(-1, self.anchor_feat_channels * self.fmap_h)
        batch_anchor_features = batch_anchor_features.reshape(-1, self.anchor_feat_channels * self.fmap_h)
        batch_anchor_features = torch.cat((attention_features, batch_anchor_features), dim=1)
        attention_weights = self.dropout(batch_anchor_features)
        # Predict
        cls_logits = self.cls_layer(batch_anchor_features)
        reg = self.reg_layer(batch_anchor_features)

        # Undo joining
        cls_logits = cls_logits.reshape(x.shape[0], -1, cls_logits.shape[1])
        reg = reg.reshape(x.shape[0], -1, reg.shape[1])

        # Add offsets to anchors
        reg_proposals = torch.zeros((*cls_logits.shape[:2], 5 + self.n_offsets), device=x.device)
        reg_proposals += self.anchors
        reg_proposals[:, :, :2] = cls_logits
        reg_proposals[:, :, 4:] += reg

        # Apply nms
        proposals_list = self.nms(reg_proposals, attention_matrix, nms_thres, nms_topk, conf_threshold)

        return proposals_list

    def nms(self, batch_proposals, batch_attention_matrix, nms_thres, nms_topk, conf_threshold):
        softmax = nn.Softmax(dim=1)
        proposals_list = []
        for proposals, attention_matrix in zip(batch_proposals, batch_attention_matrix):
            anchor_inds = torch.arange(batch_proposals.shape[1], device=proposals.device)
            # The gradients do not have to (and can't) be calculated for the NMS procedure
            with torch.no_grad():
                scores = softmax(proposals[:, :2])[:, 1]
                if conf_threshold is not None:
                    # apply confidence threshold
                    above_threshold = scores > conf_threshold
                    proposals = proposals[above_threshold]
                    scores = scores[above_threshold]
                    anchor_inds = anchor_inds[above_threshold]
                if proposals.shape[0] == 0:
                    proposals_list.append((proposals[[]], self.anchors[[]], attention_matrix[[]], None))
                    continue
                keep, num_to_keep, _ = nms(proposals, scores, overlap=nms_thres, top_k=nms_topk)
                keep = keep[:num_to_keep]
            proposals = proposals[keep]
            anchor_inds = anchor_inds[keep]
            attention_matrix = attention_matrix[anchor_inds]
            proposals_list.append((proposals, self.anchors[keep], attention_matrix, anchor_inds))

        return proposals_list

    def loss(self, proposals_list, targets, cls_loss_weight=10):
        focal_loss = FocalLoss(alpha=0.25, gamma=2.)
        cls_loss_weight=2.
        smooth_l1_loss = nn.SmoothL1Loss()
        cls_loss = 0
        reg_loss = 0
        iou_loss=0
        valid_imgs = len(targets)
        total_positives = 0
        for (proposals, anchors, _, _), target in zip(proposals_list, targets):
            # Filter lanes that do not exist (confidence == 0)
            target = target[target[:, 1] == 1]
            if len(target) == 0:
                # If there are no targets, all proposals have to be negatives (i.e., 0 confidence)
                cls_target = proposals.new_zeros(len(proposals)).long()
                cls_pred = proposals[:, :2]
                cls_loss += focal_loss(cls_pred, cls_target).sum()
                continue
            # Gradients are also not necessary for the positive & negative matching
            with torch.no_grad():
                positives_mask, invalid_offsets_mask, negatives_mask, target_positives_indices = match_proposals_with_targets(
                    self, anchors, target)

            positives = proposals[positives_mask]
            num_positives = len(positives)
            total_positives += num_positives
            negatives = proposals[negatives_mask]
            num_negatives = len(negatives)

            # Handle edge case of no positives found
            if num_positives == 0:
                cls_target = proposals.new_zeros(len(proposals)).long()
                cls_pred = proposals[:, :2]
                cls_loss += focal_loss(cls_pred, cls_target).sum()
                continue

            # Get classification targets
            all_proposals = torch.cat([positives, negatives], 0)
            cls_target = proposals.new_zeros(num_positives + num_negatives).long()
            cls_target[:num_positives] = 1.
            cls_pred = all_proposals[:, :2]

            # Regression targets
            reg_pred = positives[:, 4:]
            with torch.no_grad():
                target = target[target_positives_indices]
                positive_starts = (positives[:, 2] * self.n_strips).round().long()
                target_starts = (target[:, 2] * self.n_strips).round().long()
                target[:, 4] -= positive_starts - target_starts
                all_indices = torch.arange(num_positives, dtype=torch.long, device=invalid_offsets_mask.device)
                ends = (positive_starts + target[:, 4] - 1).round().long()

                invalid_offsets_mask = torch.zeros((num_positives, 1 + self.n_offsets + 1),
                                   dtype=torch.int, device=invalid_offsets_mask.device)  # length + S + pad
                invalid_offsets_mask[all_indices, 1 + positive_starts] = 1
                invalid_offsets_mask[all_indices, 1 + ends + 1] -= 1
                invalid_offsets_mask = invalid_offsets_mask.cumsum(dim=1) == 0
                invalid_offsets_mask = invalid_offsets_mask[:, :-1]
                invalid_offsets_mask[:, 0] = False
                reg_target = target[:, 4:]
                reg_target[invalid_offsets_mask] = reg_pred[invalid_offsets_mask]
            
            # Loss calc

            reg_loss += smooth_l1_loss(reg_pred, reg_target)
            cls_loss += focal_loss(cls_pred, cls_target).sum() / num_positives

        # Batch mean

        cls_loss /= valid_imgs
        reg_loss /= valid_imgs

        loss = cls_loss_weight * cls_loss + reg_loss
        return loss, {'cls_loss': cls_loss, 'reg_loss': reg_loss, 'batch_positives': total_positives}

    def compute_anchor_cut_indices(self, n_fmaps, fmaps_w, fmaps_h):
        # definitions
            n_proposals = len(self.anchors_cut)
            #print('This is self.anchors_cut in compute indices',self.anchors_cut)
            # indexing

            unclamped_xs = torch.flip((self.anchors_cut[:, 5:] / self.stride).round().long(), dims=(1,))
            unclamped_xs = unclamped_xs.unsqueeze(2)
            unclamped_xs = torch.repeat_interleave(unclamped_xs, n_fmaps, dim=0).reshape(-1, 1)
            cut_xs = torch.clamp(unclamped_xs, 0, fmaps_w - 1)
            unclamped_xs = unclamped_xs.reshape(n_proposals, n_fmaps, fmaps_h, 1)
            invalid_mask = (unclamped_xs < 0) | (unclamped_xs > fmaps_w)
            cut_ys = torch.arange(0, fmaps_h)
            cut_ys = cut_ys.repeat(n_fmaps * n_proposals)[:, None].reshape(n_proposals, n_fmaps, fmaps_h)
            cut_ys = cut_ys.reshape(-1, 1)
            cut_zs = torch.arange(n_fmaps).repeat_interleave(fmaps_h).repeat(n_proposals)[:, None]

            return cut_zs, cut_ys, cut_xs, invalid_mask

    def cut_anchor_features(self, features):
        # definitions
        batch_size = features.shape[0]
        #print('This is batch_size',batch_size)
        n_proposals = len(self.anchors)
        #print('This is len(self.anchors) in cut features',len(self.anchors))
        n_fmaps = features.shape[1]
        #print('This is n_fmaps in cut features',n_fmaps)
        batch_anchor_features = torch.zeros((batch_size, n_proposals, n_fmaps, self.fmap_h, 1), device=features.device)
        #print('This is batch_anchor_features in cut features',batch_anchor_features.shape)
        # actual cutting
        for batch_idx, img_features in enumerate(features):
            rois = img_features[self.cut_zs, self.cut_ys, self.cut_xs].view(n_proposals, n_fmaps, self.fmap_h, 1)
            rois[self.invalid_mask] = 0
            batch_anchor_features[batch_idx] = rois

        return batch_anchor_features

    def generate_anchors(self, lateral_n, bottom_n):
        left_anchors, left_cut = self.generate_side_anchors(self.left_angles, x=0., nb_origins=lateral_n)
        right_anchors, right_cut = self.generate_side_anchors(self.right_angles, x=1., nb_origins=lateral_n)        
        bottom_anchors, bottom_cut = self.generate_side_anchors(self.bottom_angles, y=1., nb_origins=bottom_n)
        return torch.cat([left_anchors, bottom_anchors,right_anchors]), torch.cat([left_cut, bottom_cut, right_cut])
        
    def generate_anchors1(self, lateral_n, bottom_n):
        leftc_anchors, leftc_cut = self.generate_side_anchors1(self.left_angles, x=0., nb_origins=lateral_n)
        rightc_anchors, rightc_cut = self.generate_side_anchors1(self.right_angles, x=1., nb_origins=lateral_n)
        bottomc_anchors, bottomc_cut = self.generate_side_anchors1(self.bottom_angles, y=1., nb_origins=bottom_n)
        return torch.cat([leftc_anchors, bottomc_anchors,rightc_anchors]), torch.cat([leftc_cut, bottomc_cut, rightc_cut])
        
    def generate_side_anchors(self, angles, nb_origins, x=None, y=None):
        if x is None and y is not None:
            starts = [(x, y) for x in np.linspace(1., 0., num=nb_origins)]
        elif x is not None and y is None:
            starts = [(x, y) for y in np.linspace(1., 0., num=nb_origins)]
        else:
            raise Exception('Please define exactly one of `x` or `y` (not neither nor both)')
        n_anchors = nb_origins * len(angles)

        # each row, first for x and second for y:
        # 2 scores, 1 start_y, start_x, 1 lenght, S coordinates, score[0] = negative prob, score[1] = positive prob
        anchors = torch.zeros((n_anchors, 2 + 2 + 1 + self.n_offsets))
        anchors_cut = torch.zeros((n_anchors, 2 + 2 + 1 + self.fmap_h))
        for i, start in enumerate(starts):
            for j, angle in enumerate(angles):
                k = i * len(angles) + j
                anchors[k] = self.generate_anchor(start, angle)
                anchors_cut[k] = self.generate_anchor(start, angle, cut=True)

        return anchors, anchors_cut
    def generate_side_anchors1(self, angles, nb_origins, x=None, y=None):
        if x is None and y is not None:
            starts = [(x, y) for x in np.linspace(1., 0., num=nb_origins)]
        elif x is not None and y is None:
            starts = [(x, y) for y in np.linspace(1., 0., num=nb_origins)]
        else:
            raise Exception('Please define exactly one of `x` or `y` (not neither nor both)')

        n_anchors = nb_origins * len(angles)

        # each row, first for x and second for y:
        # 2 scores, 1 start_y, start_x, 1 lenght, S coordinates, score[0] = negative prob, score[1] = positive prob
        anchors = torch.zeros((n_anchors, 2 + 2 + 1 + self.n_offsets))
        anchors_cut = torch.zeros((n_anchors, 2 + 2 + 1 + self.fmap_h))
        for i, start in enumerate(starts):
            for j, angle in enumerate(angles):
                k = i * len(angles) + j
                anchors[k] = self.generate_anchor1(start, angle)
                anchors_cut[k] = self.generate_anchor1(start, angle, cut=True)

        return anchors, anchors_cut
    def generate_anchor1(self, start, angle, cut=False):
        if cut:
            anchor_ys = self.anchor_cut_ys
            anchor = torch.zeros(2 + 2 + 1 + self.fmap_h)
        else:
            anchor_ys = self.anchor_ys
            anchor = torch.zeros(2 + 2 + 1 + self.n_offsets)
        angle = angle * math.pi / 180.  # degrees to radians
        start_x, start_y = start
        anchor[2] = 1 - start_y
        anchor[3] = start_x
        x_coords = start_x + (1 - anchor_ys - 1 + start_y) / math.tan(angle)
    
    # Introduce a perturbation that increases along the y-axis to create gradual curvature to the left
        scale = torch.linspace(0, 1, len(anchor_ys))
        perturbation = 0.5 * scale * torch.sin(2 * anchor_ys)  # Adjust amplitude (0.01) and frequency (5) as needed
        curved_ys = anchor_ys + perturbation
    
    # Calculate x-coordinates with the perturbed y-coordinates
        anchor[5:] = (start_x + (1 - curved_ys - 1 + start_y) / math.tan(angle+25)) * self.img_w
        return anchor
    def generate_anchor(self, start, angle, cut=False):
        if cut:
            anchor_ys = self.anchor_cut_ys
            anchor = torch.zeros(2 + 2 + 1 + self.fmap_h)
        else:
            anchor_ys = self.anchor_ys
            anchor = torch.zeros(2 + 2 + 1 + self.n_offsets)
        angle = angle * math.pi / 180.  # degrees to radians
        start_x, start_y = start
        anchor[2] = 1 - start_y
        anchor[3] = start_x
        anchor[5:] = (start_x + (1 - anchor_ys - 1 + start_y) / math.tan(angle)) * self.img_w

        return anchor

    def draw_anchors(self, img_w, img_h, k=None):
        base_ys = self.anchor_ys.numpy()
        img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        i = -1
        for anchor in self.anchors:
            i += 1
            if k is not None and i != k:
                continue
            anchor = anchor.numpy()
            xs = anchor[5:]
            ys = base_ys * img_h
            points = np.vstack((xs, ys)).T.round().astype(int)
            for p_curr, p_next in zip(points[:-1], points[1:]):
                img = cv2.line(img, tuple(p_curr), tuple(p_next), color=(0, 255, 0), thickness=5)

        return img

    @staticmethod
    def initialize_layer(layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            torch.nn.init.normal_(layer.weight, mean=0., std=0.001)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0)

    def proposals_to_pred(self, proposals):
        self.anchor_ys = self.anchor_ys.to(proposals.device)
        self.anchor_ys = self.anchor_ys.double()
        lanes = []
        for lane in proposals:
            lane_xs = lane[5:] / self.img_w
            start = int(round(lane[2].item() * self.n_strips))
            length = int(round(lane[4].item()))
            end = start + length - 1
            end = min(end, len(self.anchor_ys) - 1)
            # end = label_end
            # if the proposal does not start at the bottom of the image,
            # extend its proposal until the x is outside the image
            mask = ~((((lane_xs[:start] >= 0.) &
                       (lane_xs[:start] <= 1.)).cpu().numpy()[::-1].cumprod()[::-1]).astype(bool))
            lane_xs[end + 1:] = -2
            lane_xs[:start][mask] = -2
            lane_ys = self.anchor_ys[lane_xs >= 0]
            lane_xs = lane_xs[lane_xs >= 0]
            lane_xs = lane_xs.flip(0).double()
            lane_ys = lane_ys.flip(0)
            if len(lane_xs) <= 1:
                continue
            points = torch.stack((lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), dim=1).squeeze(2)
            lane = Lane(points=points.cpu().numpy(),
                        metadata={
                            'start_x': lane[3],
                            'start_y': lane[2],
                            'conf': lane[1]
                        })
            lanes.append(lane)
        return lanes

    def decode(self, proposals_list, as_lanes=False):
        softmax = nn.Softmax(dim=1)
        decoded = []
        for proposals, _, _, _ in proposals_list:
            proposals[:, :2] = softmax(proposals[:, :2])
            proposals[:, 4] = torch.round(proposals[:, 4])
            if proposals.shape[0] == 0:
                decoded.append([])
                continue
            if as_lanes:
                pred = self.proposals_to_pred(proposals)
            else:
                pred = proposals
            decoded.append(pred)

        return decoded

    def cuda(self, device=None):
        cuda_self = super().cuda(device)
        cuda_self.anchors = cuda_self.anchors.cuda(device)
        cuda_self.anchor_ys = cuda_self.anchor_ys.cuda(device)
        cuda_self.cut_zs = cuda_self.cut_zs.cuda(device)
        cuda_self.cut_ys = cuda_self.cut_ys.cuda(device)
        cuda_self.cut_xs = cuda_self.cut_xs.cuda(device)
        cuda_self.invalid_mask = cuda_self.invalid_mask.cuda(device)
        return cuda_self

    def to(self, *args, **kwargs):
        device_self = super().to(*args, **kwargs)
        device_self.anchors = device_self.anchors.to(*args, **kwargs)
        device_self.anchor_ys = device_self.anchor_ys.to(*args, **kwargs)
        device_self.cut_zs = device_self.cut_zs.to(*args, **kwargs)
        device_self.cut_ys = device_self.cut_ys.to(*args, **kwargs)
        device_self.cut_xs = device_self.cut_xs.to(*args, **kwargs)
        device_self.invalid_mask = device_self.invalid_mask.to(*args, **kwargs)
        return device_self


def get_backbone(backbone, pretrained=False):
    if backbone == 'resnet122':
        backbone = resnet122_cifar()
        fmap_c = 64
        stride = 4
    elif backbone == 'resnet34':
        backbone = torch.nn.Sequential(*list(resnet34(pretrained=pretrained).children())[:-2])
        fmap_c = 512
        stride = 32
    elif backbone == 'resnet18':
        backbone = torch.nn.Sequential(*list(resnet18(pretrained=pretrained).children())[:-2])
        fmap_c = 512
        stride = 32
    else:
        raise NotImplementedError('Backbone not implemented: `{}`'.format(backbone))

    return backbone, fmap_c, stride

