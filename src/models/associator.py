"""Fuse history before matching
"""


from copy import deepcopy
import math
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


def hungarian_matching(scores, matching_threshold):
    """ find active matching from detection to track given a score matrix
    
    Args:
        scores: [n_tracks, n_detections], matching score, between [0, 1]
        matching_threshold: threshold for active matching
    
    Returns:
        match_tracked_ids: [n_detections], the track id of each mactched track,
            -1 if not matched.
    """
    scores = scores.detach().cpu().numpy()
    match_tracked_ids = np.zeros(scores.shape[1]) - 1
    row_indices, col_indices = linear_sum_assignment(1 - scores)  # cost = 1-score
    for row_ind, col_ind in zip(row_indices, col_indices):
        if scores[row_ind, col_ind] > matching_threshold:
            match_tracked_ids[col_ind] = row_ind
    return match_tracked_ids


def get_matching(assignments, valid_list):
    indices = []
    for b, assignment in enumerate(assignments):
        n_tracks, n_detections = valid_list[b]
        match_out = assignment[0].max(0).indices.detach().cpu().numpy()[:-1]
        match_out[match_out == n_tracks] = -1
        indices.append(match_out)
    return indices


def attention(query, key, value, attn_mask=None):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    num_heads = scores.shape[1]
    if attn_mask is not None:
        attn_mask = attn_mask.unsqueeze(1).repeat(1, num_heads, 1, 1)
        scores = scores.masked_fill(attn_mask, float('-100'))
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


def MLP(channels: list, do_bn=False):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            torch.nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(torch.nn.BatchNorm1d(channels[i]))
            layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)


class MultiHeadedAttention(torch.nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = torch.nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = torch.nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value, attn_mask):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, prob = attention(query, key, value, attn_mask)
        self.prob.append(prob)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))


class AttentionalPropagation(torch.nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        # self.attn = nn.MultiHeadedAttention(feature_dim, num_heads)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        torch.nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source, attn_mask):
        message = self.attn(x, source, source, attn_mask)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(torch.nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, tracks, detections):
        """ GNN between a track and a set of detections
        Args:
            tracks: [n_tracks, n_features, n_times]
            detections: [n_tracks, n_features, n_detections]
            self_attn_mask: [n_tracks, n_detections, n_detections]
            cross_attn_mask: [n_tracks, n_detections, n_times]
            gt_mask: [n_tracks, n_detections], filter out padding positions
        """
        n_dim = detections.shape[1]
        for layer, name in zip(self.layers, self.names):
            layer.attn.prob = []
            if name == 'cross':
                src0, src1 = detections, tracks
            else:  # if name == 'self':
                src0, src1 = tracks, detections

            delta0 = layer(tracks, src0, None) 
            delta1 = layer(detections, src1, None)
            tracks, detections = (tracks + delta0), (detections + delta1)
        return tracks, detections


class SelfAttentionalGNN(torch.nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, tracks):
        """ GNN between a track and a set of detections
        Args:
            tracks: [n_tracks, n_features, n_times]
        """

        for layer in self.layers:
            layer.attn.prob = []
            delta0 = layer(tracks, tracks, None) 
            tracks = (tracks + delta0)
        return tracks        


class Associator(torch.nn.Module):
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            default_config = {
                'descriptor_dim': 256,
                'weights': 'indoor',
                'keypoint_encoder': [256, 256, 256],
                'GNN_layers': ['self', 'cross'] * 6,
                'self_GNN_layers': ['self'] * 4,
                'match_threshold': 0.5,
                'sinkhorn_iterations': 100
            }
        else:
            default_config = config
        self.config = default_config
        self.encoder = MLP(default_config['keypoint_encoder'], do_bn=False)
        self.gnn = AttentionalGNN(
            default_config['descriptor_dim'], default_config['GNN_layers'])
        self.fuser = SelfAttentionalGNN(default_config['descriptor_dim'], default_config['self_GNN_layers'])

        self.final_proj = torch.nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)
        self.positional_encoding = PositionalEncoding()
        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

    def _reshape_tracks(self, fused_tracks, valid_list):
        device = fused_tracks.device
        batch_size = len(valid_list)
        max_objs = max([v[0] for v in valid_list])
        n_features = fused_tracks.shape[1]
        out_tracks = []
        start_idx = 0

        for b in np.arange(batch_size):
            n_tracks, _ = valid_list[b]
            end_idx = start_idx + n_tracks
            padding = torch.ones((n_features, max_objs - n_tracks)).to(device) * -1
            padded_tracks = torch.cat([fused_tracks[start_idx: end_idx, :, 0].T, padding], dim=1)
            out_tracks.append(padded_tracks)
            start_idx = end_idx
        out_tracks = torch.stack(out_tracks, dim=0)
        return out_tracks            

    def forward(self, in_data, threshold, eval_only=False, device="cuda"):
        # tracks: [n_tracks, n_features, n_times]
        # detections: [b, n_features, n_detections]
        # valid_list: (list): [n_valid_tracks, n_valid_detections] * b 
        # gt_list: [track_id, detect_id] * b

        detections = in_data['detections'].to(device)
        tracks = in_data['tracks'].to(device)
        valid_list = in_data['valid_list']
        if not eval_only:
            gt_list = in_data['gt_matches']

        # fuse tracks using attentional GraphNN
        # [n_tracks, n_features, 100 (time_steps)] ->
        # [n_tracks, n_features, 1]
        batch_size = detections.shape[0]

        detection_pe = self.positional_encoding(detections[:, 0, :])
        track_pe = self.positional_encoding(tracks[:, 0, :])

        tracks = self.encoder(tracks[:, 1:, :])
        detections = self.encoder(detections[:, 1:, :])

        detections = detections + detection_pe

        fused_tracks = self.fuser(tracks + track_pe)
        n_time_steps = fused_tracks.shape[2]
        fused_tracks = F.avg_pool1d(fused_tracks, kernel_size=n_time_steps)

        # reshape fused_tracksed by batch with padding
        # [n_tracks, n_features, 1] -> [b, n_features, max_tracks]
        fused_tracks = self._reshape_tracks(fused_tracks, valid_list)

        # compute matching between tracks and detections
        # matching_features: [n_tracks, n_detections, n_features]
        mdesc0, mdesc1 = self.gnn(fused_tracks, detections)

        mdesc0, mdesc1 = self.final_proj(mdesc0), self.final_proj(mdesc1)

        # the final score between 
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.config['descriptor_dim']**.5

        loss_batch = 0
        assignments = []
        for b in range(batch_size):
            valid_ids = valid_list[b]
            valid_scores = scores[b:b+1, :valid_ids[0], :valid_ids[1]]
            assignment = log_optimal_transport(
                valid_scores, alpha=self.bin_score,
                iters=self.config['sinkhorn_iterations'])
            assignments.append(assignment)
            if torch.isnan(assignment).any():
                continue
            if not eval_only:
                gt_matches = gt_list[b]
                loss_batch += torch.sum(-1 * assignment[0, gt_matches[:, 0], gt_matches[:, 1]])
        match_out = get_matching(assignments, valid_list)
        match_out = [hungarian_matching(assignment[0, :-1, :-1].exp(), threshold) for assignment in assignments]
        # match_out = assignments[0][0].max(0).indices.detach().cpu().numpy()[:-1]
        # match_out[match_out == n_tracks] = -1
        out = {
            "pred": assignments,
            "loss": loss_batch,
            "matches": match_out
        }
        return out

    def _duplicate(self, features, n_times):
        """
        
        Args:
            features: [n_features, n_detections]
        
        Returns:
            features: [n_times, n_features, n_detections]
        """

        return features.repeat(n_times, 1, 1)


def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model=256, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).to("cuda")

    def forward(self, position):
        pe = torch.zeros(position.shape[0], position.shape[1], self.d_model).to("cuda")
        pe[:, :, 0::2] = torch.sin(position.unsqueeze(-1).detach() * self.div_term)
        pe[:, :, 1::2] = torch.cos(position.unsqueeze(-1).detach() * self.div_term)
        pe = pe.transpose(1, 2)
        return pe


def build(args):
    model_config = {
        'descriptor_dim': args.descriptor_dim,
        'weights': args.weights,
        'keypoint_encoder': args.keypoint_encoder,
        'GNN_layers': args.GNN_layers,
        'self_GNN_layers': args.self_GNN_layers,
        'match_threshold': args.match_threshold,
        'sinkhorn_iterations': args.sinkhorn_iterations
    }
    return Associator(model_config)


if __name__ == "__main__":
    model = Associator()
    model = model.to("cuda")
    tracks = torch.zeros((5, 256, 100)).float().to("cuda")
    detections = torch.zeros((256, 3)).float().to("cuda")
    
    model(tracks, detections)