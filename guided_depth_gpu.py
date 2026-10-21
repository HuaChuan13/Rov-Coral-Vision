"""Fast masked guided reconstruction; native color edges, NOT native stereo.

Fit local affine coefficients on the matching grid, evaluate on native pixels.
CUDA is used for interpolation/evaluation/depth/color, not for SGBM itself.
"""
import cv2
import numpy as np


def _nearest_support_indices(disparity, valid):
    """Indices of real support used as seeds, never measurement validity."""
    valid = valid & np.isfinite(disparity) & (disparity > 0)
    if np.count_nonzero(valid) < 32:
        return None
    _, labels = cv2.distanceTransformWithLabels((~valid).astype(np.uint8),
        cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    positions = np.zeros(int(labels.max())+1, np.int32)
    positions[labels[valid]] = np.flatnonzero(valid).astype(np.int32)
    return positions[labels]


def complete_disparity_for_display(disparity, valid):
    """Nearest supported disparity seed map, DISPLAY ONLY.

    No correspondence is claimed for extrapolated pixels, especially occlusions.
    With no useful anchors, refuse completion rather than invent a scene depth.
    """
    indices = _nearest_support_indices(disparity, valid)
    if indices is None:
        return None
    # Do not median-filter across different surfaces before edge reconstruction.
    return disparity.ravel()[indices]


def affine_coefficients(guide_small, disparity, valid, radius=2, eps=.001):
    image = guide_small.astype(np.float32) / 255.
    weight = valid.astype(np.float32)
    value = np.where(valid, disparity, 0).astype(np.float32)
    size = (2*radius+1, 2*radius+1)
    def box(x):
        return cv2.boxFilter(x, -1, size, borderType=cv2.BORDER_REFLECT)
    support = np.maximum(box(weight), 1e-6)
    mean_i = box(image*weight)/support
    mean_p = box(value)/support
    variance = np.maximum(0, box(image*image*weight)/support-mean_i*mean_i)
    covariance = box(image*value)/support-mean_i*mean_p
    a = covariance/(variance+eps)
    b = mean_p-a*mean_i
    a = box(a*weight)/support
    b = box(b*weight)/support
    return np.stack((a, b))


class CudaGuidedDepth:
    def __init__(self, device, near, far, keep_on_gpu=False, complete_display=False):
        import torch
        self.torch = torch
        self.device = torch.device('cuda:' + str(device))
        self.near, self.far = near, far
        self.keep_on_gpu = keep_on_gpu
        self.complete_display = complete_display
        self.display_depth = None
        palette = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(256, 1), cv2.COLORMAP_TURBO).reshape(256, 3)
        self.palette = torch.from_numpy(palette.copy()).to(self.device)
        from cuda_guided_kernel import FusedGuidedKernel
        self.fused = FusedGuidedKernel(torch, self.device)

    def reconstruct(self, guide, filtered, raw, valid, strict, geometry):
        torch = self.torch
        import torch.nn.functional as F
        shape = guide.shape[:2]
        scale = shape[1]/geometry.stereo_size[0]
        gray = cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY)
        low_gray = cv2.resize(gray, geometry.stereo_size, interpolation=cv2.INTER_AREA)
        coefficients = affine_coefficients(low_gray, filtered, valid)
        seeds = None
        if self.complete_display:
            in_range = valid & (filtered-geometry.cx_delta >= geometry.fB/self.far) & (filtered-geometry.cx_delta <= geometry.fB/self.near)
            seeds = _nearest_support_indices(filtered, in_range)
        # Nearest support preserves holes and avoids turning filled pixels into
        # measured observations. Raw disparity remains the consistency anchor.
        masks = np.stack((valid.astype(np.float32), strict.astype(np.float32), raw))
        with torch.inference_mode(), torch.cuda.device(self.device):
            display_inputs = None
            if seeds is not None:
                # Color hypotheses need only center-aligned guidance, not a
                # second full area reduction. Keep measurement low_gray intact.
                low_color = cv2.resize(guide, geometry.stereo_size, interpolation=cv2.INTER_LINEAR)
                display_inputs = tuple(torch.from_numpy(np.ascontiguousarray(a)).to(self.device, non_blocking=True)
                    for a in (filtered, seeds, low_color, guide))
            # Enqueue uploads on one stream instead of synchronizing after
            # every small tensor. The color download below waits for completion;
            # all NumPy source buffers remain alive until that return.
            result = self.fused(torch.from_numpy(coefficients).to(self.device, non_blocking=True),
                              torch.from_numpy(masks).to(self.device, non_blocking=True),
                              torch.from_numpy(gray).to(self.device, non_blocking=True), self.palette,
                              geometry, self.near, self.far, self.keep_on_gpu,
                              seeds is not None, display_inputs=display_inputs)
            self.display_depth = self.fused.display_depth
            return result
    
    def reference_reconstruct(self, guide, filtered, raw, valid, strict, geometry):
        """Slower tensor reference retained for CUDA kernel regression tests."""
        torch = self.torch
        import torch.nn.functional as F
        shape = guide.shape[:2]
        scale = shape[1]/geometry.stereo_size[0]
        gray = cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY)
        low_gray = cv2.resize(gray, geometry.stereo_size, interpolation=cv2.INTER_AREA)
        coefficients = affine_coefficients(low_gray, filtered, valid)
        masks = np.stack((valid.astype(np.float32), strict.astype(np.float32), raw))
        with torch.inference_mode(), torch.cuda.device(self.device):
            coeff = F.interpolate(torch.from_numpy(coefficients).to(self.device)[None], size=shape,
                                  mode='bilinear', align_corners=False)[0]
            anchors = F.interpolate(torch.from_numpy(masks).to(self.device)[None], size=shape, mode='nearest')[0]
            native = torch.from_numpy(gray).to(self.device).float()/255.
            disparity = (coeff[0]*native+coeff[1])*scale
            denominator = disparity-geometry.cx_delta*scale
            depth = geometry.fB*scale / denominator.clamp(min=.001)
            ok = (anchors[0] > .5) & (denominator > .1*scale) & (depth >= self.near) & (depth <= self.far)
            right_x = torch.arange(shape[1], device=self.device)[None, :]-disparity
            ok &= (right_x >= 0) & (right_x <= shape[1]-1)
            trusted = ok & (anchors[1] > .5) & ((disparity-anchors[2]*scale).abs() <= 2*scale)
            estimated = torch.where(ok, depth, torch.full_like(depth, float('nan')))
            measured = torch.where(trusted, depth, torch.full_like(depth, float('nan')))
            levels = (10+245*(1/depth.clamp(min=.001)-1/self.far)/(1/self.near-1/self.far)).clamp(10, 255).long()
            levels = torch.where(ok, levels, torch.zeros_like(levels))
            colors = self.palette[levels]
            return estimated.cpu().numpy(), measured.cpu().numpy(), colors.cpu().numpy()
