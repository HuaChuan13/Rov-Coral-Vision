"""Lightweight photometric enhancement for rectified underwater BGR images.

No remapping, resize, deconvolution or generated detail is performed. Apply this
to the detection/display image, retaining original stereo inputs for matching.
Use one enhancer per video stream; white balance state is deliberately slow.
"""
import cv2
import numpy as np


class UnderwaterEnhancer:
    """Conservative color correction, local contrast and thresholded sharpening."""

    def __init__(self, mode='mild'):
        if mode not in ('off', 'mild'):
            raise ValueError("Enhancement mode must be 'off' or 'mild'.")
        self.mode = mode
        self._gains = None
        self._clahe = cv2.createCLAHE(clipLimit=1.25, tileGridSize=(8, 8))
        self._levels = np.arange(256, dtype=np.float32)
        # Taper channel gains to unity near white, protecting bright highlights.
        self._highlight_weight = np.clip((255. - self._levels) / 47., 0., 1.)

    def _white_balance_lut(self, image):
        # At most ~160x90 samples for 1280x720; no full-frame float conversion.
        stride = max(1, int(np.ceil(max(image.shape[:2]) / 160.)))
        sample = image[::stride, ::stride]
        maximum = sample.max(axis=2)
        valid = (maximum >= 24) & (maximum <= 235)
        if np.count_nonzero(valid) >= 32:
            means = sample[valid].mean(axis=0)
            target = means.mean() / np.maximum(means, 1.)
            # Partial gray-world correction with strict channel gain limits.
            target = np.clip(1. + .35 * (target - 1.), .96, 1.08)
        else:
            target = np.ones(3, dtype=np.float64)
        if self._gains is None:
            self._gains = target.copy()
        else:
            self._gains += np.clip(.04 * (target - self._gains), -.002, .002)
        gains = 1. + self._highlight_weight[:, None] * (self._gains[None, :] - 1.)
        lut = np.clip(np.rint(self._levels[:, None] * gains), 0, 255).astype(np.uint8)
        return lut[:, None, :]

    def apply(self, image):
        """Return uint8 BGR of identical shape, without changing the input array."""
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or
                image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1):
            raise ValueError('Expected a nonempty uint8 BGR image with three channels.')
        if self.mode == 'off':
            return image.copy()
        balanced = cv2.LUT(image, self._white_balance_lut(image))
        ycc = cv2.cvtColor(balanced, cv2.COLOR_BGR2YCrCb)
        # Contiguous single-channel buffers avoid repeated implicit copies by
        # OpenCV. Keep full-resolution arithmetic in its compiled kernels.
        luminance = cv2.extractChannel(ycc, 0)
        # Estimate only the smooth luminance correction at up to 640 pixels.
        # Native image pixels are never replaced by a resized image.
        height, width = luminance.shape
        if width > 640:
            contrast_source = cv2.resize(luminance, (640, max(1, round(height * 640 / width))),
                                         interpolation=cv2.INTER_AREA)
        else:
            contrast_source = luminance
        local = self._clahe.apply(contrast_source)
        difference = cv2.subtract(local, contrast_source, dtype=cv2.CV_16S)
        contrast_delta = cv2.multiply(difference, .28, dtype=cv2.CV_16S)
        contrast_delta = cv2.min(cv2.max(contrast_delta, -12), 12)
        if contrast_delta.shape != luminance.shape:
            contrast_delta = cv2.resize(contrast_delta, (width, height), interpolation=cv2.INTER_LINEAR)
        adjusted = cv2.add(luminance, contrast_delta, dtype=cv2.CV_8U)
        blurred = cv2.GaussianBlur(adjusted, (5, 5), .8, borderType=cv2.BORDER_REFLECT_101)
        detail = cv2.subtract(adjusted, blurred, dtype=cv2.CV_16S)
        # max(detail,2)+min(detail,-2) is a symmetric soft threshold at 2.
        sharpen_delta = cv2.addWeighted(cv2.max(detail, 2), .40,
                                        cv2.min(detail, -2), .40, 0., dtype=cv2.CV_16S)
        sharpen_delta = cv2.min(cv2.max(sharpen_delta, -5), 5)
        sharpened = cv2.add(adjusted, sharpen_delta, dtype=cv2.CV_8U)
        cv2.insertChannel(sharpened, ycc, 0)
        return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)

    def metadata(self):
        """Serializable description and current BGR gains, for run provenance."""
        return {
            'mode': self.mode,
            'method': 'bounded_white_balance_luma_CLAHE_soft_threshold_unsharp',
            'preserves_pixel_coordinates': True,
            'preserves_dimensions': True,
            'modifies_input': False,
            'white_balance_gain_range': [.96, 1.08],
            'white_balance_ema_alpha': .04,
            'white_balance_max_step_per_frame': .002,
            'current_bgr_gains': None if self._gains is None else self._gains.tolist(),
            'clahe_clip_limit': 1.25,
            'clahe_blend': .28,
            'contrast_correction_field_max_width': 640,
            'contrast_max_luma_delta': 12,
            'sharpen_amount': .40,
            'sharpen_noise_threshold': 2,
            'sharpen_max_luma_delta': 5,
            'focus_recovery': False,
            'note': 'Photometric enhancement only; does not recover lost focus detail. Keep original stereo inputs for geometric measurement.',
        }
