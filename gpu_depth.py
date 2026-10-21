"""NVIDIA-only fused reconstruction; SGBM/WLS matching remains on CPU."""
import ctypes as C
from pathlib import Path

import cv2
import numpy as np
from stereo import _nearest_support_indices


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

SOURCE = r'''
__device__ float bilinear(const float* p, int w, int h, float x, float y) {
    x=fminf(fmaxf(x,0.0f),(float)(w-1));
    y=fminf(fmaxf(y,0.0f),(float)(h-1));
    int x0=(int)x, y0=(int)y, x1=min(x0+1,w-1), y1=min(y0+1,h-1);
    float dx=x-x0, dy=y-y0;
    return (1-dy)*((1-dx)*p[y0*w+x0]+dx*p[y0*w+x1])
           +dy*((1-dx)*p[y1*w+x0]+dx*p[y1*w+x1]);
}
__device__ float color_distance(const unsigned char* a, int ai,
                               const unsigned char* b, int bi) {
    float d0=(float)a[3*ai]-b[3*bi];
    float d1=(float)a[3*ai+1]-b[3*bi+1];
    float d2=(float)a[3*ai+2]-b[3*bi+2];
    return d0*d0+d1*d1+d2*d2;
}
extern "C" __global__ void complete_low(
    const float* disparity, const int* seeds, const unsigned char* guide,
    float* filled, int w, int h) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=w*h) return;
    int x=i%w, y=i/w, best=seeds[i];
    if(best==i) { filled[i]=disparity[i]; return; }
    float dx=x-best%w, dy=y-best/w;
    float cost=color_distance(guide,i,guide,best)/2700.0f+(dx*dx+dy*dy)/1024.0f;
    // Propagate actual support hypotheses, not blended depths. Exponential
    // offsets span wide holes cheaply; this is display inference, NOT stereo.
    for(int radius=2;radius<=64;radius*=2) {
        for(int direction=0;direction<4;direction++) {
            int sx=x+(direction==0?radius:(direction==1?-radius:0));
            int sy=y+(direction==2?radius:(direction==3?-radius:0));
            sx=min(max(sx,0),w-1); sy=min(max(sy,0),h-1);
            int candidate=seeds[sy*w+sx];
            float cx=x-candidate%w, cy=y-candidate/w;
            float score=color_distance(guide,i,guide,candidate)/2700.0f+(cx*cx+cy*cy)/1024.0f;
            if(score<cost) { cost=score; best=candidate; }
        }
    }
    filled[i]=disparity[best];
}
__device__ float edge_upsample(const float* filled, const unsigned char* low_rgb,
    const unsigned char* rgb, int i, int lw, int lh, float u, float v) {
    int cx=(int)floorf(u+0.5f), cy=(int)floorf(v+0.5f);
    float best_score=1e30f, anchor=0;
    // Select a depth hypothesis using spatial AND native RGB agreement.
    // Crucially, never average competing foreground/background hypotheses.
    for(int yy=-1;yy<=1;yy++) for(int xx=-1;xx<=1;xx++) {
        int sx=min(max(cx+xx,0),lw-1), sy=min(max(cy+yy,0),lh-1), si=sy*lw+sx;
        float dx=u-sx,dy=v-sy;
        float score=color_distance(rgb,i,low_rgb,si)/1200.0f+(dx*dx+dy*dy)/2.25f;
        if(score<best_score) { best_score=score; anchor=filled[si]; }
    }
    float sum=0,weights=0,threshold=fmaxf(1.0f,0.04f*anchor);
    for(int yy=-1;yy<=1;yy++) for(int xx=-1;xx<=1;xx++) {
        int sx=min(max(cx+xx,0),lw-1), sy=min(max(cy+yy,0),lh-1), si=sy*lw+sx;
        float value=filled[si];
        if(fabsf(value-anchor)>threshold) continue;
        float dx=fabsf(u-sx),dy=fabsf(v-sy);
        float weight=fmaxf(0.0f,2.0f-dx)*fmaxf(0.0f,2.0f-dy)
                     /(1.0f+color_distance(rgb,i,low_rgb,si)/675.0f);
        sum+=weight*value; weights+=weight;
    }
    return weights>1e-6f?sum/weights:anchor;
}
extern "C" __global__ void reconstruct(
    const float* coeff, const float* anchors, const unsigned char* guide,
    const unsigned char* palette, float* output, unsigned char* colors,
    const float* display_filled, const unsigned char* low_rgb, const unsigned char* rgb,
    int w, int h, int lw, int lh, float scale, float fB, float delta,
    float near_z, float far_z, int complete_display) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int size=w*h, low=lw*lh;
    if(i>=size) return;
    int x=i%w, y=i/w;
    float u=(x+0.5f)/scale-0.5f, v=(y+0.5f)/scale-0.5f;
    float a=bilinear(coeff,lw,lh,u,v), b=bilinear(coeff+low,lw,lh,u,v);
    float d=(a*((float)guide[i]/255.0f)+b)*scale;
    int li=min((int)(y/scale),lh-1)*lw+min((int)(x/scale),lw-1);
    float den=d-delta*scale, z=fB*scale/fmaxf(den,0.001f);
    bool ok=anchors[li]>0.5f && x-d>=0 && x-d<=w-1 && den>0.1f*scale && z>=near_z && z<=far_z && isfinite(z);
    bool trusted=ok && anchors[low+li]>0.5f && fabsf(d-anchors[2*low+li]*scale)<=2.0f*scale;
    float missing=__int_as_float(0x7fffffff);
    output[i]=ok?z:missing;
    output[size+i]=trusted?z:missing;
    // Display-only extrapolation is separate from both measured outputs.
    float display_z=z;
    bool display_ok=ok;
    if(complete_display) {
        float cd=edge_upsample(display_filled,low_rgb,rgb,i,lw,lh,u,v)*scale;
        display_z=fminf(far_z,fmaxf(near_z,fB*scale/fmaxf(cd-delta*scale,0.001f)));
        display_ok=isfinite(display_z);
    }
    output[2*size+i]=display_ok?display_z:missing;
    int level=0;
    if(display_ok) level=(int)fminf(255.0f,fmaxf(10.0f,10.0f+245.0f*(1.0f/display_z-1.0f/far_z)/(1.0f/near_z-1.0f/far_z)));
    colors[3*i]=palette[3*level];
    colors[3*i+1]=palette[3*level+1];
    colors[3*i+2]=palette[3*level+2];
}
'''

class FusedGuidedKernel:
    def __init__(self, torch, device):
        self.torch, self.device = torch, device
        library = next((Path(torch.__file__).parent/'lib').glob('nvrtc64_*.dll'))
        rtc = C.CDLL(str(library))
        void, pv, char = C.c_void_p, C.POINTER(C.c_void_p), C.c_char_p
        rtc.nvrtcCreateProgram.argtypes = [pv, char, char, C.c_int, C.POINTER(char), C.POINTER(char)]
        rtc.nvrtcCompileProgram.argtypes = [void, C.c_int, C.POINTER(char)]
        for name in ('nvrtcGetPTXSize', 'nvrtcGetProgramLogSize'):
            getattr(rtc, name).argtypes = [void, C.POINTER(C.c_size_t)]
        rtc.nvrtcGetPTX.argtypes = [void, void]
        rtc.nvrtcGetProgramLog.argtypes = [void, void]
        rtc.nvrtcDestroyProgram.argtypes = [pv]
        program = void()
        self.check(rtc.nvrtcCreateProgram(C.byref(program), SOURCE.encode(), b'guided.cu', 0, None, None))
        try:
            major, minor = torch.cuda.get_device_capability(device)
            opts = (char*2)(b'--std=c++11', ('--gpu-architecture=compute_%d%d' % (major, minor)).encode())
            status = rtc.nvrtcCompileProgram(program, len(opts), opts)
            if status:
                length = C.c_size_t()
                rtc.nvrtcGetProgramLogSize(program, C.byref(length))
                log = C.create_string_buffer(length.value)
                rtc.nvrtcGetProgramLog(program, log)
                raise RuntimeError('NVRTC: '+log.value.decode(errors='replace'))
            length = C.c_size_t()
            self.check(rtc.nvrtcGetPTXSize(program, C.byref(length)))
            ptx = C.create_string_buffer(length.value)
            self.check(rtc.nvrtcGetPTX(program, ptx))
        finally:
            rtc.nvrtcDestroyProgram(C.byref(program))
        self.driver = C.WinDLL('nvcuda.dll')
        self.driver.cuModuleLoadData.argtypes = [pv, void]
        self.driver.cuModuleGetFunction.argtypes = [pv, void, char]
        self.driver.cuLaunchKernel.argtypes = [void]+[C.c_uint]*7+[void, pv, pv]
        self.module, self.function, self.complete_function = void(), void(), void()
        with torch.cuda.device(device):
            torch.empty(1, device=device)  # establish the primary context
            self.check(self.driver.cuModuleLoadData(C.byref(self.module), ptx))
            self.check(self.driver.cuModuleGetFunction(C.byref(self.function), self.module, b'reconstruct'))
            self.check(self.driver.cuModuleGetFunction(C.byref(self.complete_function), self.module, b'complete_low'))

    @staticmethod
    def check(status):
        if status != 0:
            raise RuntimeError('CUDA/NVRTC API error %d' % status)

    def __call__(self, coeff, anchors, guide, palette, geometry, near, far, keep_on_gpu=False, complete_display=False, display_inputs=None):
        torch = self.torch
        h, w = guide.shape
        lw, lh = geometry.stereo_size
        output = torch.empty((3, h, w), device=self.device, dtype=torch.float32)
        colors = torch.empty((h, w, 3), device=self.device, dtype=torch.uint8)
        stream = C.c_void_p(torch.cuda.current_stream(self.device).cuda_stream)
        display_tensors = []
        if complete_display:
            if display_inputs is None:
                raise ValueError('Display completion requires supported disparities and RGB guides')
            disparity, seeds, low_rgb, rgb = display_inputs
            display_filled = torch.empty((lh, lw), device=self.device, dtype=torch.float32)
            fill_args = [C.c_void_p(a.data_ptr()) for a in (disparity, seeds, low_rgb, display_filled)]
            fill_args += [C.c_int(lw), C.c_int(lh)]
            fill_ptrs = (C.c_void_p*len(fill_args))(*(C.cast(C.byref(a), C.c_void_p) for a in fill_args))
            self.check(self.driver.cuLaunchKernel(self.complete_function, (lw*lh+255)//256, 1, 1,
                256, 1, 1, 0, stream, fill_ptrs, None))
            display_tensors = [*display_inputs, display_filled]
            display_pointers = [C.c_void_p(a.data_ptr()) for a in (display_filled, low_rgb, rgb)]
        else:
            display_pointers = [C.c_void_p(0)]*3
        args = [C.c_void_p(a.data_ptr()) for a in (coeff, anchors, guide, palette, output, colors)]
        args += display_pointers
        args += [C.c_int(v) for v in (w, h, lw, lh)]
        args += [C.c_float(v) for v in (w/lw, geometry.fB, geometry.cx_delta, near, far)]
        args += [C.c_int(int(complete_display))]
        pointers = (C.c_void_p*len(args))(*(C.cast(C.byref(a), C.c_void_p) for a in args))
        self.check(self.driver.cuLaunchKernel(self.function, (w*h+255)//256, 1, 1, 256, 1, 1, 0,
                                              stream, pointers, None))
        self.display_depth = output[2]
        if keep_on_gpu:
            # Uploads used the same stream; record lifetime for the asynchronous
            # kernel so the allocator cannot recycle inputs on another stream.
            for tensor in (coeff, anchors, guide, palette, *display_tensors):
                tensor.record_stream(torch.cuda.current_stream(self.device))
            return output[0], output[1], colors.cpu().numpy()
        # Blocking D2H copy below guarantees allocator lifetime until completion.
        cpu = output.cpu().numpy()
        return cpu[0], cpu[1], colors.cpu().numpy()

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
        self.fused = FusedGuidedKernel(torch, self.device)

    def reconstruct(self, guide, filtered, raw, valid, strict, geometry):
        torch = self.torch
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
