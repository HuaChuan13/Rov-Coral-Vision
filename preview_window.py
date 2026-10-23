"""Fixed-aspect OpenCV preview on Windows; recording keeps the original canvas."""
import ctypes as C
from ctypes import wintypes as W
import math
import os

import cv2


def canvas_point(x, y, fitted_rect, size):
    left, top, width, height = fitted_rect
    if not left <= x < left + width or not top <= y < top + height:
        return None
    return ((x - left) * size[0] / width, (y - top) * size[1] / height)


def constrain_rect(rect, edge, size, border, current):
    """Constrain a WM_SIZING outer rectangle, anchoring the opposite edges."""
    ratio = size[0] / size[1]
    width = max(1, rect.right - rect.left - border[0])
    height = max(1, rect.bottom - rect.top - border[1])
    vertical = edge in (3, 6) or (edge in (4, 5, 7, 8) and
        abs(height-current[1]) * ratio > abs(width-current[0]))
    if vertical:
        height = max(math.ceil(640 / ratio), height)
        width = round(height * ratio)
    else:
        width = max(640, width)
        height = round(width / ratio)
    width, height = width + border[0], height + border[1]
    if edge in (1, 4, 7):
        rect.left = rect.right - width
    else:
        rect.right = rect.left + width
    if edge in (3, 4, 5):
        rect.top = rect.bottom - height
    else:
        rect.bottom = rect.top + height


class PreviewWindow:
    def __init__(self, name, size):
        self.name, self.size = name, tuple(size)
        self.click, self.fitted_rect = None, None
        self.user32 = C.WinDLL('user32', use_last_error=True)
        self.comctl32 = C.WinDLL('comctl32', use_last_error=True)
        self.hwnd = None
        callback_type = C.WINFUNCTYPE(C.c_ssize_t, W.HWND, W.UINT, W.WPARAM,
                                     W.LPARAM, C.c_size_t, C.c_size_t)
        self.callback = callback_type(self._message)
        for name, args, result in (
            ('FindWindowW', [W.LPCWSTR, W.LPCWSTR], W.HWND),
            ('GetWindowRect', [W.HWND, C.POINTER(W.RECT)], W.BOOL),
            ('GetClientRect', [W.HWND, C.POINTER(W.RECT)], W.BOOL),
            ('GetWindowThreadProcessId', [W.HWND, C.POINTER(W.DWORD)], W.DWORD),
            ('IsWindow', [W.HWND], W.BOOL),
        ):
            function = getattr(self.user32, name)
            function.argtypes, function.restype = args, result
        self.comctl32.SetWindowSubclass.argtypes = [W.HWND, callback_type, C.c_size_t, C.c_size_t]
        self.comctl32.SetWindowSubclass.restype = W.BOOL
        self.comctl32.RemoveWindowSubclass.argtypes = [W.HWND, callback_type, C.c_size_t]
        self.comctl32.RemoveWindowSubclass.restype = W.BOOL
        self.comctl32.DefSubclassProc.argtypes = [W.HWND, W.UINT, W.WPARAM, W.LPARAM]
        self.comctl32.DefSubclassProc.restype = C.c_ssize_t
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        try:
            scale = min(1., 1613 / size[0], 900 / size[1])
            cv2.resizeWindow(self.name, round(size[0]*scale), round(size[1]*scale))
            self.hwnd = self.user32.FindWindowW(None, self.name)
            process = W.DWORD()
            thread = self.user32.GetWindowThreadProcessId(self.hwnd, C.byref(process))
            kernel32 = C.WinDLL('kernel32')
            kernel32.GetCurrentThreadId.argtypes = []
            kernel32.GetCurrentThreadId.restype = W.DWORD
            if process.value != os.getpid() or thread != kernel32.GetCurrentThreadId():
                raise RuntimeError('Cannot find the OpenCV window on the current thread.')
            if not self.comctl32.SetWindowSubclass(self.hwnd, self.callback, 1, 0):
                raise RuntimeError('Cannot install fixed-aspect window resizing.')
            cv2.setMouseCallback(self.name, self._mouse)
        except BaseException:
            cv2.destroyWindow(self.name)
            raise

    def _message(self, hwnd, message, edge, address, subclass_id, reference):
        if message == 0x0214 and address:  # WM_SIZING: includes all four edges/corners.
            outer, client = W.RECT(), W.RECT()
            if (self.user32.GetWindowRect(hwnd, C.byref(outer)) and
                    self.user32.GetClientRect(hwnd, C.byref(client))):
                current = (client.right-client.left, client.bottom-client.top)
                border = (outer.right-outer.left-current[0], outer.bottom-outer.top-current[1])
                rect = C.cast(address, C.POINTER(W.RECT)).contents
                constrain_rect(rect, edge, self.size, border, current)
                return 1
        elif message == 0x0082:  # WM_NCDESTROY: detach before OpenCV destroys the HWND.
            self.comctl32.RemoveWindowSubclass(hwnd, self.callback, subclass_id)
        return self.comctl32.DefSubclassProc(hwnd, message, edge, address)

    def _mouse(self, event, x, y, flags, userdata):
        if event == cv2.EVENT_RBUTTONDOWN:
            self.click = ("cancel", None)
        elif event == cv2.EVENT_LBUTTONDOWN and self.fitted_rect:
            point = canvas_point(x, y, self.fitted_rect, self.size)
            if point is not None:
                self.click = ("select", point)

    def pop_click(self):
        click, self.click = self.click, None
        return click

    def show(self, image):
        if image.shape[1::-1] != self.size:
            raise ValueError('Preview canvas dimensions changed.')
        _, _, width, height = cv2.getWindowImageRect(self.name)
        if width < 1 or height < 1:
            return
        scale = min(width/self.size[0], height/self.size[1])
        fitted = (min(width, max(1, round(self.size[0]*scale))),
                  min(height, max(1, round(self.size[1]*scale))))
        view = cv2.resize(image, fitted, interpolation=cv2.INTER_LINEAR) if fitted != self.size else image
        x, y = (width-fitted[0])//2, (height-fitted[1])//2
        self.fitted_rect = (x, y, *fitted)
        if fitted != (width, height):
            view = cv2.copyMakeBorder(view, y, height-fitted[1]-y,
                                     x, width-fitted[0]-x, cv2.BORDER_CONSTANT, value=(25, 25, 25))
        cv2.imshow(self.name, view)

    def close(self):
        if self.hwnd and self.user32.IsWindow(self.hwnd):
            self.comctl32.RemoveWindowSubclass(self.hwnd, self.callback, 1)
            cv2.destroyWindow(self.name)
        self.hwnd = None
