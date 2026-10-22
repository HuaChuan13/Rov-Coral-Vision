"""Run with uv run --locked python test_preview_window.py; never opens a camera."""
import ctypes as C
from ctypes import wintypes as W

import cv2
import numpy as np

from preview_window import PreviewWindow, constrain_rect


def main(size):
    border, current = (16, 39), (1280, 400)
    for edge in range(1, 9):
        for width, height in ((1600, 400), (1280, 700), (100, 50)):
            rect = W.RECT(100, 100, 100+width+border[0], 100+height+border[1])
            anchor = (rect.right if edge in (1, 4, 7) else rect.left,
                      rect.bottom if edge in (3, 4, 5) else rect.top)
            constrain_rect(rect, edge, size, border, current)
            w, h = rect.right-rect.left-border[0], rect.bottom-rect.top-border[1]
            assert abs(w-h*size[0]/size[1]) <= size[0]/size[1]/2
            assert w >= 640 and h >= 200
            assert anchor == (rect.right if edge in (1, 4, 7) else rect.left,
                              rect.bottom if edge in (3, 4, 5) else rect.top)

    window = PreviewWindow('Aspect ratio regression check', size)
    user32 = window.user32
    user32.SendMessageW.argtypes = [W.HWND, W.UINT, W.WPARAM, W.LPARAM]
    user32.SendMessageW.restype = C.c_ssize_t
    user32.SetWindowPos.argtypes = [W.HWND, W.HWND, C.c_int, C.c_int, C.c_int, C.c_int, W.UINT]
    user32.SetWindowPos.restype = W.BOOL
    user32.ShowWindow.argtypes = [W.HWND, C.c_int]
    user32.ShowWindow.restype = W.BOOL
    image = np.full((size[1], size[0], 3), 90, np.uint8)
    cv2.circle(image, (672, 420), 180, (0, 255, 255), 8)
    try:
        window.show(image)
        cv2.waitKey(30)
        for edge in range(1, 9):
            outer, client = W.RECT(), W.RECT()
            assert user32.GetWindowRect(window.hwnd, C.byref(outer))
            assert user32.GetClientRect(window.hwnd, C.byref(client))
            dw = outer.right-outer.left-client.right
            dh = outer.bottom-outer.top-client.bottom
            proposed = W.RECT(outer.left, outer.top, outer.left+1200+dw, outer.top+600+dh)
            assert user32.SendMessageW(window.hwnd, 0x0214, edge, C.addressof(proposed)) == 1
            assert user32.SetWindowPos(window.hwnd, None, proposed.left, proposed.top,
                                      proposed.right-proposed.left, proposed.bottom-proposed.top, 0x0014)
            window.show(image)
            cv2.waitKey(5)
            _, _, w, h = cv2.getWindowImageRect(window.name)
            assert abs(w-h*size[0]/size[1]) <= 2, (edge, w, h)
        user32.ShowWindow(window.hwnd, 3)  # Maximizing uses letterboxing instead of stretching.
        cv2.waitKey(10)
        window.show(image)
        user32.ShowWindow(window.hwnd, 9)
        cv2.waitKey(10)
        assert user32.SendMessageW(window.hwnd, 0x0010, 0, 0) == 0  # Title-bar close.
        cv2.pollKey()
        assert not user32.IsWindow(window.hwnd)
    finally:
        window.close()
        window.close()
    assert image.shape == (size[1], size[0], 3)
    print('PASS: all 8 resize handles, minimum size, native events, maximize and close')


if __name__ == '__main__':
    for size in ((1280, 880), (2688, 880)):
        main(size)
