import numpy as np

cache_index = {}
cache_boundary = {}


def pinhole2fisheye(image, focal_length, kb_coeffs, crop_valid, enable_cache=True):
    h, w = image.shape[:2]
    x = np.arange(w)
    y = np.arange(h)
    x, y = np.meshgrid(x, y)

    cache_key = (h, w, focal_length)
    if enable_cache and cache_key in cache_index:
        yf, xf = cache_index[cache_key]
        y_min, y_max, x_min, x_max = cache_boundary[cache_key]
    else:
        cx, cy = w / 2, h / 2
        dx = x - cx
        dy = y - cy

        rc = np.sqrt(dx ** 2 + dy ** 2)  # 每个像素到图像中点的距离
        theta = np.arctan2(rc, focal_length)  # 像素-光心-图像中点的角度
        gamma = np.arctan2(dy, dx)  # 像素-图像中间的2D角度,该角度不变

        # kb模型
        # HACK: kb_coeffs 只能有 4 个系数
        d0, d1, d2, d3 = kb_coeffs
        rf = theta * (1 + d0 * theta**2 + d1 * theta**4 + d2 * theta**6 + d3 * theta**8) * focal_length

        xf = rf * np.cos(gamma)
        yf = rf * np.sin(gamma)

        xf = xf + cx
        yf = yf + cy

        xf = np.clip(xf.astype(np.int32), 0, w - 1)
        yf = np.clip(yf.astype(np.int32), 0, h - 1)

        y_min, y_max, x_min, x_max = yf.min(), yf.max(), xf.min(), xf.max()

        if enable_cache:
            cache_index[cache_key] = (yf, xf)
            cache_boundary[cache_key] = (y_min, y_max, x_min, x_max)

    dst_arr = np.zeros(image.shape, dtype=image.dtype)
    dst_arr[yf, xf] = image[y, x]
    return dst_arr[y_min:y_max, x_min:x_max] if crop_valid else dst_arr
