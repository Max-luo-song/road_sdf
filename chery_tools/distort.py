import cv2
import numpy as np

cache_index = {}


def distort_image(
    image,
    intrinsics,
    kb_coeffs,
    new_size=None,
    enable_cache=True,
):
    h, w = image.shape[:2]
    if new_size is None:
        new_size = (w, h)
    new_w, new_h = new_size

    if len(kb_coeffs) < 8:
        kb_coeffs = np.concatenate([kb_coeffs, np.zeros(8 - len(kb_coeffs))])

    cache_key = (h, w, new_w, new_h, tuple(kb_coeffs))

    if enable_cache and cache_key in cache_index:
        mapx, mapy = cache_index[cache_key]
    else:
        # 目标图像的像素坐标网格
        x_out, y_out = np.meshgrid(np.arange(new_w), np.arange(new_h))

        cx, cy = new_w / 2.0, new_h / 2.0
        x_d = (x_out - cx) / intrinsics[0, 0]  # fx
        y_d = (y_out - cy) / intrinsics[1, 1]  # fy

        r_d_sq = x_d**2 + y_d**2
        r_d = np.sqrt(r_d_sq + 1e-10)

        # 径向畸变
        k1, k2, p1, p2, k3, k4, k5, k6 = kb_coeffs[:8]
        r_u = r_d * (
            1
            + k1 * r_d_sq
            + k2 * r_d_sq**2
            + k3 * r_d_sq**3
            + k4 * r_d_sq**4
            + k5 * r_d_sq**5
            + k6 * r_d_sq**6
        )

        # 切向畸变
        x_u = x_d + (2 * p1 * y_d + p2 * (r_d_sq + 2 * x_d**2))
        y_u = y_d + (p1 * (r_d_sq + 2 * y_d**2) + 2 * p2 * x_d)

        # 结合径向和切向畸变
        scale = np.where(r_d > 1e-10, r_u / r_d, 1)
        x_u *= scale
        y_u *= scale

        # 转换回像素坐标
        mapx = x_u * intrinsics[0, 0] + cx
        mapy = y_u * intrinsics[1, 1] + cy

        if enable_cache:
            cache_index[cache_key] = (mapx, mapy)

    # 设置边界为黑色
    dst = cv2.remap(
        image,
        mapx.astype(np.float32),
        mapy.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return dst