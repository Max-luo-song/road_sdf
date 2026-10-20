import numpy as np
import matplotlib.cm as cm

def visualize_depth_numpy(depth: np.ndarray, minmax=None, cmap='jet'):
    """
    depth: (H, W)
    """    
    x = np.nan_to_num(depth) # change nan to 0
    if minmax is None:
        mi = np.min(x[x>0]) # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi,ma = minmax
    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1

    colormap = cm.get_cmap(cmap)
    x_ = colormap(x)
    x_ = (x_[:, :, :3] * 255).astype(np.uint8)
    return x_, [mi,ma]
