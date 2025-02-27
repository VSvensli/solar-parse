import numpy as np

def create_circular_kernel(size=9):
    radius = size // 2  
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2  
    kernel = mask.astype(np.uint8) 
    return kernel

