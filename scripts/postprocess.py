import numpy as np
import cv2
from sklearn.cluster import KMeans

def postprocess_saliency_kmeans(saliency_map, num_clusters=2, top_k_components=1):
    """
    Apply KMeans clustering to saliency map for noise reduction.
    This is the default method used in MedCLIP-SAMv2 pipeline.
    
    Args:
        saliency_map: numpy array [H, W] with values in [0, 255] or [0, 1]
        num_clusters: number of clusters (default 2: foreground/background)
        top_k_components: keep only top K largest connected components
    
    Returns:
        cleaned_mask: binary mask [H, W] with values {0, 255}
    """
    # Normalize to [0, 1]
    if saliency_map.max() > 1.0:
        saliency_map = saliency_map / 255.0
    
    h, w = saliency_map.shape
    
    # Resize to 256x256 for faster clustering
    image_resized = cv2.resize(saliency_map, (256, 256), interpolation=cv2.INTER_NEAREST)
    flat_image = image_resized.reshape(-1, 1)
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=10, n_init=10)
    labels = kmeans.fit_predict(flat_image)
    
    # Reshape to 2D
    segmented = labels.reshape(256, 256)
    
    # Identify foreground (higher centroid value)
    centroids = kmeans.cluster_centers_.flatten()
    foreground_cluster = np.argmax(centroids)
    
    # Create binary mask
    binary_mask = (segmented == foreground_cluster).astype(np.uint8)
    
    # Resize back to original size
    binary_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    binary_mask = (binary_mask * 255).astype(np.uint8)
    
    # Connected components filtering (keep only top K largest)
    nb_components, labeled_img, stats, _ = cv2.connectedComponentsWithStats(binary_mask)
    sizes = stats[:, cv2.CC_STAT_AREA]
    
    # Sort by size (ignore background at index 0)
    sorted_sizes = sorted(sizes[1:], reverse=True)
    top_k_sizes = sorted_sizes[:top_k_components]
    
    # Keep only top K components
    result = np.zeros_like(labeled_img, dtype=np.uint8)
    for idx in range(1, nb_components):
        if sizes[idx] in top_k_sizes:
            result[labeled_img == idx] = 255
    
    return result


def postprocess_saliency_threshold(saliency_map, threshold=0.3, top_k_components=1):
    """
    Apply simple thresholding to saliency map.
    
    Args:
        saliency_map: numpy array [H, W] with values in [0, 255] or [0, 1]
        threshold: threshold value (default 0.3)
        top_k_components: keep only top K largest connected components
    
    Returns:
        cleaned_mask: binary mask [H, W] with values {0, 255}
    """
    # Normalize to [0, 1]
    if saliency_map.max() > 1.0:
        saliency_map = saliency_map / 255.0
    
    # Apply threshold
    binary_mask = (saliency_map > threshold).astype(np.uint8) * 255
    
    # Connected components filtering
    nb_components, labeled_img, stats, _ = cv2.connectedComponentsWithStats(binary_mask)
    sizes = stats[:, cv2.CC_STAT_AREA]
    
    sorted_sizes = sorted(sizes[1:], reverse=True)
    top_k_sizes = sorted_sizes[:top_k_components]
    
    result = np.zeros_like(labeled_img, dtype=np.uint8)
    for idx in range(1, nb_components):
        if sizes[idx] in top_k_sizes:
            result[labeled_img == idx] = 255
    
    return result
