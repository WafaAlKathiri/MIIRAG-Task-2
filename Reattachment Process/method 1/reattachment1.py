import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
from skimage.exposure import match_histograms


# ------------------ STEP 1: Find Best Road Match ------------------
def compute_color_texture_similarity(crack_path, road_path):
    """
    Computes a combined color + texture similarity score between a crack and a road image.
    Uses Lab color histograms for color similarity and SSIM for texture similarity.
    """
    crack_bgr = cv2.imread(crack_path)
    road_bgr = cv2.imread(road_path)

    if crack_bgr is None or road_bgr is None:
        print(f"Error loading: {crack_path} or {road_path}")
        return -1  

    # Resize both images to 256x256
    crack_bgr = cv2.resize(crack_bgr, (256, 256))
    road_bgr = cv2.resize(road_bgr, (256, 256))

    # Convert to Lab color space
    crack_lab = cv2.cvtColor(crack_bgr, cv2.COLOR_BGR2LAB)
    road_lab = cv2.cvtColor(road_bgr, cv2.COLOR_BGR2LAB)

    # Split channels
    crack_L, crack_A, crack_B = cv2.split(crack_lab)
    road_L,  road_A,  road_B  = cv2.split(road_lab)

    # Color histogram correlation
    def hist_correlation(ch1, ch2):
        hist1 = cv2.calcHist([ch1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([ch2], [0], None, [256], [0, 256])
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    avg_color_corr = (hist_correlation(crack_L, road_L) +
                      hist_correlation(crack_A, road_A) +
                      hist_correlation(crack_B, road_B)) / 3.0

    # Texture similarity (SSIM) on L channel
    ssim_score, _ = ssim(crack_L, road_L, full=True)

    # Combine color and texture scores
    final_score = 0.7 * avg_color_corr + 0.3 * ssim_score
    return final_score


def find_best_matching_road_for_cracks(crack_folder, road_folder):
    """
    Finds the best matching road image for each crack image.
    Returns a dictionary {crack_path: best_road_path}
    """
    results = {}

    for crack_image in os.listdir(crack_folder):
        crack_path = os.path.join(crack_folder, crack_image)
        best_match = None
        best_score = -1  

        for road_image in os.listdir(road_folder):
            road_path = os.path.join(road_folder, road_image)
            score = compute_color_texture_similarity(crack_path, road_path)

            if score > best_score:
                best_score = score
                best_match = road_path

        results[crack_path] = best_match
        print(f"[{crack_image}] -> Best match: {best_match} (Score: {best_score:.4f})")

    return results


# ------------------ STEP 2: Place Crack on Road ------------------
def find_centered_empty_region(mask, crack_shape, min_distance=50):
    """
    Finds the best empty (black) region in the mask to place the crack.
    Uses distance transform to prioritize the center while keeping distance from existing cracks.
    """
    h_mask, w_mask = mask.shape
    h_crack, w_crack = crack_shape

    # Define the center of the mask
    center_y, center_x = h_mask // 2, w_mask // 2

    # Distance transform: Calculates the distance to the nearest white pixel (crack)
    dist_transform = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)
    
    empty_regions = []

    # Scan for empty regions
    for y in range(h_mask - h_crack):
        for x in range(w_mask - w_crack):
            # Check if the region is empty (all black)
            if np.all(mask[y:y + h_crack, x:x + w_crack] == 0):
                # Calculate Euclidean distance from the center
                center_distance = np.sqrt((center_y - (y + h_crack // 2)) ** 2 + (center_x - (x + w_crack // 2)) ** 2)
                
                # Minimum distance to the nearest crack
                min_dist_to_crack = np.min(dist_transform[y:y + h_crack, x:x + w_crack])

                if min_dist_to_crack > min_distance:
                    empty_regions.append((center_distance, min_dist_to_crack, x, y))

    if not empty_regions:
        return None

    # Sort regions by proximity to center
    empty_regions.sort(key=lambda region: region[0])

    _, _, best_x, best_y = empty_regions[0]
    return best_x, best_y

def match_colors(source_image, target_image):
    """Match the colors of the source image to the target image using histogram matching."""
    return match_histograms(source_image, target_image, channel_axis=-1)

def adjust_lighting(source_image, target_image):
    """Adjust the lighting of the source image to match the target image."""
    source_lab = cv2.cvtColor(source_image, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target_image, cv2.COLOR_BGR2LAB)
    
    source_l, source_a, source_b = cv2.split(source_lab)
    target_l, _, _ = cv2.split(target_lab)
    
    source_l = np.clip(source_l * (target_l.mean() / source_l.mean()), 0, 255).astype(np.uint8)
    
    adjusted_lab = cv2.merge([source_l, source_a, source_b])
    return cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2BGR)

def add_noise(image, intensity=0.05):
    """Add noise to make the crack blend more naturally."""
    noise = np.random.randn(*image.shape) * intensity * 255
    return np.clip(image + noise, 0, 255).astype(np.uint8)

def poisson_blend(background, overlay, mask, x, y):
    """Blend the crack image onto the road using Poisson blending."""
    return cv2.seamlessClone(overlay, background, mask, (x + overlay.shape[1] // 2, y + overlay.shape[0] // 2), cv2.NORMAL_CLONE)

def place_crack_on_road(road_img, crack_img, mask, min_distance=50):
    """
    Enhances and places a crack image onto a road image using advanced blending techniques.
    """
    h_crack, w_crack = crack_img.shape[:2]

    # Find the best empty region for placement
    position = find_centered_empty_region(mask, (h_crack, w_crack), min_distance)

    if position is None:
        print("No suitable empty region found.")
        return road_img  

    x, y = position
    print(f"Placing crack at: ({x}, {y})")

    # Step 1: Enhance crack appearance
    crack_img = match_colors(crack_img, road_img)
    crack_img = adjust_lighting(crack_img, road_img)
    crack_img = add_noise(crack_img, intensity=0.05)

    # Step 2: Create a blending mask
    blend_mask = 255 * np.ones(crack_img.shape, crack_img.dtype)

    # Step 3: Use Poisson blending
    result = poisson_blend(road_img, crack_img, blend_mask, x, y)

    return result


# ------------------ STEP 3: Process All Images ------------------
def process_all_cracks(crack_folder, road_folder, mask_folder, output_folder):
    """
    Processes all crack images, finds the best matching road, places the crack, and saves the result.
    """
    os.makedirs(output_folder, exist_ok=True)

    best_matches = find_best_matching_road_for_cracks(crack_folder, road_folder)

    for crack_path, road_path in best_matches.items():
        if not road_path:
            print(f"Skipping {crack_path}: No matching road found.")
            continue

        crack_filename = os.path.basename(crack_path)
        road_filename = os.path.basename(road_path)
        mask_path = os.path.join(mask_folder, road_filename)

        # Load images
        road_img = cv2.imread(road_path)
        crack_img = cv2.imread(crack_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if road_img is None or crack_img is None or mask is None:
            print(f"Error loading images for {crack_filename}")
            continue

        # Resize crack image if needed
        scale_factor = 1.3  # Increase size by 30%
        h_crack, w_crack = crack_img.shape[:2]
        new_size = (int(w_crack * scale_factor), int(h_crack * scale_factor))
        crack_img = cv2.resize(crack_img, new_size, interpolation=cv2.INTER_LINEAR)     

        # Place the crack
        result_img = place_crack_on_road(road_img, crack_img, mask, min_distance=50)
        result_img = np.clip(result_img, 0, 255).astype(np.uint8)

        # Save output
        output_path = os.path.join(output_folder, f"cracked_{road_filename}")
        cv2.imwrite(output_path, result_img)
        print(f"Saved: {output_path}")


# ------------------ Run the Pipeline ------------------
if __name__ == "__main__":
    crack_folder = r"C:\Users\NITRO\Desktop\ramadan\cracks - Copy"
    road_folder = r"C:\Users\NITRO\Desktop\ramadan\road_images\images"
    mask_folder = r"C:\Users\NITRO\Desktop\ramadan\road_images\masks"
    output_folder = r"C:\Users\NITRO\Desktop\ramadan\15output"

    process_all_cracks(crack_folder, road_folder, mask_folder, output_folder)
