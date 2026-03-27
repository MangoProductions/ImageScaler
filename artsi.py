import os
import cv2
import numpy as np
from multiprocessing import Pool

input_folder = "input_images"
output_folder = "output_images"
os.makedirs(output_folder, exist_ok=True)

kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
], dtype=np.float32)

def add_film_grain_gradient(image):
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    brightness_map = cv2.GaussianBlur(gray, (0, 0), 25)

    brightness_norm = brightness_map / 255.0

    # Create grain size weights (large, medium, small)
    large_w = np.clip((122 - brightness_map) / 122.0, 0, 1)
    medium_w = np.clip((brightness_map - 122) / (255 - 122), 0, 1) * (1 - large_w)
    small_w = 1.0 - large_w - medium_w

    # Three sizes
    large_noise = np.random.normal(0, 1.0, (h//16, w//16, 3))
    med_noise   = np.random.normal(0, 0.5, (h//32, w//32, 3))
    small_noise = np.random.normal(0, 0.25, (h//48, w//48, 3))

    # Upsample each noise to full size
    large_noise = cv2.resize(large_noise, (w, h), interpolation=cv2.INTER_LINEAR)
    med_noise = cv2.resize(med_noise, (w, h), interpolation=cv2.INTER_LINEAR)
    small_noise = cv2.resize(small_noise, (w, h), interpolation=cv2.INTER_LINEAR)

    # Blend noise according to weights
    grain = (
        large_noise * large_w[..., None] +
        med_noise * medium_w[..., None] +
        small_noise * small_w[..., None]
    )

    # Contrast-aware suppression
    edges = cv2.Laplacian(gray, cv2.CV_32F)
    edges = np.abs(edges)
    edges = cv2.GaussianBlur(edges, (0, 0), 7)
    edges_norm = cv2.normalize(edges, None, 0, 1.0, cv2.NORM_MINMAX)
    grain_strength = 1.0 - edges_norm[..., None]
    grain *= grain_strength

    # Add color shifts
    color_shifts = np.random.randint(-3, 4, (h//4, w//4, 3))
    color_shifts = cv2.resize(color_shifts, (w, h), interpolation=cv2.INTER_NEAREST)
    grain += color_shifts

    # Blur sliiightly
    grain = cv2.GaussianBlur(grain, (0, 0), 0.8)

    output = image.astype(np.float32) + grain
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output
def generate_bloom_layer(image):
    img = image.astype(np.float32) / 255.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.0
    mask = np.clip((gray - 0.75) / 0.25, 0, 1)

    highlights = img * mask[:, :, None]

    bloom = cv2.GaussianBlur(highlights, (0, 0), 12)
    bloom += cv2.GaussianBlur(highlights, (0, 0), 24) * 0.5

    bloom[:, :, 2] *= 1.1
    bloom[:, :, 1] *= 1.05

    bloom = np.clip(bloom, 0, 1)
    return (bloom * 255).astype(np.uint8)
def apply_chromatic_aberration(image):
    h, w = image.shape[:2]
    cx, cy = w / 2, h / 2

    y, x = np.indices((h, w))
    x = x - cx
    y = y - cy

    r = np.sqrt(x*x + y*y)
    r_norm = r / r.max()

    # Stronger edge effect
    strength = (r_norm ** 2.2)

    # Slight ellipse distortion (imperfect lens)
    strength *= (1.0 + 0.12 * (x / w))

    # Magenta (R+B) outward
    rb_shift_x = x * strength * 0.0028
    rb_shift_y = y * strength * 0.0028

    # Green inward
    g_shift_x = -x * strength * 0.0018
    g_shift_y = -y * strength * 0.0018

    map_x = (x + cx).astype(np.float32)
    map_y = (y + cy).astype(np.float32)

    rb_map_x = (map_x + rb_shift_x).astype(np.float32)
    rb_map_y = (map_y + rb_shift_y).astype(np.float32)

    g_map_x = (map_x + g_shift_x).astype(np.float32)
    g_map_y = (map_y + g_shift_y).astype(np.float32)

    b, g, r = cv2.split(image)

    r_shifted = cv2.remap(r, rb_map_x, rb_map_y, cv2.INTER_LINEAR)
    b_shifted = cv2.remap(b, rb_map_x, rb_map_y, cv2.INTER_LINEAR)
    g_shifted = cv2.remap(g, g_map_x, g_map_y, cv2.INTER_LINEAR)

    result = cv2.merge([b_shifted, g_shifted, r_shifted])
    return result
def add_thorium_speks(image, count=15):
    img = image.copy()
    h, w = img.shape[:2]

    # Bias the speks toward darker areas
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    prob = 1.0 - (gray / 255.0)
    prob = prob / prob.sum()

    for _ in range(count):
        flat_index = np.random.choice(h * w, p=prob.flatten())
        y, x = divmod(flat_index, w)

        # Painfully bright speks
        value = np.random.randint(220, 256)

        if np.random.rand() < 0.2:
            color = [
                np.random.randint(200, 256),
                np.random.randint(200, 256),
                np.random.randint(200, 256)
            ]
        else:
            color = [value, value, value]

        img[y, x] = color

        # Attempt a cluster
        if np.random.rand() < 0.3:
            if x+1 < w:
                img[y, x+1] = color
            if y+1 < h:
                img[y+1, x] = color

    return img
def process_image(args):
    filename, idx = args
    try:
        png_path = os.path.join(input_folder, filename)
        img = cv2.imread(png_path)
        if img is None:
            print(f"Failed to load {filename}")
            return None

        # Center crop 1440x1440
        h, w = img.shape[:2]
        left = (w - 1440) // 2
        right = left + 1440
        img = img[0:1440, left:right]

        img = apply_chromatic_aberration(img)

        # Upscale
        upscaled = cv2.resize(img, (4320, 4320), interpolation=cv2.INTER_LANCZOS4)

        # RCAS-like sharpen
        blur = cv2.GaussianBlur(upscaled, (0, 0), 1.0)
        sharpened = cv2.addWeighted(upscaled, 1.08, blur, -0.08, 0)
        bloom_layer = generate_bloom_layer(sharpened)
        bloom_layer = apply_chromatic_aberration(bloom_layer)
        bloomed = cv2.add(sharpened, bloom_layer)
        # Grain BEFOORE downscale
        grained = add_film_grain_gradient(bloomed)

        # Downscale my beloved
        final = cv2.resize(grained, (2160, 2160), interpolation=cv2.INTER_AREA)

        # An otter sharpening
        kernel_sharp = cv2.filter2D(final, -1, kernel)
        final = cv2.addWeighted(final, 0.5, kernel_sharp, 0.5, 0)

        # Contrast-Aware Smoothing gang rise up
        gray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
        edges = cv2.Laplacian(gray, cv2.CV_32F)
        edges = np.abs(edges)
        edges = cv2.GaussianBlur(edges, (0, 0), 3)
        edges_norm = cv2.normalize(edges, None, 0, 1.0, cv2.NORM_MINMAX)

        mask = 1.0 - edges_norm
        mask = np.clip(mask, 0, 1)

        blurred = cv2.GaussianBlur(final, (0, 0), 0.5)
        smooth_factor = 0.5
        final_smooth = final.astype(np.float32) * (1 - mask * smooth_factor)[:, :, None] + blurred.astype(np.float32) * (mask * smooth_factor)[:, :, None]
        final_smooth = np.clip(final_smooth, 0, 255).astype(np.uint8)

        final = final_smooth
        final = add_thorium_speks(final, count=12)

        # Save JPEG with sequential number
        new_name = f"photo{idx+1:03d}.jpg"
        jpg_path = os.path.join(output_folder, new_name)
        cv2.imwrite(
        jpg_path,
        final,
        [
            int(cv2.IMWRITE_JPEG_QUALITY), 95,
            int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1,
            int(cv2.IMWRITE_JPEG_OPTIMIZE), 1 #You bastard work please
        ]
        )
        return filename
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

if __name__ == "__main__":
    # Get sorted list of PNG files
    filenames = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(".png")])
    # Double Core Fun Time
    with Pool(processes=2) as pool:
        pool.map(process_image, [(fn, i) for i, fn in enumerate(filenames)])
    #I wish I could memorize out of memory how to do multithreading.
    print(f"Processed {len(filenames)} images using 2 cores.")
    print("Kthxbye")
