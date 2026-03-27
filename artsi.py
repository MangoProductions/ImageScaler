import os
import cv2
import numpy as np
from multiprocessing import Pool
from typing import Optional

# Constants
INPUT_FOLDER = "input_images"
OUTPUT_FOLDER = "output_images"

#Make these values adjustable later on
JPEG_QUALITY = 95
UPSCALE_FACTOR = 3.0     # how much image blows up before processing
FINAL_SCALE = 1.5        # final size relative to cropped base
MAX_SIZE = 10000          # I fear for my pc.

def edge_aware_sharpen(image: np.ndarray, strength: float = 0.5, radius: int = 2) -> np.ndarray:
    #Attempt tp imitate FSR1, pwease.
    if image is None or image.size == 0:
        print("edge_aware_sharpen: input image is empty!")
        return image
    try:
        img = image.astype(np.float32) / 255.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        edges = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        edges = np.abs(edges)
        edges = cv2.GaussianBlur(edges, (0, 0), sigmaX=radius)
        edges = cv2.normalize(edges, None, 0, 1, cv2.NORM_MINMAX)
        
        #???
        contrast = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        contrast = np.abs(contrast)
        contrast = cv2.GaussianBlur(contrast, (0, 0), sigmaX=radius)
        contrast = cv2.normalize(contrast, None, 0, 1, cv2.NORM_MINMAX)
        #What was the idea?

        kernel = np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=np.float32) / 8.0

        sharpened = cv2.filter2D(img, -1, kernel)
        sharpened = np.clip(sharpened, 0, 1)

        sharpen_strength = edges * contrast * strength
        sharpen_strength = np.clip(sharpen_strength, 0, 1)[:, :, None]

        result = img * (1 - sharpen_strength) + sharpened * sharpen_strength
        result = np.clip(result, 0, 1)
        return (result * 255).astype(np.uint8)
    except Exception as e:
        print(f"edge_aware_sharpen error: {e}")
        return image







def add_film_grain_gradient(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        print("add_film_grain_gradient: input image is empty!")
        return image
    try:
        # Film Grain meh.
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        brightness_map = cv2.GaussianBlur(gray, (0, 0), 25)
        brightness_norm = brightness_map / 255.0
        large_w = np.clip((122 - brightness_map) / 122.0, 0, 1)
        medium_w = np.clip((brightness_map - 122) / (255 - 122), 0, 1) * (1 - large_w)
        small_w = 1.0 - large_w - medium_w
        large_noise = np.random.normal(0, 1.0, (h//16, w//16, 3))
        med_noise = np.random.normal(0, 0.5, (h//32, w//32, 3))
        small_noise = np.random.normal(0, 0.25, (h//48, w//48, 3))
        large_noise = cv2.resize(large_noise, (w, h), interpolation=cv2.INTER_LINEAR)
        med_noise = cv2.resize(med_noise, (w, h), interpolation=cv2.INTER_LINEAR)
        small_noise = cv2.resize(small_noise, (w, h), interpolation=cv2.INTER_LINEAR)
        grain = (
            large_noise * large_w[..., None] +
            med_noise * medium_w[..., None] +
            small_noise * small_w[..., None]
        )
        edges = cv2.Laplacian(gray, cv2.CV_32F)
        edges = np.abs(edges)
        edges = cv2.GaussianBlur(edges, (0, 0), 7)
        edges_norm = cv2.normalize(edges, None, 0, 1.0, cv2.NORM_MINMAX)
        grain_strength = 1.0 - edges_norm[..., None]
        grain *= grain_strength
        color_shifts = np.random.randint(-3, 4, (h//4, w//4, 3))
        color_shifts = cv2.resize(color_shifts, (w, h), interpolation=cv2.INTER_NEAREST)
        grain += color_shifts
        grain = cv2.GaussianBlur(grain, (0, 0), 0.8)
        output = image.astype(np.float32) + grain
        output = np.clip(output, 0, 255).astype(np.uint8)
        return output
    except Exception as e:
        print(f"add_film_grain_gradient error: {e}")
        return image







def generate_bloom_layer(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        print("generate_bloom_layer: input image is empty!")
        return image
    try:
        #Bloom my beloved
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
    except Exception as e:
        print(f"generate_bloom_layer error: {e}")
        return image









def apply_chromatic_aberration(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        print("apply_chromatic_aberration: input image is empty!")
        return image
    try:
        # This s maybe good??? It easily gets too strong, lo key afraid to adjust it too much.
        h, w = image.shape[:2]
        cx, cy = w / 2, h / 2
        y, x = np.indices((h, w))
        x = x - cx
        y = y - cy
        r = np.sqrt(x*x + y*y)
        r_norm = r / r.max()
        strength = (r_norm ** 2.2)
        strength *= (1.0 + 0.12 * (x / w))
        rb_shift_x = x * strength * 0.0028
        rb_shift_y = y * strength * 0.0028
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
    except Exception as e:
        print(f"apply_chromatic_aberration error: {e}")
        return image










def add_thorium_speks(image: np.ndarray, count: int = 15) -> np.ndarray:
    if image is None or image.size == 0:
        print("add_thorium_speks: input image is empty!")
        return image
    try:
        # Not happy with this code.
        img = image.copy()
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        prob = 1.0 - (gray / 255.0)
        prob = prob / prob.sum()
        for _ in range(count):
            flat_index = np.random.choice(h * w, p=prob.flatten())
            y, x = divmod(flat_index, w)
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
            if np.random.rand() < 0.3:
                if x+1 < w:
                    img[y, x+1] = color
                if y+1 < h:
                    img[y+1, x] = color
        return img
    except Exception as e:
        print(f"add_thorium_speks error: {e}")
        return image











def process_image(args: tuple) -> Optional[str]:
    filename, idx = args
    try:
        png_path = os.path.join(INPUT_FOLDER, filename)
        img = cv2.imread(png_path)
        if img is None:
            print(f"Failed to load {filename}: file not found or not an image.")
            return None

        h, w = img.shape[:2]

        # Crop to centered square (smallest dimension)
        size = min(h, w)
        top = (h - size) // 2
        left = (w - size) // 2
        img = img[top:top + size, left:left + size]

        img = apply_chromatic_aberration(img)

        base_h, base_w = img.shape[:2]

        upscaled_size = (
            int(base_w * UPSCALE_FACTOR),
            int(base_h * UPSCALE_FACTOR)
        )

        upscaled = cv2.resize(img, upscaled_size, interpolation=cv2.INTER_LANCZOS4)

        sharpened = edge_aware_sharpen(upscaled, strength=0.6, radius=2)

        bloom_layer = generate_bloom_layer(sharpened)

        #BLOOM ENCORE GO
        bloom_layer = apply_chromatic_aberration(bloom_layer)

        bloomed = cv2.add(sharpened, bloom_layer)

        grained = add_film_grain_gradient(bloomed)

        final_size = (
            int(base_w * FINAL_SCALE),
            int(base_h * FINAL_SCALE)
        )

        final = cv2.resize(grained, final_size, interpolation=cv2.INTER_AREA)

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

        new_name = f"photo{idx+1:03d}.jpg"
        jpg_path = os.path.join(OUTPUT_FOLDER, new_name)
        cv2.imwrite(
            jpg_path,
            final,
            [
                int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY,
                int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1,
                int(cv2.IMWRITE_JPEG_OPTIMIZE), 1
            ]
        )
        return filename
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None










if __name__ == "__main__":
    import sys
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    filenames = sorted([f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(".png")])
    total = len(filenames)

    def update_progress(result):
        pbar.progress += 1
        # Every 3 dots becomes an _
        dots = "." * pbar.progress
        display_dots = []
        for i in range(0, len(dots), 3):
            if i + 3 <= len(dots):
                display_dots.append("_")
            else:
                display_dots.append(dots[i:])
        display_dots = "".join(display_dots)
        display_dots = display_dots.ljust(pbar.width)
        pbar.bar = f"[{display_dots}]"
        print(f"\r{pbar.bar} {pbar.progress}/{total} images", end="", flush=True)

    class ProgressBar:
        def __init__(self, total):
            self.progress = 0
            self.total = total
            self.width = total  # width = number of images
            self.bar = f"[{' ' * self.width}]"

    pbar = ProgressBar(total)
    print(f"\r{pbar.bar} 0/{total} images", end="", flush=True)

    with Pool(processes=2) as pool:
        for _ in pool.imap_unordered(process_image, [(fn, i) for i, fn in enumerate(filenames)]):
            update_progress(_)

    # When done, fill with ¤
    done_bar = "[" + "¤" * pbar.width + "]"
    print(f"\r{done_bar} {pbar.progress}/{total} images")
    print(f"\nProcessed {total} images using 2 cores.")
    print("Done.")
