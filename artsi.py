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

        #Fixed the Contrast
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=radius * 2)
        contrast = np.abs(gray - blur)
        contrast = cv2.normalize(contrast, None, 0, 1, cv2.NORM_MINMAX)

        kernel = np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=np.float32) /2

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


def add_grain(image: np.ndarray, grain_alpha: float = 0.3) -> np.ndarray:
    if image is None or image.size == 0:
        return image

    h, w = image.shape[:2]
    img = image.astype(np.float32)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lum = gray / 255.0
    mid = 0.5

    w_dark = np.clip(1.0 - lum / mid, 0, 1) ** 1.5
    w_bright = np.clip((lum - mid) / (1.0 - mid), 0, 1) ** 1.5
    w_mid = (1.0 - w_dark) * (1.0 - w_bright)

    large = np.random.randn(h, w)
    large = cv2.GaussianBlur(large, (0, 0), 2.5)
    small = cv2.resize(large, (w//12, h//12), interpolation=cv2.INTER_LINEAR)
    large = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    large = np.repeat(large[:, :, None], 3, axis=2)

    medium = np.random.randn(h, w)
    medium = cv2.GaussianBlur(medium, (0, 0), 1.0)
    tmp = cv2.resize(medium, (w//3, h//3), interpolation=cv2.INTER_LINEAR)
    medium = cv2.resize(tmp, (w, h), interpolation=cv2.INTER_LINEAR)
    medium = np.repeat(medium[:, :, None], 3, axis=2)

    fine = np.random.randn(h, w)
    fine = fine - cv2.GaussianBlur(fine, (0, 0), 0.3)
    fine = np.repeat(fine[:, :, None], 3, axis=2)

    grain = (
        large * w_dark[..., None] * 12.0 +
        medium * w_mid[..., None] * 8.0 +
        fine * w_bright[..., None] * 6.0
    )

    r_noise = np.random.randn(h, w) * 25.0
    g_noise = np.random.randn(h, w) * 25.0
    b_noise = np.random.randn(h, w) * 25.0
    r_noise *= w_dark + 0.3 * w_mid + 0.1 * w_bright
    g_noise *= w_dark + 0.3 * w_mid + 0.1 * w_bright
    b_noise *= w_dark + 0.3 * w_mid + 0.1 * w_bright

    color_grain = np.stack([b_noise, g_noise, r_noise], axis=2)
    grain += color_grain

    specks = (np.random.rand(h, w) > 0.995).astype(np.float32)
    color_specks = np.random.randint(-50, 50, (h, w, 3)).astype(np.float32)
    grain += color_specks * specks[..., None]

    grain -= np.mean(grain, axis=(0,1), keepdims=True)
    grain /= (np.std(grain, axis=(0,1), keepdims=True) + 1e-6)

    out = img + grain * grain_alpha * 50.0  # 50.0 = visual scaling factor

    
    out = (out - 128) * 1.01 + 128

    return np.clip(out, 0, 255).astype(np.uint8)


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



def reduce_noise_bilateral(image: np.ndarray, diameter: int = 9, sigmaColor: float = 75, sigmaSpace: float = 75) -> np.ndarray:
    #This reduces noise
    #Why would I do this?
    #I like the contradictory state.
    if image is None or image.size == 0:
        return image
    return cv2.bilateralFilter(image, diameter, sigmaColor, sigmaSpace)





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
        # Not happy with this spek code.
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


        bloom_layer = generate_bloom_layer(upscaled)

        #BLOOM ENCORE GO
        bloom_layer = apply_chromatic_aberration(bloom_layer)

        bloomed = cv2.add(upscaled, bloom_layer)


        final_size = (
            int(base_w * FINAL_SCALE),
            int(base_h * FINAL_SCALE)
        )

        final = cv2.resize(bloomed, final_size, interpolation=cv2.INTER_AREA)
        final = add_grain(final, grain_alpha=0.1)
        final = reduce_noise_bilateral(final, diameter=9, sigmaColor=25, sigmaSpace=25)
        final = edge_aware_sharpen(final, strength=4, radius=1)
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
