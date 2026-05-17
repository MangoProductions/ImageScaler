import os
import cv2
import numpy as np
import time

from multiprocessing import Pool
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass(frozen=True)
class EffectsConfig:
    # Chromatic Aberration
    chromatic_aberration_strength: float = 0.002
    ca_shift: float = 1.5

    # Bloom
    bloom_threshold: float = 0.7
    bloom_scale: float = 0.3
    bloom_sigma_small: float = 8
    bloom_sigma_large: float = 16
    bloom_large_weight: float = 0.5
    bloom_highlight_threshold: float = 0.75
    bloom_highlight_range: float = 0.25
    bloom_red_boost: float = 1.1
    bloom_green_boost: float = 1.05

    # Grain
    grain_strength: float = 0.02
    grain_poisson_lambda: float = 25.0
    grain_poisson_divisor: float = 10.0

    # Sharpen
    sharpen_strength: float = 0.7
    sharpen_sigma: float = 1.0

    # Contrast
    contrast_strength: float = 0.6
    log_contrast_scale: float = 9.999
    log_contrast_base: float = 10.0

    # Chroma Processing
    chroma_resize_x: float = 0.25
    chroma_resize_y: float = 1.0
    chroma_blur_sigma: float = 3.0
    chroma_blur_weight: float = 0.5
    chroma_contrast_boost: float = 0.5
    chroma_channel_balance_cr: float = 2.0
    chroma_channel_balance_cb: float = -2.0
    chroma_roll_shift: int = 1

    # Sepia
    sepia_strength: float = 1.0
    sepia_red_coeff: float = 0.393
    sepia_green_coeff: float = 0.769
    sepia_blue_coeff: float = 0.189
    sepia_red_coeff_b: float = 0.272
    sepia_green_coeff_b: float = 0.534
    sepia_blue_coeff_b: float = 0.131
    sepia_red_coeff_g: float = 0.349
    sepia_green_coeff_g: float = 0.686
    sepia_blue_coeff_g: float = 0.168

    # Dither
    dither_amount: float = 0.5

    # JPEG
    jpeg_quality: int = 95

    # Tile
    tiles_x: int = 2
    tiles_y: int = 2
    tile_overlap: int = 20


CFG = EffectsConfig()

INPUT_FOLDER = "input_images"
OUTPUT_FOLDER = "output_images"

Tile = Tuple[
    int, int, int, int,
    int, int, int, int,
    np.ndarray
]

Image = np.ndarray

_CA_CACHE = {}

def get_ca_maps(h: int, w: int, cfg: EffectsConfig):
    key = (h, w, cfg.chromatic_aberration_strength, cfg.ca_shift)

    if key in _CA_CACHE:
        return _CA_CACHE[key]

    cx, cy = w / 2, h / 2

    y, x = np.indices((h, w), dtype=np.float32)

    x -= cx
    y -= cy

    r = np.sqrt(x * x + y * y)
    r_norm = r / (r.max() + 1e-6)
    radial = r_norm ** 2

    base_x = x + cx
    base_y = y + cy

    rb_x = base_x + x * radial * cfg.chromatic_aberration_strength * cfg.ca_shift
    rb_y = base_y + y * radial * cfg.chromatic_aberration_strength * cfg.ca_shift

    g_x = base_x - x * radial * (cfg.chromatic_aberration_strength * 0.5) * cfg.ca_shift
    g_y = base_y - y * radial * (cfg.chromatic_aberration_strength * 0.5) * cfg.ca_shift

    maps = (
        rb_x.astype(np.float32),
        rb_y.astype(np.float32),
        g_x.astype(np.float32),
        g_y.astype(np.float32),
    )

    _CA_CACHE[key] = maps
    return maps


# YCbCr

def split_ycc(img):
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    return y, cr, cb


def merge_ycc(y, cr, cb):
    ycc = cv2.merge([y, cr, cb])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


# Main effects

def apply_chromatic_aberration(image, cfg):
    h, w = image.shape[:2]

    rb_x, rb_y, g_x, g_y = get_ca_maps(h, w, cfg)

    b, g, r = cv2.split(image)

    r = cv2.remap(r, rb_x, rb_y, cv2.INTER_LINEAR)
    b = cv2.remap(b, rb_x, rb_y, cv2.INTER_LINEAR)
    g = cv2.remap(g, g_x, g_y, cv2.INTER_LINEAR)

    return cv2.merge([b, g, r])


def generate_bloom_layer(image, cfg):

    if image is None or image.size == 0:
        print("generate_bloom_layer: input image is empty!")
        return image

    try:
        img = image.astype(np.float32) / 255.0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.0

        mask = np.clip(
            (gray - cfg.bloom_highlight_threshold) / cfg.bloom_highlight_range,
            0,
            1
        )

        highlights = img * mask[:, :, None]

        bloom = cv2.GaussianBlur(
            highlights,
            (0, 0),
            cfg.bloom_sigma_small
        )

        bloom += cv2.GaussianBlur(
            highlights,
            (0, 0),
            cfg.bloom_sigma_large
        ) * cfg.bloom_large_weight

        bloom[:, :, 2] *= cfg.bloom_red_boost
        bloom[:, :, 1] *= cfg.bloom_green_boost

        bloom = np.clip(bloom, 0, 1)

        return (bloom * 255).astype(np.uint8)

    except Exception as e:
        print(f"generate_bloom_layer error: {e}")
        return image


def digital_sensor_grain(img, cfg):
    img_f = img.astype(np.float32)

    lum = np.mean(img_f / 255.0, axis=2)
    weight = (1.0 - lum)[:, :, None]

    noise = (
        np.random.poisson(
            lam=cfg.grain_poisson_lambda,
            size=img.shape
        ).astype(np.float32) - cfg.grain_poisson_lambda
    ) / cfg.grain_poisson_divisor

    img_f += noise * 255 * cfg.grain_strength * weight

    return np.clip(img_f, 0, 255).astype(np.uint8)


def apply_log_contrast(img, cfg, invert=False):
    img = img.astype(np.float32) / 255.0

    img_log = (
        np.log1p(img * cfg.log_contrast_scale) /
        np.log(cfg.log_contrast_base)
    )

    if invert:
        img_log = 1.0 - img_log

    img_log = img_log ** (1.0 / cfg.contrast_strength)

    return (np.clip(img_log, 0, 1) * 255).astype(np.uint8)


def rcas_like_sharpen(img, cfg):

    img_f = img.astype(np.float32)

    kernel = np.array([
        [-1, -1, -1],
        [-1,  0,  9],
        [-1, -1, -1]
    ]) / 2.0

    sharpened = cv2.filter2D(img_f, -1, kernel)

    return np.clip(
        img_f + (sharpened - img_f) * cfg.sharpen_strength,
        0,
        255
    ).astype(np.uint8)


def process_chroma(cr, cb, cfg):

    chroma = cv2.merge([cr, cb]).astype(np.float32)

    chroma -= 128.0

    h, w = chroma.shape[:2]

    small = cv2.resize(
        chroma,
        (
            int(w * cfg.chroma_resize_x),
            int(h * cfg.chroma_resize_y)
        ),
        interpolation=cv2.INTER_AREA
    )

    blur = cv2.GaussianBlur(
        small,
        (0, 0),
        cfg.chroma_blur_sigma
    )

    small = small + blur * cfg.chroma_blur_weight

    small *= (
        1.0 +
        cfg.contrast_strength *
        cfg.chroma_contrast_boost
    )

    noise = np.random.poisson(*small.shape).astype(np.float32)

    small += noise * 255 * cfg.grain_strength

    small[:, :, 0] += cfg.chroma_channel_balance_cr
    small[:, :, 1] += cfg.chroma_channel_balance_cb

    up = cv2.resize(
        small,
        (w, h),
        interpolation=cv2.INTER_LANCZOS4
    )

    up += 128.0

    up = np.clip(up, 0, 255).astype(np.uint8)

    cr_out = up[:, :, 0]
    cb_out = np.roll(
        up[:, :, 1],
        cfg.chroma_roll_shift,
        axis=1
    )

    return cr_out, cb_out


def dither_blend(a, b, cfg):

    h, w = a.shape[:2]

    bayer4 = np.array([
        [0,  8,  2, 10],
        [12, 4, 14,  6],
        [3, 11,  1,  9],
        [15, 7, 13,  5]
    ], dtype=np.float32) / 16.0

    tiled = np.tile(
        bayer4,
        (h // 4 + 1, w // 4 + 1)
    )[:h, :w]

    mask = (tiled < cfg.dither_amount).astype(np.uint8)
    mask3 = mask[:, :, None]

    return np.where(mask3 == 1, b, a)


def split_tiles(img, cfg):

    h, w = img.shape[:2]

    tile_h = h // cfg.tiles_y
    tile_w = w // cfg.tiles_x

    overlap = cfg.tile_overlap

    tiles = []

    for ty in range(cfg.tiles_y):
        for tx in range(cfg.tiles_x):

            y0 = ty * tile_h
            x0 = tx * tile_w

            y1 = (
                (ty + 1) * tile_h
                if ty < cfg.tiles_y - 1
                else h
            )

            x1 = (
                (tx + 1) * tile_w
                if tx < cfg.tiles_x - 1
                else w
            )

            sy0 = max(0, y0 - overlap)
            sx0 = max(0, x0 - overlap)

            sy1 = min(h, y1 + overlap)
            sx1 = min(w, x1 + overlap)

            tile = img[sy0:sy1, sx0:sx1]

            tiles.append((
                y0, y1, x0, x1,
                sy0, sy1, sx0, sx1,
                tile
            ))

    return tiles


def merge_tiles(base_img, processed_tiles):

    out = base_img.copy()

    for y0, y1, x0, x1, tile in processed_tiles:
        out[y0:y1, x0:x1] = tile

    return out


def selective_sepia(img, cfg):

    src = img.astype(np.float32)

    ycc = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2YCrCb
    ).astype(np.float32)

    Y, Cr, Cb = cv2.split(ycc)

    dist = np.sqrt(
        (Cr - 128.0) ** 2 +
        (Cb - 128.0) ** 2
    )

    max_dist = np.sqrt(128.0 ** 2 + 128.0 ** 2)

    t = np.clip(dist / max_dist, 0.0, 1.0)

    chroma_mask = np.sin(np.pi * t)

    lum = Y / 255.0
    lum_mask = np.sin(np.pi * lum)

    mask = (
        chroma_mask *
        lum_mask *
        cfg.sepia_strength
    )

    mask = np.clip(mask, 0.0, 1.0)

    B, G, R = cv2.split(src)

    sepB = (
        cfg.sepia_red_coeff_b * R +
        cfg.sepia_green_coeff_b * G +
        cfg.sepia_blue_coeff_b * B
    )

    sepG = (
        cfg.sepia_red_coeff_g * R +
        cfg.sepia_green_coeff_g * G +
        cfg.sepia_blue_coeff_g * B
    )

    sepR = (
        cfg.sepia_red_coeff * R +
        cfg.sepia_green_coeff * G +
        cfg.sepia_blue_coeff * B
    )

    sepia = cv2.merge([sepB, sepG, sepR])

    mask3 = cv2.merge([mask, mask, mask])

    out = src * (1.0 - mask3) + sepia * mask3

    return np.clip(out, 0, 255).astype(np.uint8)


def process_tile(tile, cfg):

    (
        y0, y1, x0, x1,
        sy0, sy1, sx0, sx1,
        img
    ) = tile

    y, cr, cb = split_ycc(img)

    cr_p, cb_p = process_chroma(cr, cb, cfg)

    y_sharp = rcas_like_sharpen(y, cfg)

    merged = merge_ycc(y_sharp, cr_p, cb_p)

    sepia_merged = selective_sepia(merged, cfg)

    final_tile = dither_blend(
        merged,
        sepia_merged,
        cfg
    )

    crop_top = y0 - sy0
    crop_left = x0 - sx0

    crop_bottom = crop_top + (y1 - y0)
    crop_right = crop_left + (x1 - x0)

    final_tile = final_tile[
        crop_top:crop_bottom,
        crop_left:crop_right
    ]

    return (y0, y1, x0, x1, final_tile)


POOL: Optional[Pool] = None


def init_pool(processes):

    global POOL

    POOL = Pool(processes)


def process_image(args, cfg):

    filename, idx = args

    path = os.path.join(INPUT_FOLDER, filename)

    img = cv2.imread(path)

    if img is None:
        print(f"Failed to load: {filename}")
        return None

    h, w = img.shape[:2]

    size = min(h, w)

    top = (h - size) // 2
    left = (w - size) // 2

    img = img[
        top:top + size,
        left:left + size
    ]

    img = cv2.add(
        img,
        generate_bloom_layer(img, cfg)
    )

    tiles = split_tiles(img, cfg)

    processed_tiles = POOL.starmap(
        process_tile,
        [(t, cfg) for t in tiles]
    )

    final = merge_tiles(img, processed_tiles)

    final = apply_chromatic_aberration(final, cfg)

    final = cv2.GaussianBlur(
        final,
        (3, 3),
        0.5
    )

    out_name = f"photo{idx+1:03d}.jpg"

    out_path = os.path.join(
        OUTPUT_FOLDER,
        out_name
    )

    cv2.imwrite(
        out_path,
        final,
        [
            int(cv2.IMWRITE_JPEG_QUALITY),
            cfg.jpeg_quality,

            int(cv2.IMWRITE_JPEG_PROGRESSIVE),
            1,

            int(cv2.IMWRITE_JPEG_OPTIMIZE),
            1
        ]
    )

    return filename


if __name__ == "__main__":

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    filenames = sorted([
        f for f in os.listdir(INPUT_FOLDER)
        if f.lower().endswith(".png")
    ])

    total = len(filenames)

    init_pool(4)

    image_times = []

    print(f"[{' ' * total}] 0/{total}", end="", flush=True)

    cache_counter = 0

    for i, fn in enumerate(filenames):

        start_time = time.perf_counter()

        process_image((fn, i), CFG)

        elapsed = time.perf_counter() - start_time

        image_times.append(elapsed)

        cache_counter += 1

        if cache_counter % 4 == 0:
            _CA_CACHE.clear()

        bar = ("." * (i + 1)).ljust(total)

        print(
            f"\r[{bar}] {i+1}/{total} | {elapsed:.2f}s/img",
            end="",
            flush=True
        )

    POOL.close()
    POOL.join()

    print(f"\nProcessed {total} images.")

    if image_times:

        fastest = min(image_times)
        slowest = max(image_times)

        average = sum(image_times) / len(image_times)

        sorted_times = sorted(image_times)

        n = len(sorted_times)

        if n % 2 == 1:
            median = sorted_times[n // 2]
        else:
            median = (
                sorted_times[(n // 2) - 1] +
                sorted_times[n // 2]
            ) / 2

        print("\nTiming statistics:")

        print(f"Fastest : {fastest:.2f} s")
        print(f"Slowest : {slowest:.2f} s")
        print(f"Median  : {median:.2f} s")
        print(f"Average : {average:.2f} s")
