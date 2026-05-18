import os
import cv2
import numpy as np
import time

from multiprocessing import Pool
from typing import Optional
from dataclasses import dataclass


@dataclass(frozen=True)
class EffectsConfig:

    # Chromatic Aberration
    chromatic_aberration_strength: float = 0.002
    ca_shift: float = 1.5

    # Bloom
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

    # Contrast
    contrast_strength: float = 0.6

    # Chroma
    chroma_resize_x: float = 0.25
    chroma_resize_y: float = 1.0
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
    jpeg_quality: int = 90


CFG = EffectsConfig()

INPUT_FOLDER = "input_images"
OUTPUT_FOLDER = "output_images"

_CA_CACHE = {}

GLOBAL_DARK_NOISE = None
GLOBAL_MID_NOISE = None
GLOBAL_BRIGHT_NOISE = None

GLOBAL_DITHER = None

NOISE_TILE_SIZE = 64
DITHER_BLOCK = 4

POOL: Optional[Pool] = None


def init_noise(cfg):

    global GLOBAL_DARK_NOISE
    global GLOBAL_MID_NOISE
    global GLOBAL_BRIGHT_NOISE

    GLOBAL_DARK_NOISE = (
        np.random.poisson(
            lam=cfg.grain_poisson_lambda,
            size=(
                NOISE_TILE_SIZE,
                NOISE_TILE_SIZE,
                2
            )
        ).astype(np.float32)
        - cfg.grain_poisson_lambda
    ) / cfg.grain_poisson_divisor

    GLOBAL_MID_NOISE = (
        np.random.poisson(
            lam=cfg.grain_poisson_lambda * 0.5,
            size=(
                NOISE_TILE_SIZE,
                NOISE_TILE_SIZE,
                2
            )
        ).astype(np.float32)
        - (cfg.grain_poisson_lambda * 0.5)
    ) / cfg.grain_poisson_divisor

    GLOBAL_BRIGHT_NOISE = (
        np.random.poisson(
            lam=cfg.grain_poisson_lambda * 0.2,
            size=(
                NOISE_TILE_SIZE,
                NOISE_TILE_SIZE,
                2
            )
        ).astype(np.float32)
        - (cfg.grain_poisson_lambda * 0.2)
    ) / cfg.grain_poisson_divisor



def init_dither(max_size=4096):

    global GLOBAL_DITHER

    y, x = np.indices((max_size, max_size))

    GLOBAL_DITHER = (
        (x + y) % 2
    ).astype(np.uint8)


def init_pool(processes: int):

    global POOL
    POOL = Pool(processes)


def get_ca_maps(h, w, cfg):

    key = (
        h,
        w,
        cfg.chromatic_aberration_strength,
        cfg.ca_shift
    )

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

    rb_x = (
        base_x +
        x * radial *
        cfg.chromatic_aberration_strength *
        cfg.ca_shift
    )

    rb_y = (
        base_y +
        y * radial *
        cfg.chromatic_aberration_strength *
        cfg.ca_shift
    )

    g_x = (
        base_x -
        x * radial *
        (cfg.chromatic_aberration_strength * 0.5) *
        cfg.ca_shift
    )

    g_y = (
        base_y -
        y * radial *
        (cfg.chromatic_aberration_strength * 0.5) *
        cfg.ca_shift
    )

    maps = (
        rb_x.astype(np.float32),
        rb_y.astype(np.float32),
        g_x.astype(np.float32),
        g_y.astype(np.float32),
    )

    _CA_CACHE[key] = maps

    return maps


def split_ycc(img):

    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    y, cr, cb = cv2.split(ycc)

    return y, cr, cb


def merge_ycc(y, cr, cb):

    ycc = cv2.merge([y, cr, cb])

    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def apply_chromatic_aberration(image, cfg):

    h, w = image.shape[:2]

    small = cv2.resize(
        image,
        (w // 2, h // 2),
        interpolation=cv2.INTER_AREA
    )

    sh, sw = small.shape[:2]

    rb_x, rb_y, g_x, g_y = get_ca_maps(
        sh,
        sw,
        cfg
    )

    b, g, r = cv2.split(small)

    r = cv2.remap(r, rb_x, rb_y, cv2.INTER_LINEAR)
    b = cv2.remap(b, rb_x, rb_y, cv2.INTER_LINEAR)
    g = cv2.remap(g, g_x, g_y, cv2.INTER_LINEAR)

    merged = cv2.merge([b, g, r])

    return cv2.resize(
        merged,
        (w, h),
        interpolation=cv2.INTER_LINEAR
    )


def generate_bloom_layer(image, cfg):

    h, w = image.shape[:2]

    small = cv2.resize(
        image,
        (w // 2, h // 2),
        interpolation=cv2.INTER_AREA
    )

    img = small.astype(np.float32) / 255.0

    gray = cv2.cvtColor(
        small,
        cv2.COLOR_BGR2GRAY
    ) / 255.0

    mask = np.clip(
        (
            gray -
            cfg.bloom_highlight_threshold
        ) / cfg.bloom_highlight_range,
        0,
        1
    )

    highlights = img * mask[:, :, None]

    
    pyr1 = cv2.pyrDown(highlights)
    pyr1 = cv2.pyrUp(cv2.pyrDown(pyr1))  # <-- Pyramid blur

    
    pyr2 = cv2.pyrDown(pyr1)
    pyr2 = cv2.pyrUp(cv2.pyrDown(pyr2))  # <-- Pyramid blur
    bloom_small = cv2.pyrUp(
        pyr1,
        dstsize=(
            highlights.shape[1],
            highlights.shape[0]
        )
    )

    bloom_large = cv2.pyrUp(
        cv2.pyrUp(
            pyr2,
            dstsize=(
                pyr1.shape[1],
                pyr1.shape[0]
            )
        ),
        dstsize=(
            highlights.shape[1],
            highlights.shape[0]
        )
    )

    bloom = (
        bloom_small +
        bloom_large *
        cfg.bloom_large_weight
    )

    bloom[:, :, 2] *= cfg.bloom_red_boost
    bloom[:, :, 1] *= cfg.bloom_green_boost

    bloom = np.clip(bloom, 0, 1)

    bloom = (bloom * 255).astype(np.uint8)

    return cv2.resize(
        bloom,
        (w, h),
        interpolation=cv2.INTER_LINEAR
    )


def rcas_like_sharpen(img, cfg):

    img_f = img.astype(np.float32)

    kernel = np.array([
        [-1, -1, -1],
        [-1,  0,  9],
        [-1, -1, -1]
    ]) / 2.0

    sharpened = cv2.filter2D(img_f, -1, kernel)

    return np.clip(
        img_f +
        (sharpened - img_f) *
        cfg.sharpen_strength,
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

    
    pyr = cv2.pyrDown(small)
    pyr = cv2.pyrUp(cv2.pyrDown(pyr))  

    blur = cv2.pyrUp(
        pyr,
        dstsize=(
            small.shape[1],
            small.shape[0]
        )
    )

    small = small + blur * cfg.chroma_blur_weight

    small *= (
        1.0 +
        cfg.contrast_strength *
        cfg.chroma_contrast_boost
    )

    lum = np.mean(
        small + 128.0,
        axis=2
    ) / 255.0

    sh, sw = small.shape[:2]

    rep_y = sh // NOISE_TILE_SIZE + 1
    rep_x = sw // NOISE_TILE_SIZE + 1

    dark_noise = np.tile(
        GLOBAL_DARK_NOISE,
        (rep_y, rep_x, 1)
    )[:sh, :sw]

    mid_noise = np.tile(
        GLOBAL_MID_NOISE,
        (rep_y, rep_x, 1)
    )[:sh, :sw]

    bright_noise = np.tile(
        GLOBAL_BRIGHT_NOISE,
        (rep_y, rep_x, 1)
    )[:sh, :sw]

    dark_mask = np.clip(
        (0.5 - lum) * 2.0,
        0,
        1
    )[:, :, None]

    bright_mask = np.clip(
        (lum - 0.5) * 2.0,
        0,
        1
    )[:, :, None]

    mid_mask = (
        1.0 -
        dark_mask -
        bright_mask
    )

    noise = (
        dark_noise * dark_mask +
        mid_noise * mid_mask +
        bright_noise * bright_mask
    )

    small += noise * 255 * cfg.grain_strength

    small[:, :, 0] += cfg.chroma_channel_balance_cr
    small[:, :, 1] += cfg.chroma_channel_balance_cb

    up = cv2.resize(
        small,
        (w, h),
        interpolation=cv2.INTER_LINEAR
    )

    up += 128.0

    up = np.clip(
        up,
        0,
        255
    ).astype(np.uint8)

    cr_out = up[:, :, 0]

    cb_out = np.roll(
        up[:, :, 1],
        cfg.chroma_roll_shift,
        axis=1
    )

    return cr_out, cb_out


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

    max_dist = np.sqrt(
        128.0**2 +
        128.0**2
    )

    t = np.clip(
        dist / max_dist,
        0.0,
        1.0
    )

    chroma_mask = np.sin(np.pi * t)

    lum = Y / 255.0

    lum_mask = np.sin(np.pi * lum)

    mask = np.clip(
        chroma_mask *
        lum_mask *
        cfg.sepia_strength,
        0.0,
        1.0
    )

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

    mask3 = mask[:, :, None]

    out = (
        src * (1.0 - mask3) +
        sepia * mask3
    )

    return np.clip(
        out,
        0,
        255
    ).astype(np.uint8)


def dither_blend(a, b, cfg):

    h, w = a.shape[:2]

    mask = GLOBAL_DITHER[:h, :w]

    mask3 = mask[:, :, None]

    return np.where(mask3 == 1, b, a)


def process_image(args, cfg):

    filename, idx = args

    start_time = time.perf_counter()

    path = os.path.join(
        INPUT_FOLDER,
        filename
    )

    img = cv2.imread(path)

    if img is None:
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

    y, cr, cb = split_ycc(img)

    cr_p, cb_p = process_chroma(cr, cb, cfg)

    y_sharp = rcas_like_sharpen(y, cfg)

    merged = merge_ycc(
        y_sharp,
        cr_p,
        cb_p
    )

    sepia_merged = selective_sepia(
        merged,
        cfg
    )

    final = dither_blend(
        merged,
        sepia_merged,
        cfg
    )

    final = apply_chromatic_aberration(
        final,
        cfg
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
            0,

            int(cv2.IMWRITE_JPEG_OPTIMIZE),
            0
        ]
    )

    elapsed = time.perf_counter() - start_time

    return elapsed


if __name__ == "__main__":

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    filenames = sorted([
        f for f in os.listdir(INPUT_FOLDER)
        if f.lower().endswith(".png")
    ])

    total = len(filenames)

    init_noise(CFG)
    init_dither()
    init_pool(4)

    print(f"Processing {total} images...\n")

    image_times = POOL.starmap(
        process_image,
        [
            ((fn, i), CFG)
            for i, fn in enumerate(filenames)
        ]
    )

    POOL.close()
    POOL.join()

    image_times = [
        t for t in image_times
        if t is not None
    ]

    print(f"Processed {len(image_times)} images.")

    if image_times:

        fastest = min(image_times)
        slowest = max(image_times)

        average = (
            sum(image_times) /
            len(image_times)
        )

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
