# ImageScaler
This is a personal hobby project for learning how scaling works along with some other happy little artistic effects.
This process in GIMP takes me 30 minutes per images.

## Usage

1. Create an `input_images` folder
2. Place `.png` images inside it
3. Put `artsi.py` next to the folder
4. Run:

```
python artsi.py
```
Processed images will appear in:
`output_images/`

## Pipeline:
1. Center crop
2. Bloom generation
3. YCrCb color separation
4. RCAS-inspired sharpening
5. Chroma processing
   - chroma blur
   - chroma contrast shaping
   - chroma noise
   - channel shifting
6. Selective sepia toning
7. Ordered dithering blend
8. Chromatic aberration
9. JPEG recompression

## Technical Notes

- OpenCV-accelerated processing
- Cached chromatic aberration maps
- Pre-generated grain fields
- Multi-scale bloom generation using image pyramids
- SIMD-friendly NumPy operations
- Optimized memory reuse to avoid RAM growth
- Current performance:
  - ~0.30s/image average
  - Originally ~1.5s/image
## Included Effects

### Bloom
Multi-scale highlight diffusion designed to emulate sensor bloom and lens glow.

### Chromatic Aberration
Radial channel displacement based on distance from image center.

### Digital Grain
Brightness-dependent grain response using Poisson-distributed noise.

### Chroma Processing
Low-resolution chroma manipulation inspired by analog compression artifacts.

### Selective Sepia
Adaptive sepia blending based on luminance and chroma intensity.

## Future goals
Will continue debloating this code. It currently has a few redundancies that did not help with speeding it up, specifically the attempt at generating noise values once and reusing them. 
Later on will rewrite in C#.
One more goal is to turn this into a proper GIMP plugin.
Add proper tile-based multiprocessing


