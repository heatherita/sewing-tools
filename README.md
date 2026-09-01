# Sewing Pattern Stitcher

Small OpenCV CLI for turning photographed sewing pattern pages into SVG paths.

The pipeline is:

1. Load each image.
2. Detect ArUco markers.
3. Align images using matching marker IDs when possible.
4. Convert the stitched image to grayscale.
5. Threshold the grayscale image.
6. Find contours.
7. Smooth contours.
8. Export contours as SVG paths.

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

`opencv-contrib-python` is required because ArUco support lives in OpenCV contrib.

## Use

```bash
sewstitch tmp/*.jpg -o pattern.svg
```

Useful options:

```bash
sewstitch tmp/*.jpg \
  --output pattern.svg \
  --aruco-dictionary DICT_4X4_50 \
  --calibration-width 50 \
  --calibration-unit mm \
  --threshold adaptive \
  --min-area 250 \
  --smooth-epsilon 0.003 \
  --debug-dir debug
```

Notes:

- The tool extracts dark pattern lines on light paper by default. Use `--no-invert` for light lines on a dark background.
- Images are stitched through any chain of shared ArUco marker IDs, regardless of input order. For example, image 3 can align through image 2 even if it has no marker IDs in common with image 1.
- The stitcher compares every image pair, then uses the strongest marker links to assemble each connected group.
- Each shared marker contributes its four detected corners to the alignment, so adjacent images can align with a single shared marker when detection is clean.
- If images split into disconnected marker groups, each group is stitched internally and then placed beside the others on the SVG canvas.
- `--normalize-marker-orientation` is enabled by default. It rotates disconnected groups so marker top edges point the same way. Use `--no-normalize-marker-orientation` if your markers are intentionally rotated differently.
- SVG output uses raw pixel units by default. Use `--calibration-object aruco --calibration-width 50 --calibration-unit mm` to scale from a detected ArUco marker.
- For a more stable known-width object, use `--calibration-object manual --calibration-points x1,y1,x2,y2 --calibration-width 50`. Points are in original input image pixels; use `--calibration-image-index` if the object is not in the first image.
- `--smooth-epsilon` is a fraction of each contour perimeter. Higher values produce simpler paths.
- Check `debug/01_stitched.png` when the SVG does not look right; it shows what was stitched before contour extraction.

Manual calibration example:

```bash
sewstitch tmp/*.jpg \
  -o pattern.svg \
  --calibration-object manual \
  --calibration-points 120,830,640,830 \
  --calibration-width 50 \
  --calibration-unit mm
```

## Common neighbor layout mode

this is the default. Use it for now and make sure every picture shares at least one marker. 


## Marker Layout Mode 

For the most stable workflow, give the tool a fixed ArUco board/table layout. Then each photo is warped directly into table coordinates:

```bash
sewstitch tmp/*.jpg \
  -o pattern.svg \
  --marker-layout marker-layout.json \
  --layout-pixels-per-unit 4 \
  --min-layout-markers 2 \
  --max-layout-error 3 \
  --debug-dir debug
```

`marker-layout.json` can either map each marker ID to its four table-coordinate corners in printed marker order, or define only the measured outer rectangle.

For the measured-rectangle shortcut, take one reference photo that sees the full marker border, then provide the measured width and height of the outermost marker-corner rectangle:

```json
{
  "unit": "cm",
  "width": 55,
  "height": 135
}
```

If you measured the four side lengths directly, provide left, right, top, and bottom instead. The tool averages opposite sides to make the rectangle:

```json
{
  "unit": "cm",
  "left": 139,
  "right": 138.5,
  "top": 163,
  "bottom": 163
}
```

This example is treated as a rectangle 163 cm wide and 138.75 cm tall.

The tool uses the input image with the most detected markers as the reference layout photo. It maps that photo's outer detected marker corners to the measured rectangle, then derives layout coordinates for the markers visible in that reference photo. This works best when the border markers and pattern are on the same flat plane.

For fully measured coordinates, use:

```json
{
  "unit": "mm",
  "markers": {
    "5": [[0, 0], [100, 0], [100, 100], [0, 100]],
    "8": [[500, 0], [600, 0], [600, 100], [500, 100]],
    "12": [[0, 700], [100, 700], [100, 800], [0, 800]]
  }
}
```

Use measured coordinates. If your marker is 38.4 mm square, write `38.4`; do not round it to a convenient value. When `unit` is set, the SVG is written with that unit.

Layout controls:

- `--layout-pixels-per-unit` controls raster processing resolution. With `unit: "mm"`, `4` means OpenCV processes at 4 px/mm, then writes the SVG back in millimeters.
- `--min-layout-markers` controls how many known layout markers a photo must contain before it can be warped into table coordinates. The default is `2`.
- `--max-layout-error` rejects photos whose marker reprojection error is too high. With millimeter coordinates, the default `3` means 3 mm.
- `debug/stitch-report.txt` lists each photo's accepted/rejected layout solve, matched markers, inliers, mean error, and max error.

##PHOTOGRAPHY TIPS:  

  Tape black posterboard to wall with Aruco markers around it. See wall-example.jpg.
  Iron the pattern tissue before photographing
  See tmp/examples for some example images

## TODO:

 - modularize this script by breaking it into a directory structure with several .py files

   
