Three folders and a helper are given up front, all as absolute paths. `image_dir` holds the source images: every file directly inside it decodes, through Pillow, to an `"RGB"` image, and its filename (with extension) is its *image name*. `transformation_dir` holds *pipeline files*: every file directly inside it is a JSON file whose filename is its *json name*, with the shape

```text
{"transformations": [{"type": "<type>", ...params}, ...]}
```

— an ordered list of *steps*, each a JSON object whose `"type"` selects one of six transformations together with whatever parameters that type needs; the steps a json name lists, in order, are its *pipeline*. `output_dir` is where results go, through the given helper

```py
def get_output_path(image_name: str, json_name: str) -> str: ...
```

which returns the absolute path to write that image's result under that pipeline to; the directory it lives in already exists.

For every image in `image_dir` and every pipeline in `transformation_dir` — every one of the $|\text{images}| \times |\text{pipelines}|$ combinations — run that pipeline's steps against that image, in order, and save the result at `get_output_path(image_name, json_name)` as a PNG. A pipeline's first step takes the freshly decoded source image as input; every later step takes the previous step's output. A pipeline may list zero steps, in which case its output is the source image, unchanged, saved as a PNG.

Six step types are supported, named by the step's `"type"` field. `grayscale` and the two flips take no other parameters; `scale`, `blur` and `rotate` each take one, named below.

- `grayscale` converts the image to Pillow's single-channel mode `"L"` (`image.convert("L")`).
- `flip_horizontal` mirrors the image left-to-right (`image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)`).
- `flip_vertical` mirrors the image top-to-bottom (`image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)`).
- `scale` — `"factor"`, a positive float — resizes the image to `(max(1, round(width * factor)), max(1, round(height * factor)))`, where `width` and `height` are its current size, `round` is Python's round-half-to-even, and the resampling filter is `Image.Resampling.BILINEAR`.
- `blur` — `"radius"`, a non-negative int — applies a Gaussian blur of that radius (`image.filter(ImageFilter.GaussianBlur(radius=radius))`).
- `rotate` — `"angle"`, a float in degrees — rotates the image counter-clockwise about its centre by `angle` degrees, expanding the canvas (`expand=True`) so the whole rotated image fits, with the newly exposed corners filled in black (`0` in mode `"L"`, `(0, 0, 0)` in mode `"RGB"`); the resampling filter is again `Image.Resampling.BILINEAR`.

`grayscale` is the only step that changes an image's mode; `scale` and `rotate` are the only ones that change its size; every other combination of mode and size is left exactly as it was.

For example, `sample.png` decodes to an `"RGB"` image of size `(60, 40)`. The pipeline in `p.json`,

```json
{"transformations": [{"type": "grayscale"}, {"type": "scale", "factor": 0.5}, {"type": "rotate", "angle": 90}]}
```

is applied like this:

```text
decode                            mode "RGB", size (60, 40)
grayscale                      -> mode "L",   size (60, 40)   # mode changes; size does not
scale(factor=0.5)              -> mode "L",   size (30, 20)   # round(60*0.5)=30, round(40*0.5)=20
rotate(angle=90, expand=True)  -> mode "L",   size (20, 30)   # a multiple of 90 degrees swaps width and height
```

and the result is saved at `get_output_path("sample.png", "p.json")`.

### Part 1 — Sequential pipeline

Implement `process_images`.

```py
def process_images(
    image_dir: str,
    transformation_dir: str,
    output_dir: str,
    get_output_path: Callable[[str, str], str],
) -> None: ...
```

```text
image_dir:           a.png, b.png
transformation_dir:  x.json, y.json
process_images writes exactly 4 files, one per (image, pipeline) pair:
  get_output_path("a.png", "x.json")
  get_output_path("a.png", "y.json")
  get_output_path("b.png", "x.json")
  get_output_path("b.png", "y.json")
```

### Part 2 — Parallel batch

`image_dir` may now hold images large enough, and `transformation_dir` pipelines numerous enough, that running them through `process_images` sequentially is too slow to be practical. Implement `process_images_parallel`: the same outputs as `process_images`, for every input, but spread across more than one CPU core. It must not decode the same source image once per pipeline that uses it, and nothing but file paths and JSON-sized data may cross a process boundary — never a decoded image or its raw pixel bytes.

```py
def process_images_parallel(
    image_dir: str,
    transformation_dir: str,
    output_dir: str,
    get_output_path: Callable[[str, str], str],
    max_workers: int | None = None,
) -> None: ...
```

`max_workers` bounds how many worker processes run at once; omitted or `None` uses `os.cpu_count()`.

```text
big_images/ has 200 images; transformations/ has 5 pipeline files.
process_images_parallel must still write all 200 * 5 = 1,000 outputs,
decoding only 200 images in total -- once each -- rather than 1,000.
```

### Part 3 — Shared-prefix reuse

A *prefix* of a pipeline is its first $k$ steps, for some $0 \le k \le$ its length; two prefixes are *the same prefix* only if they have the same length and, step by step, the same `"type"` and the same parameter value(s). Implement `process_images_shared_prefix`: the same signature, outputs and cross-process rules as `process_images_parallel`, but for a given image, every prefix shared by two or more of its pipelines is computed only once — continuing one of them past that prefix must not visibly affect another that shares it. Sharing only ever happens between pipelines applied to the *same* image; two pipelines for different images never share work, even when their steps are identical. Do not keep a shared prefix's image alive for the whole run: once every pipeline that needed it has used it, it may be released.

```py
def process_images_shared_prefix(
    image_dir: str,
    transformation_dir: str,
    output_dir: str,
    get_output_path: Callable[[str, str], str],
    max_workers: int | None = None,
) -> None: ...
```

For one image with the pipelines

```text
p1: grayscale -> scale(factor=0.5) -> rotate(angle=90)
p2: grayscale -> scale(factor=0.5) -> blur(radius=2)
p3: grayscale -> scale(factor=0.5) -> rotate(angle=90)      # identical to p1
p4: flip_horizontal -> blur(radius=2)
```

applying every pipeline independently would apply `grayscale` and `scale(factor=0.5)` three times each (p1, p2, p3), `rotate(angle=90)` twice (p1, p3), and `flip_horizontal` and `blur(radius=2)` once each for p4: 11 step applications. With sharing, `grayscale` and `scale(factor=0.5)` are each applied once and reused by p1, p2 and p3; `rotate(angle=90)` is applied once, and its result is saved as the output of both p1 and p3; the `blur(radius=2)` after `flip_horizontal` is a separate application from the one after `scale(factor=0.5)` in p2, since the two are reached by different prefixes. Six step applications in total: `grayscale`, `scale(factor=0.5)`, `rotate(angle=90)`, `blur(radius=2)` (after `scale`), `flip_horizontal`, `blur(radius=2)` (after `flip_horizontal`).
