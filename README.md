# jxltran

**jxltran** is a small command-line tool in the spirit of **jpegtran**, but for **JPEG XL** (ISO/IEC 18181). It rewrites JXL files **without** a full decode-and-encode round trip: it operates on the container and codestream layout so pixel data and compression remain bit-identical where the tool does not intentionally change them.

Typical uses:

- Remux the **ISOBMFF container** (strip or add a container, adjust `jxlp` box layout).
- **Reorder or merge** `jxlp` partial-codestream boxes (including out-of-order delivery cases).
- Edit **orientation** and **crop** metadata in the image header when that can be done as a lossless metadata change.
- Adjust **restoration filter (Gaborish)** parameters where the bitstream path supports it.

It is **not** a general-purpose encoder or decoder; for creating or decoding pixels to/from JXL, use tools such as **cjxl** and **djxl** from the JPEG XL reference implementation.

## Requirements

- **C++17** compiler  
- **CMake** 3.16 or newer  
- Optional: **Brotli** (encode + decode libraries and headers) for `brob`-wrapped boxes. Without Brotli, the tool still builds; brob handling is disabled at compile time.

## Build (standalone)

From this directory:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

The `jxltran` binary is written to `build/` (Release or your chosen configuration).

To disable the optional Brotli probe:

```bash
cmake -S . -B build -DJXLTRAN_ENABLE_BROTLI=OFF
```

## Usage

Synopsis:

```text
jxltran INPUT [OUTPUT] [OPTION]...
jxltran INPUT --info
```

With **`--info`**, omit **`OUTPUT`**: the tool prints to stdout: `dimensions: W x H` (orientation-correct, like `djxl`), `animation: yes|no`, and one **`frame_region: N WxH+X+Y`** line per frame (same display space and `WxH+X+Y` convention as **`--crop`**), then exits.

Use **`jxltran -h`** or **`jxltran --help`** for the full generated help (including verbose per-option text). The table below is a compact overview.

| Option | Argument | What it does |
|--------|----------|----------------|
| *(positional)* | `INPUT` | Source JPEG XL file (container or bare codestream). |
| *(positional)* | `OUTPUT` | Destination file (not used with `--info`). |
| `--info` | — | Print dimensions, animation, and per-frame `frame_region` lines (oriented display canvas for each frame); exit (no `OUTPUT`; no other options except optional `-v`). |
| `--container` | `keep` \| `no` \| `yes` \| `if-needed` | Whether to emit a bare codestream or an ISOBMFF container (default `keep`). `no` writes **only** the codestream: the container and **all** metadata boxes are dropped. Multiple `jxlp` parts are concatenated in counter order, i.e. the same compressed payload as after `--jxlp merge` (`--jxlp` is ignored for bare output). |
| `--jxlp` | `keep` \| `sort` \| `merge` | When the **output** is a container: leave `jxlp` layout unchanged, sort by counter (normalize out-of-order delivery), or merge into one `jxlc` (default `keep`). |
| `--strip` | `exif` \| `xmp` \| `jumbf` \| `jbrd` \| `all` \| comma list | Remove metadata boxes from the container. |
| `--box-order` | `keep` \| `before-codestream` \| `after-codestream` \| `TYPE[,…]` | Control ordering of boxes in the output container. |
| `--brob` | `keep` \| `compress[:TYPE[,…]]` \| `decompress` | Brotli-compressed `brob` boxes: leave, wrap eligible metadata, or expand. Only if built with Brotli; otherwise `keep` is the only effective mode. |
| `--set-orientation` | `1`–`8`, `90`, `180`, `270` | Set Exif-style orientation in the image header (absolute). `90`/`180`/`270` are clockwise rotation (same as tags `6`/`3`/`8`). |
| `--set-orientation-relative` | `1`–`8`, `90`, `180`, `270` | Compose this transform *after* the file’s current orientation (e.g. current `8` + relative `6` or `90` → `1`). Mutually exclusive with `--set-orientation`. |
| `--set-bits-per-sample` | `1`–`32` | Set `bits_per_sample` in the image header (XYB-encoded images only). |
| `--set-num-loops` | `N` | Animation loop count (`0` = infinite); animated images only. |
| `--set-tps` | `N`, `N/D`, or `P%` | Animation ticks per second (`P%` scales the file's current TPS); animated images only. |
| `--crop` | `WxH` or `WxH+X+Y` | Metadata-only canvas resize/crop; **X/Y/W/H are in display space** (as `djxl` after orientation). The rectangle may extend outside the current canvas to **expand** it; areas beyond existing frame data are implicit padding (zeros / transparent). |
| `--gab-blur` | `A` | Reversible Gaborish blur strength (`A` ≥ 0); mutually exclusive with the other `--gab-*` options. |
| `--gab-sharpen` | `A` | Sharpen via scaled default Gaborish weights (`A` ≥ 0); mutually exclusive with the other `--gab-*` options. |
| `--gab-weights` | `x1,x2,y1,y2,b1,b2` | Six explicit Gaborish weights; mutually exclusive with the other `--gab-*` options. |
| `-v`, `--verbose` | — | Trace codestream header parse/rewrite to stderr (byte+bit positions). |


## License

Same as the JPEG XL reference implementation (BSD-style; see the repository `LICENSE` file).
