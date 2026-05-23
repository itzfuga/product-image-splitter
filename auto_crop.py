import cv2
import numpy as np
import os
from pathlib import Path
import re

try:
    from rembg import remove, new_session
    _REMBG_OK = True
except Exception:
    _REMBG_OK = False


class AutoCrop:
    """
    Crop a Taobao product card down to just the product photo.

    The logic (what we actually want):
        Crop to the tight bounding box of the PRODUCT = the model plus whatever
        it is shown with (pedestal / clothes pile / floor). Remove the logo at
        the top, the caption/footer at the bottom, and every empty margin around
        it (white OR grey). Never cut into the product itself -- the whole head
        stays, the full product width stays.

    Why this needs a salient-object detector (rembg / U2-Net) and not a
    brightness threshold: a thin head reads like an empty row, and a light/cream
    or grey product reads like the grey logo/caption -- so no pixel threshold can
    tell "product" from "decoration" without either chopping the head or keeping
    the logo. rembg identifies the model the way the eye does, regardless of
    colour. We use it ONLY to find the bounding box; the crop keeps the original
    pixels and background (no cut-out, no white matting).

    Steps:
      1. rembg -> mask of the model -> its bounding box (head to feet, full width).
      2. Extend the box DOWNWARD to include the base the model stands on
         (pedestal / pile / floor = real content below the feet). Stop at a
         light-grey caption strip or a white gap, so the caption stays out.
      3. Tighten left/right to the product's actual content columns (drop empty
         side margins).
      4. Crop the ORIGINAL image to that box.

    Falls back to a plain content bounding box if rembg is unavailable.
    """

    ALPHA     = 30     # rembg alpha above this = model pixel
    CONTENT   = 235    # gray < this = real content (catches light/cream/grey product)
    DOWNFRAC  = 0.05   # while extending down, a row with < this content fraction is "empty"
    GAPRUN    = 12     # this many empty rows below the feet = end of the base -> stop
    BAND_LIGHT = 0.55  # a caption/logo strip: > this fraction is non-white ...
    BAND_DARK  = 0.06  # ... while < this fraction is genuinely dark (mostly light grey)
    DARK      = 150    # gray < this = genuinely dark
    LIGHT     = 248    # gray < this = non-white
    EDGE_STD  = 11     # an edge row/col with std < this and bright mean is a uniform frame/margin
    EDGE_MEAN = 205    # ... and mean gray > this (light grey frame or white margin)
    EDGE_MAXFRAC = 0.06  # trim at most this fraction off each edge, so a wide textured
                         # base (pedestal/pile/floor) is never eaten as if it were a frame

    def __init__(self):
        self.images = []
        self._sess = None

    def natural_sort_key(self, s):
        return [int(t) if t.isdigit() else t.lower()
                for t in re.split(r'([0-9]+)', str(s))]

    def _read(self, path):
        img = cv2.imread(path)
        if img is None:
            try:
                from PIL import Image
                img = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
            except Exception:
                img = None
        return img

    def load_images(self, input_dir):
        self.images = []
        files = [f for f in os.listdir(input_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.avif', '.gif'))]
        files.sort(key=self.natural_sort_key)
        for f in files:
            path = os.path.join(input_dir, f)
            img = self._read(path)
            if img is not None:
                self.images.append({'filename': f, 'image': img, 'path': path})
                print(f"Loaded: {f} - Shape: {img.shape}")
        print(f"\nTotal images loaded: {len(self.images)}")
        return len(self.images)

    def _session(self):
        if self._sess is None:
            self._sess = new_session("u2net")
        return self._sess

    def _figure_mask(self, path, H, W):
        if not _REMBG_OK:
            return None
        try:
            with open(path, 'rb') as fh:
                out = remove(fh.read(), session=self._session())
            rgba = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_UNCHANGED)
        except Exception:
            return None
        if rgba is None or rgba.ndim < 3 or rgba.shape[2] < 4:
            return None
        if rgba.shape[:2] != (H, W):
            rgba = cv2.resize(rgba, (W, H), interpolation=cv2.INTER_NEAREST)
        m = (rgba[:, :, 3] > self.ALPHA).astype(np.uint8)
        if m.sum() < 0.01 * H * W:
            return None
        return m

    def crop_to_product(self, img, path):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        H, W = g.shape
        fig = self._figure_mask(path, H, W)

        if fig is None:
            # fallback: plain content bounding box (still drops white margins)
            ys, xs = np.where(g < self.CONTENT)
            if len(ys) == 0:
                return img
            return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

        ys, xs = np.where(fig > 0)
        fy0, fy1 = ys.min(), ys.max() + 1

        flight = (g < self.LIGHT).mean(axis=1)
        fdark = (g < self.DARK).mean(axis=1)
        crow = (g < self.CONTENT).mean(axis=1)

        def light_band(y):
            return flight[y] > self.BAND_LIGHT and fdark[y] < self.BAND_DARK

        # extend down through the base (pedestal/pile/floor); stop at caption strip or white gap
        y1 = fy1
        run = 0
        y = fy1
        while y < H:
            if light_band(y):
                break
            if crow[y] < self.DOWNFRAC:
                run += 1
                if run >= self.GAPRUN:
                    break
            else:
                run = 0
                y1 = y + 1
            y += 1

        # trim the thin grey frame line and empty (white/grey) margins off all four edges.
        # A frame/margin is a UNIFORM bright strip (low std); the product and its base are
        # textured (high std). Capped at EDGE_MAXFRAC so a wide uniform-ish pedestal is safe.
        sub = g[fy0:y1]
        hh, ww = sub.shape
        cap_x = int(ww * self.EDGE_MAXFRAC)
        cap_y = int(hh * self.EDGE_MAXFRAC)

        def col_uniform(x):
            c = sub[:, x]
            return c.std() < self.EDGE_STD and c.mean() > self.EDGE_MEAN

        def row_uniform(yy):
            r = sub[yy]
            return r.std() < self.EDGE_STD and r.mean() > self.EDGE_MEAN

        x0 = 0
        while x0 < cap_x and col_uniform(x0):
            x0 += 1
        x1 = ww
        while x1 > ww - cap_x and col_uniform(x1 - 1):
            x1 -= 1
        ry0 = 0
        while ry0 < cap_y and row_uniform(ry0):
            ry0 += 1
        ry1 = hh
        while ry1 > hh - cap_y and row_uniform(ry1 - 1):
            ry1 -= 1

        return img[fy0 + ry0:fy0 + ry1, x0:x1]

    def process(self, input_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        if self.load_images(input_dir) == 0:
            print("No images found!")
            return []
        if not _REMBG_OK:
            print("WARNING: rembg not installed - using plain content-bbox fallback.")
        print("\n=== AUTO-CROPPING IMAGES (product bounding box) ===")
        cropped_paths = []
        for img_data in self.images:
            cropped = self.crop_to_product(img_data['image'], img_data['path'])
            stem = Path(img_data['filename']).stem
            out_path = os.path.join(output_dir, f"cropped_{stem}.png")
            cv2.imwrite(out_path, cropped)
            cropped_paths.append(out_path)
            print(f"  OK {img_data['filename']}: {img_data['image'].shape[1]}x{img_data['image'].shape[0]}"
                  f" -> {cropped.shape[1]}x{cropped.shape[0]}")
        print(f"\n=== PROCESSED {len(cropped_paths)}/{len(self.images)} IMAGES ===")
        return cropped_paths


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python auto_crop.py <input_dir> <output_dir>")
        sys.exit(1)
    AutoCrop().process(sys.argv[1], sys.argv[2])
