import cv2
import numpy as np
import os
from pathlib import Path
import re


class AutoCrop:
    """
    Crop a Taobao "card" product image down to just the product photo.

    Taobao templates wrap the real product photo in decoration: a logo (a
    full-width bar at the very top, or a swoosh embedded inside the photo), a
    thin frame, white margin all around, and often a gibberish caption block at
    the bottom. The product photo itself sits on its natural (soft grey / studio)
    background.

    What we want is NOT a cut-out on white -- it is the original photo, untouched,
    with only the decoration removed: drop the logo, the frame, the caption and
    the surrounding margin, keep the product (model AND any pedestal it stands on)
    exactly as shot, on its own background.

    Method (simple, robust, no ML):
      1. Photo panel = largest connected region of non-white pixels. The outer
         white margin is excluded; a separate top logo BAR (its own component)
         is excluded. Inset a few px to drop the thin frame line.
      2. Inside the panel, take the largest contiguous vertical run of content
         rows. This drops a logo that sits ABOVE the product with a gap (it is a
         shorter, separate run).
      3. Inside that run, bound to the rows that actually contain product
         (rows with enough genuinely dark pixels). Logo swooshes and caption
         text sit on uniform LIGHT-grey bands that have almost no dark pixels, so
         they are excluded; the model and the (darker) pedestal are kept.
      4. Trim near-white margin columns, add a little breathing room, and crop.
         Original pixels and the natural background are kept -- nothing is matted.

    Note: this targets the common Taobao card layouts (logo top, caption bottom,
    product in the middle on a light background). A product part that is itself
    very light and sits at the extreme top/bottom edge could be trimmed; in
    practice these shots have a dark head/footwear/pedestal at the extremes.
    """

    WHITE     = 250    # gray >= this is pure-white margin (not content)
    INK       = 235    # gray < this counts as "any content" (incl. light product)
    DARK      = 205    # gray < this counts as genuinely dark product content
    ROWFRAC   = 0.01   # a row has content if this fraction of pixels is < INK
    DARKFRAC  = 0.12   # a row is product (not a light caption/logo band) if this
                       # fraction of pixels is < DARK
    INSET     = 8      # px shaved off the panel to drop the thin frame line
    PAD_V     = 24     # vertical breathing room kept around the product
    PAD_H     = 20     # horizontal breathing room kept around the product
    MIN_FRAC  = 0.05   # ignore "panels" smaller than this fraction of the image

    def __init__(self):
        self.images = []

    def natural_sort_key(self, s):
        return [int(t) if t.isdigit() else t.lower()
                for t in re.split(r'([0-9]+)', str(s))]

    def _read(self, path):
        img = cv2.imread(path)
        if img is None:  # webp / avif / odd codecs -> PIL fallback
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

    def _runs(self, mask):
        out, s = [], None
        for i, v in enumerate(mask):
            if v and s is None:
                s = i
            elif not v and s is not None:
                out.append((s, i)); s = None
        if s is not None:
            out.append((s, len(mask)))
        return out

    def crop_to_product(self, img):
        """Crop to the product (logo, frame, caption and white margin removed)."""
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        H, W = g.shape

        # 1) photo panel = largest non-white component (drops outer margin + separate logo bar)
        content = (g < self.WHITE).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(content, 8)
        if n <= 1:
            return img
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[i, cv2.CC_STAT_AREA] < self.MIN_FRAC * H * W:
            return img
        px, py = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        pw, ph = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        ins = self.INSET if (pw > 4 * self.INSET and ph > 4 * self.INSET) else 0
        px, py, pw, ph = px + ins, py + ins, pw - 2 * ins, ph - 2 * ins
        pg = g[py:py + ph, px:px + pw]

        # 2) largest contiguous vertical run of content rows (drops a separated top logo)
        flight = (pg < self.INK).mean(axis=1)
        runs = self._runs(flight > self.ROWFRAC)
        if not runs:
            return img[py:py + ph, px:px + pw]
        a, b = max(runs, key=lambda r: r[1] - r[0])

        # 3) bound to rows that hold genuinely dark product (drops light caption/logo bands)
        fdark = (pg < self.DARK).mean(axis=1)
        drows = [y for y in range(a, b) if fdark[y] > self.DARKFRAC]
        top, bot = (drows[0], drows[-1] + 1) if drows else (a, b)
        y0 = max(0, top - self.PAD_V)
        y1 = min(ph, bot + self.PAD_V)

        # 4) trim near-white margin columns, keep a little breathing room
        band = pg[y0:y1]
        cmask = (band < self.WHITE).mean(axis=0) > 0.02
        cols = np.where(cmask)[0]
        if len(cols):
            x0 = max(0, int(cols.min()) - self.PAD_H)
            x1 = min(pw, int(cols.max()) + 1 + self.PAD_H)
        else:
            x0, x1 = 0, pw

        return img[py + y0:py + y1, px + x0:px + x1]

    def process(self, input_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        if self.load_images(input_dir) == 0:
            print("No images found!")
            return []

        print("\n=== AUTO-CROPPING IMAGES (photo-panel crop) ===")
        cropped_paths = []
        for img_data in self.images:
            cropped = self.crop_to_product(img_data['image'])
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
