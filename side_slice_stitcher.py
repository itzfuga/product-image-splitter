"""
side_slice_stitcher.py — Reconstruct Taobao product photos that were split
LEFT/RIGHT (vertical slices) instead of top/bottom.

Some Taobao anti-copy templates cut each full photo into a left half and a right
half, pad each with a white margin on its OUTER side, and serve them as separate
files with hash filenames (so order isn't derivable from the name). One stitched
strip can also still contain several photos stacked vertically, separated by a
white band.

This engine:
  1. trims the near-white padding around every slice,
  2. classifies each slice as a left-piece (white margin on the right) or a
     right-piece (white margin on the left),
  3. pairs left+right pieces by matching original height,
  4. auto-detects the correct join order (tries both) and the seam overlap by
     minimising edge mismatch, then feather-blends the seam,
  5. splits the stitched strip into individual photos at white horizontal bands,
  6. writes each photo as product_N.png  (one model shot = one product image).

Drop-in for the Flask app: class `SideSliceStitcher` exposes `.process(input_dir,
output_dir) -> [paths]` and a `.images` list, mirroring SimpleBoxStitcher / AutoCrop.
Also runnable standalone:  python3 side_slice_stitcher.py <input_dir> [output_dir]
"""
import cv2
import numpy as np
import os
import glob
import re


class SideSliceStitcher:
    WHITE = 245          # brightness >= this counts as white padding/separator
    WHITE_FRAC = 0.985   # a row/col is "white" if this fraction of pixels is white
    MAX_OVERLAP = 140    # max seam overlap to search (px)
    EDGE_BAND = 12       # columns compared for a butt-join (overlap 0) cost
    FEATHER = 8          # blend width applied across a detected overlap
    SEP_MIN_FRAC = 0.05  # a vertical photo segment must be >= this fraction of height

    def __init__(self):
        self.images = []

    # ---- io ----
    def _natural_key(self, s):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))]

    def _load(self, path):
        img = cv2.imread(path)
        if img is None:  # webp / avif / odd codecs -> PIL fallback
            from PIL import Image
            img = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
        return img

    # ---- geometry helpers ----
    def _content_box(self, gray):
        colw = (gray >= self.WHITE).mean(axis=0)
        roww = (gray >= self.WHITE).mean(axis=1)
        cols = np.where(colw < self.WHITE_FRAC)[0]
        rows = np.where(roww < self.WHITE_FRAC)[0]
        if len(cols) == 0 or len(rows) == 0:
            return None
        return rows.min(), rows.max() + 1, cols.min(), cols.max() + 1

    def _trim(self, img):
        box = self._content_box(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if box is None:
            return img
        t, b, l, r = box
        return img[t:b, l:r]

    def _white_side(self, img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        w = g.shape[1]
        lw = (g[:, :w // 5] >= self.WHITE).mean()
        rw = (g[:, 4 * w // 5:] >= self.WHITE).mean()
        return 'left' if lw > rw else 'right'

    def _join_cost(self, P, Q):
        """Best (cost, overlap) for placing P left of Q (same height)."""
        h = min(P.shape[0], Q.shape[0])
        P = P[:h]; Q = cv2.resize(Q, (Q.shape[1], h))
        pg = cv2.cvtColor(P, cv2.COLOR_BGR2GRAY).astype(np.float32)
        qg = cv2.cvtColor(Q, cv2.COLOR_BGR2GRAY).astype(np.float32)
        best = (1e18, 0)
        for ov in range(0, min(self.MAX_OVERLAP, P.shape[1] // 2, Q.shape[1] // 2)):
            if ov == 0:
                c = np.abs(pg[:, -self.EDGE_BAND:].mean(1) - qg[:, :self.EDGE_BAND].mean(1)).mean()
            else:
                c = np.abs(pg[:, -ov:] - qg[:, :ov]).mean()
            if c < best[0]:
                best = (c, ov)
        return best

    def _stitch(self, P, Q, ov):
        h = min(P.shape[0], Q.shape[0])
        P = P[:h].astype(np.float32)
        Q = cv2.resize(Q, (Q.shape[1], h)).astype(np.float32)
        if ov <= 0:
            out = np.hstack([P, Q])
        else:
            a = np.linspace(1, 0, ov)[None, :, None]
            blended = P[:, P.shape[1] - ov:] * a + Q[:, :ov] * (1 - a)
            out = np.hstack([P[:, :P.shape[1] - ov], blended, Q[:, ov:]])
        return np.clip(out, 0, 255).astype(np.uint8)

    def _split_stacked(self, img):
        """Split a tall reconstructed strip into individual photos at white bands."""
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h = g.shape[0]
        white_row = (g >= self.WHITE).mean(axis=1) >= self.WHITE_FRAC
        segs, start = [], None
        for y, w in enumerate(white_row):
            if not w and start is None:
                start = y
            elif w and start is not None:
                segs.append((start, y)); start = None
        if start is not None:
            segs.append((start, h))
        segs = [(a, b) for a, b in segs if b - a > h * self.SEP_MIN_FRAC]
        return [self._trim(img[a:b]) for a, b in segs] or [self._trim(img)]

    # ---- public API (matches SimpleBoxStitcher / AutoCrop) ----
    def load_images(self, input_dir):
        self.images = []
        files = []
        for ext in ("*.webp", "*.jpg", "*.jpeg", "*.png", "*.avif", "*.gif"):
            files += glob.glob(os.path.join(input_dir, ext))
        files.sort(key=self._natural_key)
        for f in files:
            img = self._load(f)
            if img is not None:
                self.images.append({
                    "filename": os.path.basename(f),
                    "h": img.shape[0],
                    "side": self._white_side(img),
                    "trim": self._trim(img),
                })
        return len(self.images)

    def process(self, input_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        if self.load_images(input_dir) == 0:
            return []

        lefts = [it for it in self.images if it["side"] == "right"]   # white margin right => content flush left
        rights = [it for it in self.images if it["side"] == "left"]

        used, pairs = set(), []
        for L in sorted(lefts, key=lambda x: x["h"]):
            cands = sorted([R for R in rights if R["filename"] not in used],
                           key=lambda R: abs(R["h"] - L["h"]))
            if cands:
                used.add(cands[0]["filename"]); pairs.append((L, cands[0]))

        out_paths, n = [], 0
        for (L, R) in pairs:
            A, B = L["trim"], R["trim"]
            cLR = self._join_cost(A, B)
            cRL = self._join_cost(B, A)
            if cRL[0] <= cLR[0]:
                P, Q, ov = B, A, cRL[1]
            else:
                P, Q, ov = A, B, cLR[1]
            full = self._stitch(P, Q, ov)
            for photo in self._split_stacked(full):
                n += 1
                out = os.path.join(output_dir, f"product_{n}.png")
                cv2.imwrite(out, photo)
                out_paths.append(out)

        # fallback: any unmatched single slices -> trim and emit as-is
        if not pairs and self.images:
            for it in self.images:
                for photo in self._split_stacked(it["trim"]):
                    n += 1
                    out = os.path.join(output_dir, f"product_{n}.png")
                    cv2.imwrite(out, photo)
                    out_paths.append(out)

        return out_paths


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 side_slice_stitcher.py <input_dir> [output_dir]")
        sys.exit(1)
    s = SideSliceStitcher()
    res = s.process(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else os.path.join(sys.argv[1], "reconstructed"))
    print(f"Done: {len(res)} product(s) from {len(s.images)} slice(s)")
    for p in res:
        print("  ", p)
