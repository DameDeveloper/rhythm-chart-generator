"""Generate the app icon (app.ico) — a purple lightning bolt on a dark rounded
square, matching the frontend favicon. Run: python assets/make_icon.py"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent
SS = 4  # supersampling factor for crisp anti-aliased edges
BASE = 256
S = BASE * SS

BG = (18, 18, 38, 255)       # dark navy, close to app background #0e0e1a
BG_EDGE = (58, 58, 90, 255)  # subtle border (#3a3a5a)
BOLT = (134, 59, 255, 255)   # brand purple #863bff
BOLT_HI = (192, 132, 252, 255)  # highlight #c084fc


def rounded_rect(draw: ImageDraw.ImageDraw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def main() -> None:
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # Background rounded square with a thin edge.
    margin = int(S * 0.06)
    rounded_rect(
        d,
        [margin, margin, S - margin, S - margin],
        radius=int(S * 0.22),
        fill=BG,
        outline=BG_EDGE,
        width=max(2, int(S * 0.01)),
    )

    # Lightning bolt polygon (normalized 0..1), pointing down.
    bolt = [
        (0.56, 0.10),
        (0.30, 0.52),
        (0.47, 0.52),
        (0.40, 0.90),
        (0.72, 0.44),
        (0.54, 0.44),
    ]
    pts = [(x * S, y * S) for (x, y) in bolt]

    # Soft drop shadow for depth.
    shadow = [(x + S * 0.012, y + S * 0.012) for (x, y) in pts]
    d.polygon(shadow, fill=(0, 0, 0, 120))
    # Main bolt.
    d.polygon(pts, fill=BOLT)
    # Highlight strip near the top for a subtle glossy look.
    hi = [pts[0], pts[1], ((pts[1][0] + pts[2][0]) / 2, pts[1][1])]
    d.polygon(hi, fill=BOLT_HI)

    icon = img.resize((BASE, BASE), Image.LANCZOS)
    icon.save(OUT / "app_preview.png")
    sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    icon.save(OUT / "app.ico", sizes=sizes)
    print("wrote", OUT / "app.ico", "and app_preview.png")


if __name__ == "__main__":
    main()
