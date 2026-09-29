import argparse
import numpy as np
from PIL import Image, ImageDraw
from src.datasets.annotations import read_split
from src.datasets.mvfoul import read_frames


def main():
    p = argparse.ArgumentParser(description='Contact sheet: one row per view, eight times across the input window')
    p.add_argument('--data-root', required=True)
    p.add_argument('--split', default='Valid')
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--center-seconds', type=float, default=3.0)
    p.add_argument('--output', default='incident_preview.png')
    a = p.parse_args()
    records, _ = read_split(a.data_root, a.split)
    item = records[a.index]
    canvas = Image.new('RGB', (8 * 160, len(item['paths']) * 120 + 30), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 5), f"{item['id']} labels={item['labels']} | each row: one camera view", fill='black')
    for row, path in enumerate(item['paths']):
        frames = read_frames(path, a.center_seconds, frames=8).permute(0, 2, 3, 1).numpy()
        for col, frame in enumerate(frames):
            canvas.paste(Image.fromarray(np.asarray(frame)).resize((160, 100)), (col*160, row*120+30))
        draw.text((5, row*120+132), path.name, fill='black')
    canvas.save(a.output)
    print(a.output)


if __name__ == '__main__':
    main()
