"""Audit video readability and five-window eligibility without decoding frames."""
import argparse
from collections import Counter
from pathlib import Path

import cv2

from src.datasets.annotations import read_split
from src.utils.runtime import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--splits', nargs='+', default=['Train', 'Valid'])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    reports = []
    for split in args.splits:
        records, dataset_report = read_split(args.data_root, split)
        fps_counts, frame_counts, window_counts = Counter(), Counter(), Counter()
        failures, clips = [], 0
        for record in records:
            for path in record['paths']:
                capture = cv2.VideoCapture(str(path))
                try:
                    if not capture.isOpened():
                        failures.append({'path': str(path), 'reason': 'cannot_open'})
                        continue
                    fps = float(capture.get(cv2.CAP_PROP_FPS))
                    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    if fps <= 0 or frames <= 0:
                        failures.append({'path': str(path), 'reason': 'invalid_metadata',
                                         'fps': fps, 'frames': frames})
                        continue
                    windows = min(5, int((frames / fps) // 1.0))
                    if windows < 1:
                        failures.append({'path': str(path), 'reason': 'shorter_than_one_second',
                                         'fps': fps, 'frames': frames})
                    fps_counts[round(fps, 3)] += 1
                    frame_counts[frames] += 1
                    window_counts[windows] += 1
                    clips += 1
                finally:
                    capture.release()
        report = {'split': split, 'incidents': len(records), 'clips': clips,
                  'fps_histogram': dict(fps_counts),
                  'frame_count_histogram': dict(frame_counts),
                  'window_count_histogram': dict(window_counts),
                  'failures': failures, 'dataset_report': dataset_report}
        reports.append(report)
        print(f'{split}: {clips} clips, {len(failures)} failures, '
              f'windows={dict(window_counts)}', flush=True)
    write_json(Path(args.output), reports)


if __name__ == '__main__':
    main()
