import argparse
from src.datasets.annotations import read_split
from src.utils.runtime import write_json


def main():
    p = argparse.ArgumentParser(description='Audit official splits, labels, exclusions and view correspondence')
    p.add_argument('--data-root', required=True)
    p.add_argument('--splits', nargs='+', default=['Train', 'Valid', 'Test'])
    p.add_argument('--annotations-only', action='store_true')
    p.add_argument('--output', default='experiments/dataset_audit.json')
    args = p.parse_args()
    reports = []
    for split in args.splits:
        records, report = read_split(args.data_root, split, not args.annotations_only)
        report['examples'] = [{'id': x['id'], 'paths': [str(p) for p in x['paths']],
                               'labels': x['labels']} for x in records[:3]]
        reports.append(report)
        print(split, report['retained_incidents'], 'retained;', len(report['excluded']), 'excluded')
    write_json(args.output, reports)


if __name__ == '__main__':
    main()
