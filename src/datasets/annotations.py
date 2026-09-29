"""Strict adapter for the official MVFoul Actions annotation schema."""
import json
from collections import Counter
from pathlib import Path

# Official index order, not the order in which classes appear in the paper.
FOUL_CLASSES = ('Tackling', 'Standing tackling', 'High leg', 'Holding',
                'Pushing', 'Elbowing', 'Challenge', 'Dive/Simulation')
SEVERITY_CLASSES = ('No offence', 'Offence + No card',
                    'Offence + Yellow card', 'Offence + Red card')
SPLITS = ('Train', 'Valid', 'Test', 'Chall')


def parse_labels(action):
    """Return ((type_id, severity_id), None), or (None, exclusion_reason).

    Implements the official loader's missing/borderline policy, including its
    Dive fallback. Unknown values fail loudly instead of becoming no-offence.
    """
    kind, offence, severity = (action[k] for k in ('Action class', 'Offence', 'Severity'))
    if kind in ('', 'Dont know'):
        return None, 'unknown_action'
    names = {name: i for i, name in enumerate(FOUL_CLASSES)}
    names['Dive'] = 7
    if kind not in names:
        raise ValueError(f'Unknown Action class: {kind!r}')
    dive = names[kind] == 7
    no_offence = offence in ('No offence', 'No Offence')
    if offence in ('', 'Between') and not dive:
        return None, 'ambiguous_offence'
    if severity in ('', '2.0', '4.0') and not dive and not no_offence:
        return None, 'ambiguous_severity'
    if offence in ('', 'Between'):
        offence = 'Offence'
    if severity in ('', '2.0', '4.0'):
        severity = '1.0'
    if no_offence:
        level = 0
    elif offence == 'Offence' and severity in ('1.0', '3.0', '5.0'):
        level = {'1.0': 1, '3.0': 2, '5.0': 3}[severity]
    else:
        raise ValueError(f'Unknown offence/severity: {offence!r}/{severity!r}')
    return (names[kind], level), None


def read_split(root, split, check_files=True, subset_ids=None):
    if split not in SPLITS:
        raise ValueError(f'split must be one of {SPLITS}')
    folder = Path(root).expanduser().resolve() / split
    annotation = folder / 'annotations.json'
    if not annotation.is_file():
        raise FileNotFoundError(f'Missing {annotation}. See data/README.md; no data is bundled.')
    actions = json.loads(annotation.read_text())['Actions']
    if not isinstance(actions, dict):
        raise ValueError('Expected Actions to be an object keyed by incident ID')
    selected = None
    if subset_ids is not None:
        selected = [str(x) for x in subset_ids]
        if not selected or len(set(selected)) != len(selected):
            raise ValueError('subset_ids must be nonempty and unique')
        if any(not x.isdigit() for x in selected):
            raise ValueError('subset_ids must contain numeric action IDs')
        missing_ids = set(selected) - actions.keys()
        if missing_ids:
            raise ValueError(f'Unknown subset IDs: {sorted(missing_ids)}')
    records, excluded = [], []
    for key, action in (actions.items() if selected is None else ((k, actions[k]) for k in selected)):
        if not str(key).isdigit():
            raise ValueError(f'Expected numeric action ID, got {key!r}')
        labels, reason = (None, None) if split == 'Chall' else parse_labels(action)
        if reason:
            excluded.append({'id': str(key), 'reason': reason})
            continue
        clips = action['Clips']
        if not isinstance(clips, (list, dict)) or len(clips) < 2:
            raise ValueError(f'{split}/{key}: expected at least two annotated views')
        paths = [folder / f'action_{key}' / f'clip_{i}.mp4' for i in range(len(clips))]
        if check_files:
            missing = [str(p) for p in paths if not p.is_file()]
            if missing:
                raise FileNotFoundError(f'Missing annotated video(s): {missing}')
        records.append({'id': f'{split}/{key}', 'paths': paths, 'labels': labels})
    if not records:
        raise ValueError(f'No usable incidents in {annotation}')
    report = {'split': split, 'raw_incidents': len(actions), 'selected_incidents': len(actions) if selected is None else len(selected), 'retained_incidents': len(records),
              'excluded': excluded, 'exclusion_counts': dict(Counter(x['reason'] for x in excluded)),
              'views_per_incident': dict(Counter(len(x['paths']) for x in records))}
    for name, col, classes in [('foul', 0, FOUL_CLASSES), ('severity', 1, SEVERITY_CLASSES)]:
        counts = Counter(x['labels'][col] for x in records if x['labels'] is not None)
        report[name + '_counts'] = {label: counts[i] for i, label in enumerate(classes)}
    return records, report
