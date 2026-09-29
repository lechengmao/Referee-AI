import json
import cv2
import numpy as np
import pytest
import torch
from src.datasets.annotations import parse_labels, read_split
from src.datasets.mvfoul import (MVFoulDataset, collate_multiview, sample_indices,
                                 temporal_window_indices, read_temporal_windows)
from src.utils.synthetic import SyntheticIncidents


@pytest.mark.parametrize('offence,severity,expected', [
    ('No offence', '', 0), ('No Offence', '4.0', 0),
    ('Offence', '1.0', 1), ('Offence', '3.0', 2), ('Offence', '5.0', 3)])
def test_severity_mapping(offence, severity, expected):
    labels, reason = parse_labels({'Action class': 'Tackling', 'Offence': offence, 'Severity': severity})
    assert reason is None and labels == (0, expected)


def test_borderline_policy():
    a = {'Action class': 'Tackling', 'Offence': 'Offence', 'Severity': '4.0'}
    assert parse_labels(a) == (None, 'ambiguous_severity')
    a.update({'Action class': 'Dive', 'Offence': 'Between'})
    assert parse_labels(a) == ((7, 1), None)
    a['Action class'] = 'invented'
    with pytest.raises(ValueError):
        parse_labels(a)


def test_grouping_exclusions_and_missing_files(tmp_path):
    split = tmp_path / 'Train'
    split.mkdir()
    good = {'Action class': 'Holding', 'Offence': 'Offence', 'Severity': '3.0', 'Clips': [{}, {}]}
    bad = {**good, 'Severity': '2.0'}
    (split / 'annotations.json').write_text(json.dumps({'Actions': {'19': good, '2': bad}}))
    records, report = read_split(tmp_path, 'Train', check_files=False)
    assert records[0]['id'] == 'Train/19'
    assert records[0]['paths'][1] == split / 'action_19' / 'clip_1.mp4'
    assert report['exclusion_counts'] == {'ambiguous_severity': 1}
    with pytest.raises(FileNotFoundError):
        read_split(tmp_path, 'Train')


def test_temporal_window():
    indices = sample_indices(125, 25, 3.0)
    assert len(indices) == 16 and indices.min() >= 62 and indices.max() <= 87
    assert (indices < 75).sum() == 8
    with pytest.raises(ValueError):
        sample_indices(20, 25, 3.0)


def test_five_nonoverlapping_windows():
    indices = temporal_window_indices(125, 25)
    assert indices.shape == (5, 16)
    assert indices[0].min() == 0 and indices[-1].max() < 125
    assert all(indices[i].max() < indices[i + 1].min() for i in range(4))
    assert temporal_window_indices(75, 25).shape == (3, 16)
    with pytest.raises(ValueError):
        temporal_window_indices(20, 25)


def test_variable_view_batch():
    ds = SyntheticIncidents()
    batch = collate_multiview([ds[0], ds[2]])
    assert batch['video'].shape == (2, 3, 3, 4, 8, 8)
    assert batch['mask'].tolist() == [[True, False, False], [True, True, True]]
    assert batch['labels'].shape == (2, 2)


def test_video_decode_preprocess(tmp_path):
    folder = tmp_path / 'Valid' / 'action_0'
    folder.mkdir(parents=True)
    for i in range(2):
        writer = cv2.VideoWriter(str(folder / f'clip_{i}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 25, (64, 48))
        assert writer.isOpened()
        for frame in range(125):
            writer.write(np.full((48, 64, 3), frame, dtype=np.uint8))
        writer.release()
    action = {'Action class': 'High leg', 'Offence': 'Offence', 'Severity': '5.0', 'Clips': [{}, {}]}
    (folder.parent / 'annotations.json').write_text(json.dumps({'Actions': {'0': action}}))
    dataset = MVFoulDataset(tmp_path, 'Valid')
    a, b = dataset[0], dataset[0]
    assert a['video'].shape == (2, 3, 16, 224, 224) and a['labels'] == (2, 3)
    torch.testing.assert_close(a['video'], b['video'])
    assert torch.isfinite(a['video']).all()
    windows, indices = read_temporal_windows(folder / 'clip_0.mp4')
    assert windows.shape == (5, 16, 3, 48, 64)
    assert indices.shape == (5, 16)


def test_subset_before_file_checks(tmp_path):
    folder=tmp_path/'Train'; folder.mkdir()
    good={'Action class':'Holding','Offence':'Offence','Severity':'3.0','Clips':[{},{}]}
    (folder/'annotations.json').write_text(json.dumps({'Actions':{'0':good,'99':good}}))
    (folder/'action_0').mkdir()
    for i in range(2): (folder/'action_0'/f'clip_{i}.mp4').touch()
    records,report=read_split(tmp_path,'Train',subset_ids=['0'])
    assert len(records)==1 and report['selected_incidents']==1
    with pytest.raises(FileNotFoundError): read_split(tmp_path,'Train')
    with pytest.raises(FileNotFoundError): read_split(tmp_path,'Train',subset_ids=['99'])
    for ids in ([], ['0','0'], ['100'], ['../0']):
        with pytest.raises(ValueError): read_split(tmp_path,'Train',subset_ids=ids)


def test_deterministic_train_all_views(tmp_path, monkeypatch):
    folder=tmp_path/'Train'; folder.mkdir()
    good={'Action class':'Holding','Offence':'Offence','Severity':'3.0','Clips':[{}, {}, {}]}
    (folder/'annotations.json').write_text(json.dumps({'Actions':{'0':good}}))
    (folder/'action_0').mkdir()
    for i in range(3): (folder/'action_0'/f'clip_{i}.mp4').touch()
    shifts=[]
    def decode(path,center,frames,duration,shift):
        shifts.append(shift)
        return torch.zeros(16,3,8,8,dtype=torch.uint8)
    monkeypatch.setattr('src.datasets.mvfoul.read_frames',decode)
    ds=MVFoulDataset(tmp_path,'Train',subset_ids=['0'],deterministic=True)
    a,b=ds[0],ds[0]
    assert len(a['paths'])==3 and a['paths']==b['paths'] and shifts==[0.0]*6
    assert ds.augment is None and ds.report['split']=='Train'
    torch.testing.assert_close(a['video'],b['video'])


def test_deterministic_fixed_views(tmp_path, monkeypatch):
    folder=tmp_path/'Train'; folder.mkdir()
    good={'Action class':'Holding','Offence':'Offence','Severity':'3.0','Clips':[{}, {}, {}]}
    (folder/'annotations.json').write_text(json.dumps({'Actions':{'0':good}}))
    (folder/'action_0').mkdir()
    for i in range(3): (folder/'action_0'/f'clip_{i}.mp4').touch()
    monkeypatch.setattr('src.datasets.mvfoul.read_frames',
                        lambda *args, **kwargs: torch.zeros(16,3,8,8,dtype=torch.uint8))
    ds=MVFoulDataset(tmp_path,'Train',subset_ids=['0'],deterministic=True,
                     deterministic_views=2)
    item=ds[0]
    assert len(item['paths']) == 2
    assert item['paths'][0].endswith('clip_0.mp4')
    assert item['paths'][1].endswith('clip_1.mp4')
