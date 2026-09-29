"""Fixed real Train subset audit and end-to-end memorization, never validation."""
import argparse
import hashlib
import json
import resource
import time
from collections import Counter
from pathlib import Path
import cv2
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from src.datasets.annotations import read_split
from src.datasets.mvfoul import MVFoulDataset, collate_multiview, read_frames, sample_indices
from src.models.vars_model import VARSModel
from src.training.losses import multitask_loss
from src.training.trainer import evaluate
from src.utils.runtime import environment, load_config, seed_everything, write_json

EXPECTED = [str(i) for i in range(64) if i not in (8,20,21,23,24,30,35,39,52,55,56)]


def criterion_met(metrics, initial_loss, criterion):
    """Evaluate current or legacy tiny-overfit criterion dictionaries."""
    shared = criterion.get('accuracy')
    foul = criterion.get('foul_accuracy', shared)
    severity = criterion.get('severity_accuracy', shared)
    if foul is None or severity is None:
        raise ValueError('criterion needs foul/severity_accuracy or legacy accuracy')
    return (metrics['foul']['accuracy'] >= foul and
            metrics['severity']['accuracy'] >= severity and
            metrics['loss_unweighted'] <= initial_loss * (1-criterion['loss_reduction']))


def audit(root, out, config):
    records, report = read_split(root, 'Train', subset_ids=list(map(str, range(64))))
    ids = [r['id'].split('/')[1] for r in records]
    assert ids == EXPECTED, ids
    assert list(report['foul_counts'].values()) == [9,19,5,5,2,2,10,1]
    assert list(report['severity_counts'].values()) == [8,29,15,1]
    assert report['views_per_incident'] == {2:41,3:10,4:2}
    manifest = {'action_ids': ids, 'annotation_sha256': hashlib.sha256((root/'Train/annotations.json').read_bytes()).hexdigest(),
                'selection': 'valid labels among first 64 numerical Train actions; nonrepresentative',
                'report': report, 'records': [{**r, 'paths':[str(p.relative_to(root)) for p in r['paths']]} for r in records]}
    write_json(out/'manifest.json', manifest)
    clips = []
    for record in records:
        sheet = Image.new('RGB', (5*224, len(record['paths'])*150), 'white')
        draw = ImageDraw.Draw(sheet)
        for v, path in enumerate(record['paths']):
            cap = cv2.VideoCapture(str(path))
            fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            frames = read_frames(path, 3.0)
            clips.append({'path':str(path.relative_to(root)), 'fps':fps, 'frame_count':count,
                          'indices':sample_indices(count,fps,3.0).tolist(), 'decoded_shape':list(frames.shape)})
            for col, idx in enumerate((0,4,8,12,15)):
                im=Image.fromarray(frames[idx].permute(1,2,0).numpy()); im.thumbnail((224,126))
                sheet.paste(im,(col*224,v*150+24))
                draw.text((col*224,v*150),f"{record['id']} v{v} f{clips[-1]['indices'][idx]}",fill='black')
        if record['labels'][1] == 3 or record['id'] in ('Train/0','Train/2','Train/4','Train/42'):
            sheet.save(out/(record['id'].replace('/','_')+'.jpg'))
    assert len(clips)==120
    write_json(out/'decode_audit.json',clips)
    return ids


def sync(device):
    if device=='mps': torch.mps.synchronize()
    elif device=='cuda': torch.cuda.synchronize()


def memory(device):
    result={'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    if device=='mps': result.update(mps_allocated_bytes=torch.mps.current_allocated_memory(),mps_driver_bytes=torch.mps.driver_allocated_memory())
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', default='data/SoccerNet/mvfouls')
    p.add_argument('--output',required=True)
    p.add_argument('--config',default='configs/tiny_overfit.yaml')
    p.add_argument('--audit-only',action='store_true')
    p.add_argument('--device',default='auto')
    a=p.parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=False)
    cfg=load_config(a.config); seed_everything(cfg['seed']); torch.set_num_threads(4)
    device=a.device if a.device!='auto' else ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    write_json(out/'config.json',cfg)
    write_json(out/'environment.json',{**environment(),'device':device,'mps_available':torch.backends.mps.is_available(),'hardware':'Apple M3, 8 CPU cores, 16 GB unified memory','threads':4})
    ids=audit(Path(a.data_root).resolve(),out,cfg)
    ds=MVFoulDataset(a.data_root,'Train',subset_ids=ids,**cfg['data'])
    example=collate_multiview([ds[0],ds[4]])
    write_json(out/'input_check.json',{'shape':list(example['video'].shape),'mask':example['mask'].tolist(),'labels':example['labels'].tolist(),'min':example['video'].min().item(),'max':example['video'].max().item(),'finite':bool(torch.isfinite(example['video']).all()),'preprocess':str(ds.preprocess)})
    del example
    if a.audit_only: return
    loader=DataLoader(ds,batch_size=1,collate_fn=collate_multiview,num_workers=0)
    model=VARSModel(**cfg['model']).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=cfg['training']['lr'])
    start=time.perf_counter(); initial=None; steps=0; success=False
    def record(epoch,train_loss=None):
        nonlocal initial
        metrics,predictions=evaluate(model,loader,device)
        if initial is None: initial=metrics['loss_unweighted']
        row={'epoch':epoch,'optimizer_steps':steps,'TRAIN_SET':metrics,'train_online_loss':train_loss,'elapsed_seconds':time.perf_counter()-start,'memory':memory(device)}
        with (out/'history.jsonl').open('a') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
        write_json(out/'latest_train_predictions.json',predictions)
        print(f"epoch={epoch} steps={steps} TRAIN loss={metrics['loss_unweighted']:.6f} foul={metrics['foul']['accuracy']:.4f} severity={metrics['severity']['accuracy']:.4f} elapsed={row['elapsed_seconds']:.1f}s",flush=True)
        return metrics
    try:
        record(0)
        for epoch in range(1,cfg['training']['epochs']+1):
            model.train(); losses=[]
            for index in torch.randperm(len(ds)).tolist():
                batch=collate_multiview([ds[index]]); opt.zero_grad(set_to_none=True)
                groups={'encoder':model.encoder,'aggregation':model.aggregation,'foul_head':model.heads.foul,'severity_head':model.heads.severity}
                if steps==0: before={k:next(m.parameters()).detach().cpu().clone() for k,m in groups.items()}
                sync(device); step_start=time.perf_counter()
                output=model(batch['video'].to(device),batch['mask'].to(device))
                assert all(torch.isfinite(v).all() for v in output.values())
                loss=multitask_loss(output,batch['labels'].to(device)); assert torch.isfinite(loss)
                loss.backward()
                if steps==0:
                    grads={k:sum(p.grad.detach().norm().item() for p in m.parameters() if p.grad is not None) for k,m in groups.items()}
                    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                opt.step(); sync(device); steps+=1; losses.append(loss.item())
                if steps==1:
                    changes={k:(next(m.parameters()).detach().cpu()-before[k]).abs().max().item() for k,m in groups.items()}
                    assert all(v>0 for v in changes.values()) and all(v>0 for v in grads.values())
                    write_json(out/'first_update.json',{'ids':batch['ids'],'shape':list(batch['video'].shape),'loss':loss.item(),'gradient_norm_sums':grads,'parameter_max_changes':changes,'step_seconds':time.perf_counter()-step_start,'memory':memory(device),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'frozen_parameters':sum(p.numel() for p in model.parameters() if not p.requires_grad)})
                    print('First real update verified',flush=True)
                if steps%10==0:
                    print(f'progress epoch={epoch} step={steps} loss={loss.item():.5f} '
                          f'elapsed={time.perf_counter()-start:.1f}s', flush=True)
            # Preserve completed updates before the potentially long full-subset
            # evaluation. This runtime checkpoint is not a validation-selected model.
            torch.save({'model':model.state_dict(),'config':cfg,'epoch':epoch,
                        'optimizer_steps':steps,'pre_evaluation':True},out/'pre_eval.pt')
            if cfg['training'].get('skip_post_epoch_evaluation', False):
                write_json(out/'result.json', {'status':'training_complete_evaluation_deferred',
                    'epochs':epoch,'steps':steps,'initial_loss':initial,
                    'train_online_loss':sum(losses)/len(losses),
                    'elapsed_seconds':time.perf_counter()-start,'exact_resume':False,
                    'checkpoint':'pre_eval.pt'})
                break
            metrics=record(epoch,sum(losses)/len(losses))
            success=criterion_met(metrics, initial, cfg['criterion'])
            if success or epoch%5==0 or epoch==cfg['training']['epochs'] or time.perf_counter()-start>cfg['training']['max_hours']*3600:
                torch.save({'model':model.state_dict(),'config':cfg,'epoch':epoch,'optimizer_steps':steps,'TRAIN_SET':metrics},out/'last.pt')
            if ((success and cfg['training'].get('stop_on_success', True)) or
                    time.perf_counter()-start>cfg['training']['max_hours']*3600):
                break
        if not cfg['training'].get('skip_post_epoch_evaluation', False):
            write_json(out/'result.json',{'success':success,'epochs':epoch,'steps':steps,'initial_loss':initial,'final_train_metrics':metrics,'elapsed_seconds':time.perf_counter()-start,'criterion':cfg['criterion'],'exact_resume':False})
    except Exception as e:
        write_json(out/'failure.json',{'type':type(e).__name__,'message':str(e),'steps':steps,'elapsed_seconds':time.perf_counter()-start})
        raise

if __name__=='__main__': main()
