import json
import time
from pathlib import Path
import torch
from src.evaluation.metrics import multitask_metrics
from src.training.losses import multitask_loss
from src.utils.runtime import write_json, environment


def forward_batch(model, batch, device):
    key = 'features' if 'features' in batch else 'video'
    value = batch[key].to(device)
    if key == 'features':
        value = value.float()
    return model(value, batch['mask'].to(device))


def train_epoch(model, loader, optimizer, device, class_weights=None,
                loss_settings=None):
    model.train()
    total, count = 0.0, 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        output = forward_batch(model, batch, device)
        loss = multitask_loss(output, batch['labels'].to(device), class_weights,
                              loss_settings)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss')
        loss.backward()
        optimizer.step()
        size = len(batch['ids'])
        total += loss.item() * size
        count += size
    if count == 0:
        raise ValueError('Empty training loader')
    return total / count


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    labels, foul, severity, predictions = [], [], [], []
    total, count = 0.0, 0
    for batch in loader:
        output = forward_batch(model, batch, device)
        fp = output['foul_logits'].softmax(-1).cpu()
        sp = output['severity_logits'].softmax(-1).cpu()
        if batch['labels'] is not None:
            loss = multitask_loss(output, batch['labels'].to(device))
            total += loss.item() * len(fp)
            labels.extend(batch['labels'].tolist())
        count += len(fp)
        foul.extend(fp.tolist())
        severity.extend(sp.tolist())
        for i, incident in enumerate(batch['ids']):
            valid = batch['mask'][i].to(device)
            prediction = {'id': incident, 'paths': batch['paths'][i],
                'foul_probabilities': fp[i].tolist(), 'severity_probabilities': sp[i].tolist(),
                'view_weights': output['attention'][i][valid].cpu().tolist(),
                'calibrated': False}
            if 'foul_attention' in output:
                if output['foul_attention'].ndim == 3:
                    view_count = len(batch['paths'][i])
                    window_count = len(batch['window_starts'][i])
                    prediction.update({
                        'window_starts': batch['window_starts'][i],
                        'foul_spatiotemporal_weights': output['foul_attention'][i, :view_count, :window_count].cpu().tolist(),
                        'severity_spatiotemporal_weights': output['severity_attention'][i, :view_count, :window_count].cpu().tolist(),
                        'foul_view_weights': output['foul_view_attention'][i, :view_count].cpu().tolist(),
                        'severity_view_weights': output['severity_view_attention'][i, :view_count].cpu().tolist(),
                        'foul_temporal_weights': output['foul_temporal_attention'][i, :window_count].cpu().tolist(),
                        'severity_temporal_weights': output['severity_temporal_attention'][i, :window_count].cpu().tolist(),
                    })
                else:
                    prediction['foul_view_weights'] = output['foul_attention'][i][valid].cpu().tolist()
                    prediction['severity_view_weights'] = output['severity_attention'][i][valid].cpu().tolist()
            predictions.append(prediction)
    if not count:
        raise ValueError('Empty evaluation loader')
    metrics = multitask_metrics(labels, foul, severity) if labels else None
    if metrics is not None:
        metrics['loss_unweighted'] = total / count
    return metrics, predictions


def fit(model, train_loader, valid_loader, config, output_dir, device,
        class_weights=None, loss_settings=None):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'last.pt').exists() or (directory / 'history.jsonl').exists():
        raise FileExistsError('Run already exists; choose a new output directory')
    write_json(directory / 'config.json', config)
    write_json(directory / 'environment.json', environment())
    model.to(device)
    train = config['training']
    optimizer_type = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}[train['optimizer']]
    optimizer = optimizer_type(model.parameters(), lr=train['lr'],
                               weight_decay=train['weight_decay'], eps=train['eps'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, train['step_size'], train['gamma'])
    best = float('-inf')
    stale_epochs = 0
    selection_metric = train.get('checkpoint_metric', 'validation_loss')
    patience = train.get('early_stopping_patience')
    min_epochs = train.get('early_stopping_min_epochs', 1)
    for epoch in range(train['epochs']):
        start, lr = time.perf_counter(), optimizer.param_groups[0]['lr']
        loss = train_epoch(model, train_loader, optimizer, device, class_weights,
                           loss_settings)
        metrics, _ = evaluate(model, valid_loader, device)
        if metrics is None:
            raise ValueError('Validation labels required')
        if selection_metric == 'balanced_macro_composite':
            score = sum(
                metrics[task][metric]
                for task in ('foul', 'severity')
                for metric in ('balanced_accuracy', 'macro_f1')
            ) / 4
        elif selection_metric == 'validation_loss':
            score = -metrics['loss_unweighted']
        else:
            raise ValueError(f'Unsupported checkpoint metric: {selection_metric}')
        scheduler.step()
        state = {'epoch': epoch + 1, 'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                 'config': config, 'validation_loss': metrics['loss_unweighted'],
                 'selection_score': score, 'selection_metric': selection_metric}
        torch.save(state, directory / 'last.pt')
        if score > best:
            best = score
            stale_epochs = 0
            torch.save(state, directory / 'best.pt')
        else:
            stale_epochs += 1
        record = {'epoch': epoch + 1, 'lr': lr, 'train_loss': loss,
                  'validation': metrics, 'selection_score': score,
                  'elapsed_seconds': time.perf_counter() - start}
        with (directory / 'history.jsonl').open('a') as handle:
            handle.write(json.dumps(record, allow_nan=False) + '\n')
        print(f"epoch={epoch+1} train_loss={loss:.5f} "
              f"val_loss={metrics['loss_unweighted']:.5f} "
              f"selection={score:.5f} lr={lr:g}", flush=True)
        if (patience is not None and epoch + 1 >= min_epochs
                and stale_epochs >= patience):
            print(f'early_stop epoch={epoch+1} stale_epochs={stale_epochs}', flush=True)
            break
    return directory / 'best.pt'
