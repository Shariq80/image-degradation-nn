







"""Evaluate RESISC45 ResNet-18 checkpoints under test-time degradation."""
import argparse
import csv
from pathlib import Path
import torch
from torch import nn
from torchvision import models
from dataset import get_dataloader


def load_model(path, device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 45)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(device).eval()


def metrics(cm):
    cm = cm.double()
    tp = cm.diag()
    denominator = cm.sum(0) + cm.sum(1)
    f1 = torch.where(denominator > 0, 2 * tp / denominator, 0.0)
    return 100 * (tp.sum() / cm.sum()).item(), 100 * f1.mean().item()


def conditions():
    for scale in (1.0, 0.75, 0.5, 0.25):
        yield scale, None, 0.0
        for noise, levels in [('gaussian', (0.05, 0.1, 0.2)),
                              ('salt_and_pepper', (0.01, 0.05, 0.1))]:
            for level in levels:
                yield scale, noise, level


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen', default='checkpoints/best_baseline_frozen.pth')
    p.add_argument('--finetuned', default='checkpoints/best_finetuned.pth')
    p.add_argument('--model', choices=['both', 'frozen', 'finetuned'], default='both')
    p.add_argument('--data-dir', default='NWPU-RESISC45')
    p.add_argument('--splits', default='splits.json')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--output', default='results')
    p.add_argument('--clean-only', action='store_true')
    args = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    paths = {'frozen': args.frozen, 'finetuned': args.finetuned}
    selected = paths if args.model == 'both' else {args.model: paths[args.model]}
    for path in [*selected.values(), args.splits, args.data_dir]:
        if not Path(path).exists():
            p.error(f'Missing required file or directory: {path}')
    nets = {name: load_model(path, device) for name, path in selected.items()}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    clean = {}
    rows = []
    cases = [(1.0, None, 0.0)] if args.clean_only else list(conditions())
    with torch.inference_mode():
        for scale, noise, level in cases:
            loader = get_dataloader(args.splits, args.data_dir, split='test',
                batch_size=args.batch_size, num_workers=args.workers,
                res_scale=scale, noise_type=noise, noise_level=level)
            matrices = {name: torch.zeros((45, 45), dtype=torch.long) for name in nets}
            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.cpu()
                for name, net in nets.items():
                    pred = net(inputs).argmax(1).cpu()
                    matrices[name] += torch.bincount(labels * 45 + pred, minlength=2025).reshape(45,45)
            for name, cm in matrices.items():
                acc, f1 = metrics(cm)
                if scale == 1.0 and noise is None:
                    clean[name] = acc
                rows.append(dict(model=name, resolution_scale=scale,
                    resolution_reduction_pct=round(100*(1-scale)), noise_type=noise or 'none',
                    noise_level=level, test_images=int(cm.sum()), accuracy_pct=round(acc, 6),
                    macro_f1_pct=round(f1, 6), accuracy_drop_pp=round(clean[name]-acc, 6)))
                print(f'{name}: scale={scale}, noise={noise}, level={level}: accuracy={acc:.2f}%, macro F1={f1:.2f}%', flush=True)
                with (out / f'confusion_{name}_{scale}_{noise or "none"}_{level}.csv').open('w', newline='') as f:
                    writer = csv.writer(f)
                    names = loader.dataset.class_names
                    writer.writerow(['true/predicted', *names])
                    writer.writerows([label, *row] for label, row in zip(names, cm.tolist()))
    suffix = '_clean' if args.clean_only else ''
    result = out / f'evaluation_{args.model}{suffix}.csv'
    with result.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Results saved to {result}')


if __name__ == '__main__':
    main()





