# Copyright 2026 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Matched-feature tokenizer initialization and joint DIGER training."""
import hashlib
import math
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .diger import SemanticTokenizer


def tensor_hash(value):
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str(array.shape).encode() + array.tobytes()).hexdigest()


def pretrain_tokenizer(features, options=None, *, epochs=100, batch_size=256,
                       learning_rate=1e-3, seed=2024,
                       warmup_epochs=20):
    """Learn a matched-width RQ autoencoder initializer on catalog features.

    This explicitly declared shared-feature initializer uses deterministic nearest
    residual codes, straight-through reconstruction, and codebook/commitment MSE.
    It is an adaptation, not a claim to reproduce an external released initializer.
    Padding is excluded. The decoder is discarded before recommendation training.
    """
    if features.ndim != 2 or len(features) < 2 or not torch.isfinite(features).all():
        raise ValueError("Require finite features with padding plus catalog rows")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    torch.manual_seed(seed)
    options = dict(options or {})
    options['freeze_encoder'] = False
    tokenizer = SemanticTokenizer(features.shape[1], **options).to(features.device)
    widths = (options.get('latent_dim', 256),) + tuple(reversed(options.get('hidden_dims', (2048, 1024, 512)))) + (features.shape[1],)
    layers = []
    for index, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(nn.Linear(left, right))
        if index < len(widths)-2:
            layers.append(nn.ReLU())
    decoder = nn.Sequential(*layers).to(features.device)
    optimizer = torch.optim.AdamW(list(tokenizer.parameters()) + list(decoder.parameters()), lr=learning_rate)
    # Reconstruction warmup learns a nonconstant feature representation before
    # nearest-code training. Residual Lloyd initialization avoids trapping all
    # normalized semantic vectors in the same tiny random code vector.
    warmup_optimizer = torch.optim.AdamW(list(tokenizer.encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
    reconstruction_scale = features[1:].var(dim=0, unbiased=False).mean().clamp_min(1e-8)
    warmup_losses = []
    for epoch in range(warmup_epochs):
        total = 0.
        for ids in (torch.randperm(len(features)-1, device=features.device)+1).split(batch_size):
            loss = F.mse_loss(decoder(tokenizer.encoder(features[ids])), features[ids])/reconstruction_scale
            warmup_optimizer.zero_grad(); loss.backward(); warmup_optimizer.step()
            total += float(loss.detach())*len(ids)
        warmup_losses.append(total/(len(features)-1))
    with torch.no_grad():
        residual = tokenizer.encoder(features[1:])
        for book in tokenizer.codebooks:
            centers = residual[torch.randint(len(residual), (book.num_embeddings,), device=features.device)].clone()
            for _ in range(20):
                assignments = tokenizer.distances(residual, centers).argmin(-1)
                counts = torch.bincount(assignments, minlength=len(centers))
                sums = torch.zeros_like(centers).index_add_(0, assignments, residual)
                occupied = counts > 0
                centers[occupied] = sums[occupied]/counts[occupied, None]
            book.weight.copy_(centers)
            residual = residual-centers[assignments]
    losses = []
    for epoch in range(epochs):
        order = torch.randperm(len(features)-1, device=features.device)+1
        total = 0.
        for ids in order.split(batch_size):
            quantized = tokenizer(features[ids], mode='plain')
            straight = quantized.latent + (quantized.quantized-quantized.latent).detach()
            loss = F.mse_loss(decoder(straight), features[ids])/reconstruction_scale + quantized.vq_per_item.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(ids)
        losses.append(total/(len(features)-1))
    tokenizer.eval(); decoder.eval()
    with torch.no_grad():
        final_reconstruction = 0.
        for values in features[1:].split(batch_size):
            quantized = tokenizer(values, mode='plain')
            final_reconstruction += float(F.mse_loss(decoder(quantized.quantized), values)/reconstruction_scale)*len(values)
        final_reconstruction /= len(features)-1
    bundle = {'format': 'cornac-diger-initializer-v1', 'semantic_width': features.shape[1],
              'feature_hash': tensor_hash(features), 'tokenizer_options': options,
              'terminal_normalized_reconstruction': final_reconstruction,
              'mean_feature_baseline_normalized_reconstruction': 1.0,
              'pretrain_config': {'epochs': epochs, 'warmup_epochs': warmup_epochs, 'batch_size': batch_size, 'learning_rate': learning_rate, 'weight_decay': .01, 'lloyd_iterations': 20},
              'state': {k: v.detach().cpu() for k, v in tokenizer.state_dict().items()},
              'losses': losses, 'warmup_losses': warmup_losses, 'reconstruction_variance_scale': float(reconstruction_scale), 'warmup_epochs': warmup_epochs, 'seed': seed,
              'recipe': 'matched-feature AE warmup + residual Lloyd20 + variance-scaled nearest-code ST reconstruction + VQ'}
    return bundle


def batch(rows, indices, device, max_length):
    selected = [rows[int(i)] for i in indices]
    histories = [history[-max_length:]
                 for history, _ in selected]
    padded = torch.zeros(len(selected), max(map(len, histories)), dtype=torch.long, device=device)
    for index, history in enumerate(histories):
        padded[index, :len(history)] = torch.tensor(history, device=device)
    return padded, torch.tensor([target for _, target in selected], device=device)


def validate_rows(rows, catalog_size):
    if not rows:
        raise ValueError('Training and validation examples must be nonempty')
    for history, target in rows:
        if not history or any(not 1 <= int(i) <= catalog_size for i in history) or not 1 <= int(target) <= catalog_size:
            raise ValueError('Example contains empty history, padding, or unknown item')


def train(backend, train_rows, validation_rows, *, config, verbose=False):
    """Select generator, tokenizer and catalog together by validation NDCG@10."""
    validate_rows(train_rows, len(backend.features)-1)
    validate_rows(validation_rows, len(backend.features)-1)
    device = backend.features.device
    steps = math.ceil(len(train_rows)/config['batch_size'])*config['epochs']
    parameters = list(backend.tokenizer.codebooks.parameters())
    backend.tokenizer.requires_grad_(True)
    backend.tokenizer.encoder.requires_grad_(False)
    rec = torch.optim.AdamW(backend.generator.parameters(), lr=config['lr_rec'], weight_decay=config['weight_decay'])
    ids = torch.optim.AdamW(parameters, lr=config['lr_id'], weight_decay=config['weight_decay'])
    def schedule(step):
        warmup = config['warmup_steps']
        if step < warmup:
            return step/max(1, warmup)
        return max(0., .5*(1+math.cos(math.pi*(step-warmup)/max(1, steps-warmup))))
    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, schedule) for opt in (rec, ids)]
    global_step, best_score, best_epoch, stale = 0, -float('inf'), None, 0
    best_state = None
    history = []
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    catalog = backend.catalog()
    for epoch in range(config['epochs']):
        backend.train()
        total = 0.
        order = torch.randperm(len(train_rows))
        for indices in order.split(config['batch_size']):
            h, t = batch(train_rows, indices, device, config['max_length'])
            rec.zero_grad(); ids.zero_grad()
            result = backend.batch_loss(h, t, catalog)
            result['loss'].backward()
            nn.utils.clip_grad_norm_(backend.generator.parameters(), 1.)
            nn.utils.clip_grad_norm_(parameters, 1.)
            rec.step()
            ids.step()
            for scheduler in schedulers:
                scheduler.step()
            global_step += 1
            total += float(result['loss'].detach())*len(indices)
        backend.eval()
        catalog = backend.catalog()
        ndcg = 0.
        with torch.no_grad():
            for indices in torch.arange(len(validation_rows)).split(config['eval_batch_size']):
                h, t = batch(validation_rows, indices, device, config['max_length'])
                paths, _ = backend.generator.beam_search(catalog[h], h != 0, beams=config['beams'])
                matches = (paths[:, :10] == catalog[t, None]).all(-1)
                discounts = 1/torch.log2(torch.arange(matches.shape[1], device=device)+2.)
                ndcg += float((matches*discounts).sum())
        score = ndcg/len(validation_rows)
        history.append({'epoch': epoch, 'loss': total/len(train_rows), 'ndcg@10': score})
        if best_epoch is None or score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {'state': {k: v.detach().cpu().clone() for k, v in backend.state_dict().items()},
                          'catalog': catalog.detach().cpu(), 'epoch': epoch, 'ndcg@10': score}
        else:
            stale += 1
        if verbose:
            print(f'DIGER epoch {epoch}: loss={history[-1]["loss"]:.6f}, ndcg@10={score:.6f}', flush=True)
        if stale >= config['patience']:
            break
    if best_state is None:
        raise ValueError('No training epochs remain')
    backend.load_state_dict(best_state['state'], strict=True)
    backend.eval()
    return best_state, {'best_epoch': best_epoch, 'best_score': best_score,
                        'global_step': global_step, 'history': history}
