import math
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification
import argparse
import os
import sys
import tqdm
import inspect
import logging

from notimedata import AZDataset, AZDataCollator
from utils import *
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
logging.disable(logging.WARNING) # disable WARNING, INFO and DEBUG logging everywhere

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#eval_results, ppl, loss = evaluate(val_dataloader, tokenizer, ctx, model)
@torch.no_grad()
def evaluate(dataloader, tokenizer, ctx, model, threshold, logits_all, labels_all):
    model.eval()
    if logits_all is None:
        predictions_all = []
        labels_all = []
        total_loss = 0
        total = 0
        logits_all = []
        for batch in tqdm.tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            max_length = attention_mask.sum(-1).max()
            input_ids = input_ids[:, :max_length]
            attention_mask = attention_mask[:, :max_length]
            batch_size = input_ids.shape[0]
            with ctx:
                outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            loss = outputs.loss
            logits = outputs.logits
            total_loss += loss.item() * batch_size
            total += batch_size
            #import pdb; pdb.set_trace()
            logits = logits.softmax(dim=-1)
            predictions = (logits[:, 1] > threshold).long()
            #predictions = torch.argmax(logits, dim=-1)
            predictions_all.append(predictions.cpu())
            labels_all.append(labels.cpu())
            logits_all.append(logits.cpu())
        logits_all = torch.cat(logits_all, dim=0)
        predictions_all = torch.cat(predictions_all, dim=0)
        labels_all = torch.cat(labels_all, dim=0)
    else:
        predictions_all = (logits_all[:, 1] > threshold).long()
        total_loss = 0
        total = 1e-9

    eval_results = compute_metrics(predictions_all, labels_all)
    loss = total_loss/total
    ppl = math.exp(loss)
    model.train()
    return eval_results, ppl, loss, logits_all, labels_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default='microsoft/deberta-v3-large')
    parser.add_argument('--data_path', type=str, default='dataset.pt')
    parser.add_argument('--save_model', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--truncation', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--accumulate', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--target_ratio', type=float, default=0.5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--bf16', action='store_true')
    parser.set_defaults(bf16=False)
    parser.add_argument('--nomultiplier', action='store_true')
    parser.set_defaults(nomultiplier=False)
    args = parser.parse_args()

    print (args)
    data = torch.load(args.data_path)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=2)

    dtype = 'float32'
    if args.bf16:
        dtype = 'bfloat16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)
    print (ptdtype, dtype, device)
    #import pdb; pdb.set_trace()

    model = model.to(device).to(ptdtype)
    # Load data
    collate_fn = AZDataCollator(tokenizer)
    train_dataset = AZDataset(tokenizer, data['train'], args.truncation, target_ratio=args.target_ratio)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    val_dataset = AZDataset(tokenizer, data['valid'], args.truncation)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    test_dataset = AZDataset(tokenizer, data['test'], args.truncation)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    # Create Optimizer
    trainable_params = list(model.parameters())
    use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, **extra_args)

    # Train
    step = 0
    best_ppl = float('inf')

    steps_per_epoch = int(len(train_dataloader) / train_dataset.multiplier)
    print (f'steps_per_epoch: {steps_per_epoch}, multiplier: {train_dataset.multiplier}')
    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")
        model.train()
        #import pdb; pdb.set_trace()
        for batch in tqdm.tqdm(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            max_length = attention_mask.sum(-1).max()
            input_ids = input_ids[:, :max_length]
            attention_mask = attention_mask[:, :max_length]
            with ctx:
                outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            #import pdb; pdb.set_trace()
            if args.nomultiplier:
                train_dataset.multiplier = 1
            loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0/train_dataset.multiplier]).to(device))
            loss = outputs.loss
            loss = loss_fn(logits, labels)
            loss.div(args.accumulate).backward()
            if step % args.accumulate == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            ppl = loss.exp().item()
            if step % 100 == 0:
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                eval_results = compute_metrics(predictions, labels)
                print (f"Step: {step}. PPL: {ppl}. Los: {loss}")
                for k in eval_results:
                    print (f'  - {k}: {eval_results[k]}')
                sys.stdout.flush()
            if step % steps_per_epoch == 0:
                thresholds = [0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.07, 0.05, 0.03, 0.01]
                logits_val = None
                logits_test = None
                labels_val = None
                labels_test = None
                for threshold in thresholds:
                    print (f'threshold: {threshold}')
                    eval_results, ppl, loss, logits_val, labels_val = evaluate(val_dataloader, tokenizer, ctx, model, threshold, logits_val, labels_val)
                    if threshold == thresholds[0] and ppl < best_ppl:
                        best_ppl = ppl
                        is_best = True
                    else:
                        is_best = False
                    if threshold == thresholds[0]:
                        if is_best:
                            print (f"Val. {step}. PPL: {ppl}. Los: {loss} (best)")
                        else:
                            print (f"Val. {step}. PPL: {ppl}. Los: {loss}")
                    for k in eval_results:
                        print (f'  val - {k}: {eval_results[k]}')
                    eval_results, ppl, loss, logits_test, labels_test = evaluate(test_dataloader, tokenizer, ctx, model, threshold, logits_test, labels_test)
                    if threshold == thresholds[0]:
                        print (f"Test. {step}.  PPL: {ppl}. Los: {loss}")
                    for k in eval_results:
                        print (f'  test - {k}: {eval_results[k]}')
                    print ('='*20)
                    if is_best:
                        print ('saving best')
                        model.save_pretrained(os.path.join(args.save_model, f'checkpoint_{epoch}_step_{step}'))
                sys.stdout.flush()
            step += 1
        model.save_pretrained(os.path.join(args.save_model, f'checkpoint_{epoch}'))

if __name__ == "__main__":
    main()
