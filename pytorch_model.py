import logging

import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from transformers import BertForMaskedLM, AdamW, get_linear_schedule_with_warmup, set_seed, TrainingArguments
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from utils import InputFeatures
from progress_bar import ProgressBar

from early_stoping import EarlyStopping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask, all_token_type_ids, all_lens, label = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    label = label[:, :max_len]
    return all_input_ids, all_attention_mask, all_token_type_ids, label


class Processor:
    def __init__(self, tokens):
        self.tokens = tokens

    def random_mask(self, text_ids):
        """随机mask
        """
        input_ids, output_ids = [], []
        rands = np.random.random(len(text_ids))
        for r, i in zip(rands, text_ids):
            if r < 0.15 * 0.8:
                input_ids.append(4)
                output_ids.append(i)
            elif r < 0.15 * 0.9:
                input_ids.append(i)
                output_ids.append(i)
            elif r < 0.15:
                input_ids.append(np.random.choice(len(self.tokens)) + 7)
                output_ids.append(i)
            else:
                input_ids.append(i)
                output_ids.append(0)
        return input_ids, output_ids

    def sample_convert(self, text1, text2, label, random=False):
        """转换为MLM格式
        """
        text1_ids = [self.tokens.get(t, 1) for t in text1]
        text2_ids = [self.tokens.get(t, 1) for t in text2]
        if random:
            if np.random.random() < 0.5:
                text1_ids, text2_ids = text2_ids, text1_ids
            text1_ids, out1_ids = self.random_mask(text1_ids)
            text2_ids, out2_ids = self.random_mask(text2_ids)
        else:
            out1_ids = [0] * len(text1_ids)
            out2_ids = [0] * len(text2_ids)
        token_ids = [2] + text1_ids + [3] + text2_ids + [3]
        segment_ids = [0] * len(token_ids)
        output_ids = [label + 5] + out1_ids + [0] + out2_ids + [0]
        return token_ids, segment_ids, output_ids

    @staticmethod
    def sequence_padding(inputs, length=None, padding=0, mode='post'):
        """Numpy函数，将序列padding到同一长度
        """
        if length is None:
            length = max([len(x) for x in inputs])

        pad_width = [(0, 0) for _ in np.shape(inputs[0])]
        outputs = []
        for x in inputs:
            x = x[:length]
            if mode == 'post':
                pad_width[0] = (0, length - len(x))
            elif mode == 'pre':
                pad_width[0] = (length - len(x), 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
            x = np.pad(x, pad_width, 'constant', constant_values=padding)
            outputs.append(x)

        return np.array(outputs)

    def get_examples(self, data, random=False):
        all_token_ids, all_attention_mask, all_segment_ids, all_input_len, all_label = [], [], [], [], []
        for i, line in enumerate(data):
            text1, text2, label = line
            token_ids, segment_ids, output_ids = self.sample_convert(text1, text2, label, random)
            attention_mask = [1] * len(token_ids)
            input_len = len(token_ids)
            all_token_ids.append(token_ids)
            all_attention_mask.append(attention_mask)
            all_segment_ids.append(segment_ids)
            all_input_len.append(input_len)
            all_label.append(output_ids)
        max_len = max(all_input_len)
        all_token_ids = self.sequence_padding(all_token_ids, length=max_len)
        all_attention_mask = self.sequence_padding(all_attention_mask, length=max_len)
        all_segment_ids = self.sequence_padding(all_segment_ids, length=max_len)
        all_label = self.sequence_padding(all_label, length=max_len)
        features = [
            InputFeatures(
                input_ids=all_token_ids[j],
                attention_mask=all_attention_mask[j],
                token_type_ids=all_segment_ids[j],
                label=all_label[j],
                input_len=all_input_len[j]
            ) for j in range(len(data))
        ]
        return features


class Model:
    def __init__(self, path):
        self.model_dir = path
        self.model = None

    def prepare_dataset(self, data):
        all_input_ids = torch.tensor([f.input_ids for f in data], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in data], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in data], dtype=torch.long)
        all_lens = torch.tensor([f.input_len for f in data], dtype=torch.long)
        label = torch.tensor([f.label for f in data], dtype=torch.long)
        inputs = [all_input_ids, all_attention_mask, all_token_type_ids, all_lens, label]
        dataset = TensorDataset(*inputs)
        return dataset

    def train(self, train_data, eval_data=None, args=None):
        if args is None:
            output_dir = "tmp_trainer"
            logger.info(f"No `TrainingArguments` passed, using `output_dir={output_dir}`.")
            args = TrainingArguments(output_dir=output_dir)
        self.load(self.model_dir)
        early_stopping = EarlyStopping(verbose=True)
        # args.n_gpu = torch.cuda.device_count()
        # args.train_batch_size = args.per_device_train_batch_size * max(1, args.n_gpu)
        dataset = self.prepare_dataset(train_data)
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.train_batch_size, collate_fn=collate_fn)
        if eval_data:
            eval_dataset = self.prepare_dataset(eval_data)
            eval_sampler = SequentialSampler(eval_dataset)
            # predict_batch_size = args.per_device_eval_batch_size * max(1, args.n_gpu)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate_fn)
        else:
            eval_dataloader = None

        if args.max_steps > 0:
            num_training_steps = args.max_steps
            args.num_train_epochs = args.max_steps // (len(dataloader) // args.gradient_accumulation_steps) + 1
        else:
            num_training_steps = len(dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        args.warmup_steps = int(num_training_steps * args.warmup_ratio)
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps)
        # multi-gpu training
        if args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                    args.train_batch_size * args.gradient_accumulation_steps)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", num_training_steps)

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()
        set_seed(args.seed)  # Added here for reproductibility (even between python 2 and 3)
        print('total epochs : {}'.format(args.num_train_epochs))
        print('train_dataloader length : {}'.format(len(dataloader)))
        for _ in range(int(args.num_train_epochs)):
            pbar = ProgressBar(n_total=len(dataloader), desc='Training')
            self.model.train()
            for step, batch in enumerate(dataloader):
                batch = tuple(t.to(args.device) for t in batch)
                label = torch.where(batch[3] > 0, batch[3], torch.full_like(batch[3], -100))
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'labels': label}
                outputs = self.model(**inputs)
                loss = outputs[0]
                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1
                pbar(step, {'loss': loss.item()})
            print(" ")
            if 'cuda' in str(args.device):
                torch.cuda.empty_cache()
            if eval_dataloader:
                score = self.evaluate(eval_dataloader)
                early_stopping(score, self.model, args.output_dir)

    def evaluate(self, dataloader):
        pbar = ProgressBar(n_total=len(dataloader), desc="Predicting")
        self.model.eval()
        Y_true, Y_pred = [], []
        for step, batch in enumerate(dataloader):
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2]}
                y_true = batch[3]
                outputs = self.model(**inputs)
                y_pred = outputs[0][:, 0, 5:7]
                y_pred = y_pred[:, 1] / (y_pred.sum(axis=1) + 1e-8)
                y_true = y_true[:, 0] - 5
                Y_pred.extend(y_pred.cpu().numpy())
                Y_true.extend(y_true.cpu().numpy())
            pbar(step)
        return roc_auc_score(Y_true, Y_pred)

    def load(self, path):
        self.model = BertForMaskedLM.from_pretrained(path)
        self.model.to(device)
