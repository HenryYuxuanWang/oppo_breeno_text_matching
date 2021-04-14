import os
import logging

import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from transformers import BertPreTrainedModel, BertModel, BertConfig, AdamW, set_seed, TrainingArguments
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, MaskedLMOutput
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset, TensorDataset
from torch.nn import CrossEntropyLoss
from torch import nn

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


class TaskDataset(Dataset):
    def __init__(self, data, tokens, random: bool = True, max_len: int = 128):
        self.data = data
        self.tokens = tokens
        self.max_len = max_len
        self.random = random

    def __getitem__(self, index):
        text1, text2, label = self.data[index]
        token_ids, segment_ids, output_ids = self.sample_convert(text1, text2, label, self.random)
        attention_mask = [1] * len(token_ids)
        input_len = len(token_ids)

        token_ids = self.sequence_padding([token_ids], length=self.max_len)[0]
        attention_mask = self.sequence_padding([attention_mask], length=self.max_len)[0]
        segment_ids = self.sequence_padding([segment_ids], length=self.max_len)[0]
        label = self.sequence_padding([output_ids], length=self.max_len)[0]

        token_ids = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        input_len = torch.tensor(input_len, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)
        return [token_ids, attention_mask, segment_ids, input_len, label]

    def __len__(self):
        return len(self.data)

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


class BertForMaskedLM(BertPreTrainedModel):
    def __init__(self, path, config, keep_tokens=None, embedding_dim=768):
        super().__init__(config)
        self.bert = BertModel.from_pretrained(path)
        self.config = self.bert.config
        if keep_tokens:
            embedding_size = len(keep_tokens)
            keep_word_embeddings = nn.Embedding(embedding_size, embedding_dim)
            weight = self.bert.embeddings.word_embeddings(torch.tensor(keep_tokens, device=self.bert.device))
            keep_word_embeddings.weight = nn.Parameter(weight)
            self.bert.embeddings.word_embeddings = keep_word_embeddings
            self.config.vocab_size = len(keep_tokens)
        self.cls = BertOnlyMLMHead(self.config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Model:
    def __init__(self, path='/data/pretrained_models/torch/bert/bert-base-chinese'):
        self.model_dir = path
        self.model = None
        self.keep_tokens = None
        self.checkpoint = None

    @staticmethod
    def prepare_dataset(data):
        all_input_ids = torch.tensor([f.input_ids for f in data], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in data], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in data], dtype=torch.long)
        all_lens = torch.tensor([f.input_len for f in data], dtype=torch.long)
        label = torch.tensor([f.label for f in data], dtype=torch.long)
        inputs = [all_input_ids, all_attention_mask, all_token_type_ids, all_lens, label]
        dataset = TensorDataset(*inputs)
        return dataset

    def train(self, train_data, tokens, keep_tokens, eval_data=None, args=None, checkpoint=None):
        self.checkpoint = checkpoint
        self.keep_tokens = keep_tokens
        if args is None:
            output_dir = "tmp_trainer"
            logger.info(f"No `TrainingArguments` passed, using `output_dir={output_dir}`.")
            args = TrainingArguments(output_dir=output_dir)
        self.load(self.model_dir)
        early_stopping = EarlyStopping(verbose=True)
        max_len_train = max([len(x[0]) + len(x[1]) + 3 for x in train_data])
        max_len_eval = max([len(x[0]) + len(x[1]) + 3 for x in eval_data])
        dataset = TaskDataset(train_data, tokens, random=True, max_len=max_len_train)
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.train_batch_size, collate_fn=collate_fn)
        if eval_data:
            eval_dataset = TaskDataset(eval_data, tokens, random=False, max_len=max_len_eval)
            eval_sampler = SequentialSampler(eval_dataset)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate_fn)
        else:
            eval_dataloader = None

        if args.max_steps > 0:
            num_training_steps = args.max_steps
            args.num_train_epochs = args.max_steps // (len(dataloader) // args.gradient_accumulation_steps) + 1
        else:
            num_training_steps = len(dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        # multi-gpu training
        if args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                    args.train_batch_size * args.gradient_accumulation_steps)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", num_training_steps)

        global_step = 0
        # tr_loss = 0.0
        optimizer = AdamW(self.model.parameters(), lr=args.learning_rate)
        self.model.zero_grad()
        set_seed(args.seed)  # Added here for reproductibility (even between python 2 and 3)
        print('total epochs : {}'.format(args.num_train_epochs))
        print('train_dataloader length : {}'.format(len(dataloader)))
        for epoch in range(int(args.num_train_epochs)):
            print('Epoch: %d' % epoch)
            pbar = ProgressBar(n_total=len(dataloader), desc='Training')
            losses = []
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
                losses.append(loss.cpu().detach().numpy())
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                # tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                pbar(step, {'loss': np.mean(losses)})
            if 'cuda' in str(args.device):
                torch.cuda.empty_cache()
            if eval_dataloader:
                score = self.evaluate(eval_dataloader)
                early_stopping(score, self.model, args.output_dir)

    def evaluate(self, dataloader):
        pbar = ProgressBar(n_total=len(dataloader), desc='Evaluating')
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
        config = BertConfig(path)
        self.model = BertForMaskedLM(path, config, keep_tokens=self.keep_tokens)
        if self.checkpoint:
            self.model.load_state_dict(torch.load(os.path.join(self.checkpoint, 'pytorch_model.bin'), map_location=device))
        self.model.to(device)
