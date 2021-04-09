from transformers import TrainingArguments

args = TrainingArguments(
    output_dir='/data/oppo_breeno/results',
    num_train_epochs=600,
    per_gpu_train_batch_size=32,
    per_device_eval_batch_size=64,
    max_steps=0,
    learning_rate=1e-6,
    gradient_accumulation_steps=1,
    warmup_ratio=0.1,
    weight_decay=0.01,
    adam_epsilon=1e-8,
    seed=42,
    max_grad_norm=1.0)
