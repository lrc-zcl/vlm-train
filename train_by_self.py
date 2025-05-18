from modelscope import AutoTokenizer
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

from load_model.models import PeftModel
from utils.VlDatasets import VlDataset

model_path = r"D:\code\model_path_chatglm3-6b\Qwen2.5-VL-7B-Instruct"
dataset_path = "./data_datasets/coco_2014/coco-2024-dataset.csv"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

train_dataset = VlDataset(dataset_path, tokenizer=tokenizer)
peft_model = PeftModel(model_path).get_peft_model()["peft_model"]
args = TrainingArguments(
    output_dir="./output/train_qwen25vl",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    logging_steps=10,
    num_train_epochs=2,
    save_steps=1,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)
trainer = Trainer(
    model=peft_model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
trainer.train()
