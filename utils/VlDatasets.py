from transformers import AutoTokenizer, AutoProcessor
from datasets import Dataset
from PIL import Image
import pandas as pd
import torch

"""
注意，这里可以不使用processer，可以自定义加载图像信息
示例如下：
from torchvision import transforms

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),  # 一般是这个，但最好查官方
])

image_tensor = image_transforms(image_input).unsqueeze(0)  # 加上 batch 维度
inputs = {
    "input_ids": tokenized_text["input_ids"],
    "attention_mask": tokenized_text["attention_mask"],
    "pixel_values": image_tensor
}

"""


class VlDataset(Dataset):
    """
    自定义数据集，最终直接返回inputs_ids、attention_mask和labels
    这样做就不需要再自定义Collator,直接使用DataCollatorForSeq2Seq
    """

    def __init__(self, data_path, tokenizer=None):
        self.csv_data = pd.read_csv(data_path)
        self.train_length = self.csv_data.shape[0] - 50
        self.max_length = 8192

    def __len__(self):
        return self.train_length

    def __getitem__(self, index):
        signal_data = self.csv_data.iloc[index]

        image_path = signal_data[0]
        caption = signal_data[1]
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"{image_path}",
                        "resized_height": 280,
                        "resized_width": 280,
                    },
                    {"type": "text", "text": "COCO Yes:"},
                ],
            },
            {
                "role": "assistant",
                "content": f"{caption}"
            }
        ]
        all_inputs_message = messages
        prompt_inputs_message = messages[:-1]

        all_inputs_text_ids = tokenizers.apply_chat_template(all_inputs_message, tokenize=False,
                                                             add_generation_prompt=True,
                                                             return_tensors="pt")

        prompts_ids = tokenizers.apply_chat_template(prompt_inputs_message, tokenize=True, add_generation_prompt=True,
                                                     return_tensors="pt").tolist()
        image_input = Image.open(image_path)

        inputs = processor(
            text=all_inputs_text_ids,
            images=image_input,
            videos=None,
            padding=True,
            return_tensors="pt",
        )

        inputs = {key: value.tolist() for key, value in inputs.items()}  # 把tensor转成list
        final_all_input_text_ids = inputs["input_ids"][0] + [tokenizers.pad_token_id]
        final_all_attention_mask = inputs["attention_mask"][0] + [1]  # 这里为什么要加一 是因为要和final_all_input_text_ids对齐

        final_all_lables = [-100] * (len(prompts_ids[0])) + inputs["input_ids"][0][len(prompts_ids[0]):] + [
            tokenizers.pad_token_id]

        assert len(final_all_input_text_ids) == len(final_all_attention_mask) == len(
            final_all_lables), "input_text_ids,attention_mask,labels,三者长度必须统一"

        if len(final_all_input_text_ids) > self.max_length:  # 做一个截断
            final_all_input_text_ids = final_all_input_text_ids[:self.max_length]
            final_all_attention_mask = final_all_attention_mask[:self.max_length]
            final_all_lables = final_all_lables[:self.max_length]

        input_ids = torch.tensor(final_all_input_text_ids)
        attention_mask = torch.tensor(final_all_attention_mask)
        labels = torch.tensor(final_all_lables)
        inputs['pixel_values'] = torch.tensor(inputs['pixel_values'])
        inputs['image_grid_thw'] = torch.tensor(inputs['image_grid_thw']).squeeze(0)  # 由（1,h,w)变换为（h,w）
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
                "pixel_values": inputs['pixel_values'], "image_grid_thw": inputs['image_grid_thw']}


if __name__ == "__main__":
    path = "../data_datasets/coco_2014/coco-2024-dataset.csv"
    model_path = "D:\code\model_path_chatglm3-6b\Qwen2.5-VL-7B-Instruct"
    tokenizers = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    vl_dataset = VlDataset(path, tokenizers)
    data = vl_dataset.__getitem__(0)
    print(data)
