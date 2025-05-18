from transformers import AutoTokenizer

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "this is image path",
                "resized_height": 280,
                "resized_width": 280,
            },
            {"type": "text", "text": "COCO Yes:"},
        ],
    }
]

model_path = "D:\code\model_path_chatglm3-6b\Qwen2.5-VL-7B-Instruct"
tokenizers = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
template = tokenizers.apply_chat_template(messages)
print(tokenizers.decode(template))