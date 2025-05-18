import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from peft import get_peft_model, LoraConfig
from loguru import logger


class PeftModel():
    def __init__(self, model_path):
        self.model_path = model_path
        self.load_init_model()  # 加载模型

    def load_init_model(self):
        self.init_model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True).half()

    def make_inputs_require_grad(self):
        """
        为模型的输入嵌入层启用梯度计算，从而允许在微调过程中更新输入嵌入层的权重。
        :return:
        """
        if hasattr(self.init_model, "enable_input_require_grads"):
            self.init_model.enable_input_require_grads()  # 如果有该属性，直接启动
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            self.init_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    def find_all_linears(self):
        """
        找到模型中的全部全连接层
        :return: list
        """
        # assert self.train_args.bf16 or self.train_args.fp16, "模型数据类型须是bf16或者fp16"
        lora_module_names = set()
        for name, module in self.init_model.named_modules():
            if isinstance(module, torch.nn.Linear):
                name = name.split('.')
                lora_module_names.add(name[0] if len(name) == 1 else name[-1])  # 将所有的全连接层名称保存下来

        if "lm_head" in lora_module_names:
            lora_module_names.remove("lm_head")
        logger.info(f"lora_module_names is {lora_module_names}")
        return list(lora_module_names)

    def get_peft_model(self):
        self.config = LoraConfig(
            target_modules=self.find_all_linears(),
            inference_mode=False,  # 训练模式
            r=2,  # Lora 秩
            lora_alpha=2,  # Lora alaph，具体作用参见 Lora 原理
            lora_dropout=0.05,  # Dropout 比例
            bias="none",
        )
        peft_model = get_peft_model(self.init_model, self.config)
        logger.info(f"Lora微调的模型参数量是{sum(p.numel() for p in peft_model.parameters())}")
        logger.info(f"Lora微调的模型可训练参数量是{sum(p.numel() for p in peft_model.parameters() if p.requires_grad)}")
        return {
            "peft_model": peft_model,
            "ref_model": None,
            "peft_config": self.config
        }
