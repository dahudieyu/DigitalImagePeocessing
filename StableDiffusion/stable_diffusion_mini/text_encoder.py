import sys
sys.path.append('/Users/dahudieyu/Documents/Education/MyPython/DigitalImagePeocessing/StableDiffusion/stable_diffusion_mini')

# 进行环境变量设置  镜像源 
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import CLIPTokenizer, CLIPTextModel
import torch

# 如果想在texts里面输入中文 model_name = "OFA-Sys/chinese-clip-vit-base-patch16"
class TextEncoder(torch.nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name)

    def forward(self, texts):
        """
        输入: texts = ["a cat on a bed", "a dog in the grass"]
        输出: text embeddings, shape = (batch_size, seq_len, dim)
        """
        inputs = self.tokenizer(texts, padding="max_length", max_length=77, return_tensors="pt", truncation=True)
        outputs = self.text_encoder(**inputs)

        text_embedding = outputs.last_hidden_state  # [batch_size, 77, 512]
        
        # 选择第一个 token (CLS token) 作为文本表示
        text_embedding = text_embedding[:, 0, :]  # [batch_size, text_embed_dim]，形状变为 [1, 512]
        return text_embedding
       # return outputs.last_hidden_state  # shape: [batch_size, 77, 768]
if __name__ == '__main__':
    texts = ["a cat on a bed"]
    text_encoder = TextEncoder()
    text_embeddings = text_encoder(texts)
    print(text_embeddings.shape)
    print(text_embeddings)

# ----------------------------------------------------------------------
#  return outputs.last_hidden_state 
# 运行结果：torch.Size([1, 77, 512])
	# •	1：表示你输入了两个句子（batch_size = 1）
	# •	77：是 CLIP 模型固定的最大输入长度
	# •	512：是 openai/clip-vit-base-patch32 模型的 hidden size（特征维度）