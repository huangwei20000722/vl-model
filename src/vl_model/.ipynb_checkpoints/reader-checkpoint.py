# import base64

# from dotenv import find_dotenv, load_dotenv

# from vl_model.client import get_client

# load_dotenv(find_dotenv())



# class MeterImageReader(object):
#   """读取图片中的仪表读数，使用的文本的形式返回。"""

#   def __init__(self, image_path):
#     self.image_path = image_path
#     self.client = get_client()

#   def read_image(self):
#     img_type = self.image_path.split(".")[-1]
#     with open(self.image_path, "rb") as image_file:
#       image_data = image_file.read()
#       base64_image = base64.b64encode(image_data).decode("utf-8")

#     response = self.client.chat.completions.create(
#       model="Qwen/Qwen2.5-VL-72B-Instruct",
#       messages=[
#         {
#           "role": "system",
#           "content": "You are a helpful assistant that can read images and extract the meter value.",
#         },
#         {
#           "role": "user",
#           "content": [
#             {
#               "type": "image_url",
#               "image_url": {"url": f"data:image/{img_type};base64,{base64_image}"},
#             }
#           ],
#         },
#       ],
#     )
#     return response.choices[0].message.content


# def test_last_one():
#   image_path = "resources/gas-crop.png"
#   image_reader = MeterImageReader(image_path)
#   print(image_reader.read_image())


# def test_all():
#   image_list = [
#     "resources/image.jpg",
#     "resources/crop.png",
#     "resources/meter-2.jpg",
#     "resources/gas.jpg",
#   ]
#   for image_path in image_list:
#     image_reader = MeterImageReader(image_path)
#     print(image_reader.read_image())


# if __name__ == "__main__":
#   # test_all()
#   test_last_one()





import base64
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import torch
import os
from dotenv import find_dotenv, load_dotenv
import gc
import tempfile  # 添加临时文件处理

load_dotenv(find_dotenv())

# 全局加载模型（单例模式）
MODEL = None
TOKENIZER = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model_once():
    global MODEL, TOKENIZER
    if MODEL is None or TOKENIZER is None:
        TOKENIZER = AutoTokenizer.from_pretrained(
            'stepfun-ai/GOT-OCR2_0', 
            trust_remote_code=True
        )
        MODEL = AutoModel.from_pretrained(
            'stepfun-ai/GOT-OCR2_0',
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map='auto',
            use_safetensors=True,
            torch_dtype=torch.float16,  # 使用半精度减少显存
            pad_token_id=TOKENIZER.eos_token_id
        ).eval().to(DEVICE)

class MeterImageReader:
    def __init__(self, image_path):
        self.image_path = image_path
        load_model_once()  # 确保模型只加载一次

    def read_image(self):
        try:
            image = Image.open(self.image_path)
        except Exception as e:
            print(f"打开图像文件时出错: {e}")
            return None
        
        # 创建临时文件（解决模型需要文件路径的问题）
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=True) as temp_file:
            image.save(temp_file.name, format='JPEG')
            
            # 使用临时文件路径执行 OCR
            with torch.inference_mode():  # 禁用梯度计算
                res = MODEL.chat(TOKENIZER, temp_file.name, ocr_type='ocr')
        
        # 显存清理
        torch.cuda.empty_cache()
        gc.collect()
        
        return res

# 测试函数保持不变
def test_last_one():
    image_path = "resources/gas-crop.png"
    image_reader = MeterImageReader(image_path)
    print(image_reader.read_image())

def test_all():
    image_list = [
        "resources/image.jpg",
        "resources/crop.png",
        "resources/meter-2.jpg",
        "resources/gas.jpg",
    ]
    for image_path in image_list:
        print(f"\n处理图像: {image_path}")
        image_reader = MeterImageReader(image_path)
        result = image_reader.read_image()
        print(f"识别结果: {result}")
        
        # 显式释放资源
        del image_reader
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    # 设置内存分配优化
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    # 选择测试模式
    test_all()
    # test_last_one()

