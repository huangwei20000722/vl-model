# 基于vl-model用GOT-OCR或者openai api识别resources图像

import base64
import os
import argparse
import torch
import gc
import tempfile
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from dotenv import find_dotenv, load_dotenv

# 加载环境变量
load_dotenv(find_dotenv())

# 全局模型变量
GOT_MODEL = None
GOT_TOKENIZER = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_got_model():
    """加载GOT-OCR模型（单例模式）"""
    global GOT_MODEL, GOT_TOKENIZER
    if GOT_MODEL is None or GOT_TOKENIZER is None:
        print("正在加载GOT-OCR模型...")
        GOT_TOKENIZER = AutoTokenizer.from_pretrained(
            'stepfun-ai/GOT-OCR2_0', 
            trust_remote_code=True
        )
        GOT_MODEL = AutoModel.from_pretrained(
            'stepfun-ai/GOT-OCR2_0',
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map='auto',
            use_safetensors=True,
            torch_dtype=torch.float16,
            pad_token_id=GOT_TOKENIZER.eos_token_id
        ).eval().to(DEVICE)
        print("GOT-OCR模型加载完成")

class GOTMeterImageReader:
    """使用GOT-OCR模型读取仪表图像"""
    def __init__(self, image_path):
        self.image_path = image_path
        load_got_model()  # 确保模型已加载

    def read_image(self):
        try:
            image = Image.open(self.image_path)
        except Exception as e:
            print(f"打开图像文件时出错: {e}")
            return None
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=True) as temp_file:
            image.save(temp_file.name, format='JPEG')
            
            # 使用临时文件路径执行OCR
            with torch.inference_mode():
                res = GOT_MODEL.chat(GOT_TOKENIZER, temp_file.name, ocr_type='ocr')
        
        # 清理资源
        torch.cuda.empty_cache()
        gc.collect()
        
        return res

class OpenAIMeterImageReader:
    """使用OpenAI API读取仪表图像"""
    def __init__(self, image_path):
        self.image_path = image_path
        from vl_model.client import get_client  # 延迟导入
        self.client = get_client()

    def read_image(self):
        img_type = self.image_path.split(".")[-1]
        with open(self.image_path, "rb") as image_file:
            image_data = image_file.read()
            base64_image = base64.b64encode(image_data).decode("utf-8")

        response = self.client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-72B-Instruct",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that can read images and extract the meter value.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{img_type};base64,{base64_image}"},
                        },
                    ],
                },
            ],
        )
        return response.choices[0].message.content

def main():
    """主命令行程序"""
    parser = argparse.ArgumentParser(description='仪表图像读数识别工具')
    parser.add_argument('image_path', type=str, help='要识别的图像路径')
    parser.add_argument('--model', type=str, choices=['got', 'openai'], default='got',
                        help='选择使用的模型: got (本地GOT-OCR) 或 openai (OpenAI API)')
    parser.add_argument('--test_all', action='store_true', help='测试所有示例图像')
    
    args = parser.parse_args()
    
    # 设置显存优化
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    if args.test_all:
        test_images = [
            "resources/image.jpg",
            "resources/crop.png",
            "resources/meter-2.jpg",
            "resources/gas.jpg",
        ]
        for img_path in test_images:
            print(f"\n处理图像: {img_path}")
            if not os.path.exists(img_path):
                print(f"文件不存在: {img_path}")
                continue
            
            if args.model == 'got':
                reader = GOTMeterImageReader(img_path)
            else:
                reader = OpenAIMeterImageReader(img_path)
            
            result = reader.read_image()
            print(f"识别结果: {result}")
            
            # 清理资源
            if args.model == 'got':
                torch.cuda.empty_cache()
                gc.collect()
    else:
        if not os.path.exists(args.image_path):
            print(f"错误: 文件不存在 - {args.image_path}")
            return
        
        if args.model == 'got':
            reader = GOTMeterImageReader(args.image_path)
        else:
            reader = OpenAIMeterImageReader(args.image_path)
        
        result = reader.read_image()
        print(f"\n图像: {args.image_path}")
        print(f"模型: {'GOT-OCR' if args.model == 'got' else 'OpenAI'}")
        print(f"识别结果: {result}")

if __name__ == "__main__":
    main()