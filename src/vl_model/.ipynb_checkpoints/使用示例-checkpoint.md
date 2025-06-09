## 使用示例
**基本用法**
python3 reader.py 图像路径 [--model got|openai]

**使用GOT-ocr测试单个图像（默认）**
python3 reader.py resources/gas.jpg --model got

**使用openai测试单个图像**
python3 reader.py resources/gas-crop.png --model openai

**使用GOT-ocr测试所有图像**
python3 reader.py dummy --test_all --model got

**使用openai测试所有图像**
python3 reader.py dummy --test_all --model openai