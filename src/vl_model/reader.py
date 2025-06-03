import base64

from dotenv import find_dotenv, load_dotenv

from vl_model.client import get_client

load_dotenv(find_dotenv())


class MeterImageReader(object):
  """读取图片中的仪表读数，使用的文本的形式返回。"""

  def __init__(self, image_path):
    self.image_path = image_path
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
            }
          ],
        },
      ],
    )
    return response.choices[0].message.content


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
    image_reader = MeterImageReader(image_path)
    print(image_reader.read_image())


if __name__ == "__main__":
  # test_all()
  test_last_one()
