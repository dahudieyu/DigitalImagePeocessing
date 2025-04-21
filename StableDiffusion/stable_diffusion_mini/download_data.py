import os
import requests
import zipfile

def download_mini_data():
    url = "https://huggingface.co/datasets/lmz/catdog-mini-stable-diffusion/resolve/main/mini_diffusion_data.zip"
    save_path = "mini_diffusion_data.zip"

    if not os.path.exists("data"):
        print("📥 下载小型图像数据中...")
        r = requests.get(url)
        with open(save_path, "wb") as f:
            f.write(r.content)
        print("正在尝试打开的文件路径:", save_path)


        with zipfile.ZipFile(save_path, 'r') as zip_ref:
            zip_ref.extractall("data")

        os.remove(save_path)
        print("✅ 下载完成并解压到 data/ 目录")
    else:
        print("✅ 数据目录已存在，跳过下载")

download_mini_data()
