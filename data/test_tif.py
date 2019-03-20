#coding:utf-8

import PIL.Image as Image

def test():
    img_file = "/home/chen/dataset/kaggle/cancer-detection/0a0a85db9218e366569c913185cc0740f59f4d9e.tif"
    img = Image.open(img_file)
    print(img.size)
    print(img)


if __name__ == "__main__":
    test()