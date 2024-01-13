from PIL import Image

def generate_checkerboard(size, square_size):
    image = Image.new("L", (size, size), color=255)  # 创建一个白色背景的图像

    for i in range(0, size, square_size):
        for j in range(0, size, square_size):
            if (i // square_size + j // square_size) % 2 == 1:
                # 如果当前方块格是奇数行奇数列或偶数行偶数列，则将其填充为黑色
                image.paste(0, (j, i, j + square_size, i + square_size))

    return image

size = 10000
square_size = 100

checkerboard_image = generate_checkerboard(size, square_size)
checkerboard_image.show()  # 显示生成的图像
checkerboard_image.save("checkerboard.png")  # 保存生成的图像
