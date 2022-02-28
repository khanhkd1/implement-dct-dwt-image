from PIL import Image
import numpy as np
import pywt
import sys


def create_watermark(image_path, n):
    # Bước 1: Đọc ảnh, chuyển định dạng sang RGB và tính ma trận trung bình R, G, B của mỗi pixel
    img = Image.open(image_path).convert('RGB')
    img_arr = np.array(img)
    n_h, n_w, h_c = img_arr.shape
    if n_h != n_w:
        sys.exit()
    mean_array = np.zeros((n_h, n_w))
    for i in range(n_h):
        for j in range(n_w):
            mean_array[i, j] = np.mean(img_arr[i, j, :])

    # Bước 2: Áp dụng DWT trích xuất ma trận LL
    LL, (LH, HL, HH) = pywt.dwt2(mean_array, 'haar')

    # Bước 3: Phân ma trận LL thành các ma trận 2*2 không chồng nhau
    matrix_2_2 = []
    for i in range(0, LL.shape[0], 2):
        for j in range(0, LL.shape[0], 2):
            matrix_2_2.append(LL[i:i + 2, j:j + 2])

    # Bước 4: Xây dựng ma trận Mb(p, q) trong đó p = {1, 2, ..., N/4} và q = {1, 2, ..., N/4}
    Mb_array = []
    for i in range(len(matrix_2_2)):
        Mb_array.append(np.min(matrix_2_2[i]))
    Mb_array = np.array(Mb_array).reshape((int(LL.shape[0] / 2), int(LL.shape[0] / 2)))

    # Bước 5: Thực hiện Biến đổi Arnold n lần dựa trên giá trị khóa Mb và thu được ma trận Ms
    Ms_array = np.copy(Mb_array)
    N = Ms_array.shape[0]
    x, y = np.meshgrid(range(N), range(N))
    x_map = (x + y) % N
    y_map = (x + 2*y) % N

    for i in range(n):
        Ms_array = Ms_array[x_map, y_map]

    # Bước 6: Tạo mẫu watermark để nhúng vào hình ảnh gốc
    W_array = np.copy(Ms_array)
    for i in range(N):
        for j in range(N):
            W_array[i, j] = 0 if int(np.around(Ms_array[i, j]))%2==0 else 1

    return W_array


def embed_watermark(image_path):
    # Bước 1: Đọc ảnh rồi chuyển sang không gian màu YCbCr
    img = Image.open(image_path).convert('RGB').convert('YCbCr')
    # print(np.asarray(img).shape)
    # np.set_printoptions(threshold=np.inf)
    # LL, (LH, HL, HH) = pywt.dwt2(img, 'haar')
    coeffs = pywt.dwtn(img, 'haar')
    # print(LL.shape)
    print((coeffs.keys()))


if __name__ == '__main__':
    embed_watermark('camera.png')
    # print(np.around(1.4))
