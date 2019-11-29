import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

M, N = 8, 8


def DCT(input):

    # declare temporary output array
    tmp_output = np.zeros((8, 8))

    # declare C(u), C(v)
    Cu, Cv = 0.0, 0.0

    for u in range(M):
        for v in range(N):

            # according from the formula of DCT
            if u == 0:
                Cu = 1 / np.sqrt(M)
            else:
                Cu = np.sqrt(2) / np.sqrt(M)
            if v == 0:
                Cv = 1 / np.sqrt(N)
            else:
                Cv = np.sqrt(2) / np.sqrt(N)

            # calculate DCT
            tmp_sum = 0
            for x in range(M):
                for y in range(N):
                    dct = input[x][y] * math.cos((2 * x + 1) * u * math.pi / (
                        2 * M)) * math.cos((2 * y + 1) * v * math.pi / (2 * N))
                    tmp_sum += dct

            tmp_output[u][v] = Cu * Cv * tmp_sum

    return tmp_output


def IDCT(input):

    # declare temporary output array
    tmp_output = np.zeros((8, 8))

    # declare C(u), C(v)
    Cu, Cv = 0.0, 0.0

    for x in range(M):
        for y in range(N):

            # calculate IDCT
            tmp_sum = 0
            for u in range(M):
                for v in range(N):

                    # according from the formula of IDCT
                    if u == 0:
                        Cu = 1 / np.sqrt(M)
                    else:
                        Cu = np.sqrt(2) / np.sqrt(M)
                    if v == 0:
                        Cv = 1 / np.sqrt(N)
                    else:
                        Cv = np.sqrt(2) / np.sqrt(N)

                    idct = input[u][v] * math.cos((2 * x + 1) * u * math.pi / (
                        2 * M)) * math.cos((2 * y + 1) * v * math.pi / (2 * N))
                    tmp_sum += Cu * Cv * idct

            tmp_output[x][y] = tmp_sum

    return tmp_output


def PSNR(dct, idct):

    # declare array
    error_lena = np.zeros((512, 512))

    # calculate error
    for x in range(512):
        for y in range(512):
            error_lena[x][y] = dct[x, y] - idct[x, y]

    # calculate MSE
    mse = np.mean((dct - idct) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.

    # return error image and PSNR value
    return error_lena, 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def main():

    # read image
    lena = cv2.imread('InputData/lena.png', 0)

    # declare arrarys with full padding zeros
    dct_transform_lena = np.zeros((512, 512))
    quantized_lena = np.zeros((512, 512))
    inverse_quantized_lena = np.zeros((512, 512))
    idct_transform_lena = np.zeros((512, 512))

    # declare quantization block
    block = [[16, 11, 10, 16, 24, 40, 51, 61],
             [12, 12, 14, 19, 26, 58, 60, 55],
             [14, 13, 16, 24, 40, 57, 69, 56],
             [14, 17, 22, 29, 51, 87, 80, 62],
             [18, 22, 37, 56, 68, 109, 103, 77],
             [24, 35, 55, 64, 81, 104, 113, 92],
             [49, 64, 78, 87, 103, 121, 120, 101],
             [72, 92, 95, 98, 112, 100, 103, 99]]

    # cut lena into 8x8 and send into DCT() to calculate
    for x in range(0, 512, 8):
        for y in range(0, 512, 8):
            cut_lena = lena[x: x + 8, y: y + 8]
            dct_lena = DCT(cut_lena)
            dct_transform_lena[x: x + 8, y: y + 8] = np.copy(dct_lena)

    # use quantization block to quantize lena
    for x in range(0, 512, 8):
        for y in range(0, 512, 8):
            quantized_lena[x: x + 8, y: y +
                           8] = np.divide(dct_transform_lena[x: x + 8, y: y + 8], block)

    # inverse the quantization
    for x in range(0, 512, 8):
        for y in range(0, 512, 8):
            inverse_quantized_lena[x: x + 8, y: y +
                                   8] = np.multiply(quantized_lena[x: x + 8, y: y + 8], block)

    # cut inverse_quantized_lena into 8x8 and send into IDCT() to calculate
    for x in range(0, 512, 8):
        for y in range(0, 512, 8):
            cut_lena = inverse_quantized_lena[x: x + 8, y: y + 8]
            idct_lena = IDCT(cut_lena)
            idct_transform_lena[x: x + 8, y: y + 8] = np.copy(idct_lena)

    # normalize the lena after DCT transform
    normalized_dct_lena = (dct_transform_lena-dct_transform_lena.min()) / \
        (dct_transform_lena.max() - dct_transform_lena.min()) * 255

    # normalize the lena after IDCT transform
    normalized_idct_lena = (idct_transform_lena - idct_transform_lena.min()) / \
        (idct_transform_lena.max() - idct_transform_lena.min()) * 255

    # calculate the PSNR and also error image
    error_lena, psnr = PSNR(normalized_dct_lena, normalized_idct_lena)

    # write images as png files
    cv2.imwrite('OutputData/dct_lena.png', dct_transform_lena)
    cv2.imwrite('OutputData/quantized_lena.png', quantized_lena)
    cv2.imwrite('OutputData/inverse_quantized_lena.png',
                inverse_quantized_lena)
    cv2.imwrite('OutputData/idct_lena.png', idct_transform_lena)
    cv2.imwrite('OutputData/error_lena.png', error_lena)

    # read the png images and show them
    dct_lena = cv2.imread('OutputData/dct_lena.png', 0)
    idct_lena = cv2.imread('OutputData/idct_lena.png', 0)
    error_lena = cv2.imread('OutputData/error_lena.png', 0)
    cv2.imshow('Original Image', lena)
    cv2.waitKey()
    cv2.imshow('DCT Transform', dct_lena)
    cv2.waitKey()
    cv2.imshow('After IDCT', idct_lena)
    cv2.waitKey()
    cv2.imshow('Error Image', error_lena)
    cv2.waitKey()

    # save PSNR result as txt
    f = open('OutputData/psnr.txt', 'w')
    f.write(str(psnr))
    f.close()
    print(psnr)


if __name__ == "__main__":
    main()
