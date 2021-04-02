# coding=utf-8
#In[]
import cv2
import os
import numpy as np

code_patch_dir = '/data2/process_data/caojihua/data/code_patches/'
background_dir = '/data2/Public_Data/COCO/unzip_data/2017/train2017' 

codes = os.listdir(code_patch_dir)
bgs = os.listdir(background_dir)

#In[]
def read_img(code_patch_dir, background_dir, code_name, bg_name):
    code_path = os.path.join(code_patch_dir, code_name)
    code_img = cv2.imread(code_path, cv2.IMREAD_GRAYSCALE)

    bg_path = os.path.join(background_dir, bg_name)
    bg_img = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE)
    return code_img, bg_img

code_img, bg_img = read_img(code_patch_dir, background_dir, codes[1], bgs[0])
code_img = cv2.resize(code_img, (bg_img.shape[1] // 4, bg_img.shape[0] // 4))
print(code_img.shape)
print(bg_img.shape)
import matplotlib.pyplot as plt
plt.imshow(code_img, cmap='gray')
plt.show()
plt.imshow(bg_img, cmap='gray')
plt.show()
#In[]
code_img_bgr = cv2.cvtColor(code_img, cv2.COLOR_GRAY2BGR)
bg_img_bgr = cv2.cvtColor(bg_img, cv2.COLOR_GRAY2BGR)
mask = 255 * np.ones(code_img_bgr.shape, code_img_bgr.dtype)
center = (bg_img_bgr.shape[0] // 2 , bg_img_bgr.shape[1] // 2)
result = cv2.seamlessClone(code_img_bgr, bg_img_bgr, mask, center, cv2.NORMAL_CLONE)
plt.imshow(result)
plt.show()
#In[]
code_img_bgr = cv2.cvtColor(code_img, cv2.COLOR_GRAY2BGR)
bg_img_bgr = cv2.cvtColor(bg_img, cv2.COLOR_GRAY2BGR)
code_img_bgr_fill_to_bg = np.zeros(bg_img_bgr.shape, bg_img_bgr.dtype)
h_begin = code_img_bgr_fill_to_bg.shape[0] // 2 - code_img_bgr.shape[0] // 2
w_begin = code_img_bgr_fill_to_bg.shape[1] // 2 - code_img_bgr.shape[1] // 2
code_img_bgr_fill_to_bg[h_begin: code_img_bgr.shape[0] + h_begin, w_begin: code_img_bgr.shape[1] + w_begin, :] = code_img_bgr
plt.imshow(code_img_bgr_fill_to_bg)
plt.show()
mask = 255 * np.ones(bg_img_bgr.shape, bg_img_bgr.dtype)
mask[h_begin: code_img_bgr.shape[0] + h_begin, w_begin: code_img_bgr.shape[1] + w_begin, :] = 0
mask[0: h_begin - code_img_bgr.shape[0] // 2, 0: w_begin - code_img_bgr.shape[1] // 2, :] = 0
bg_img_bgr[h_begin: code_img_bgr.shape[0] + h_begin, w_begin: code_img_bgr.shape[1] + w_begin, :] = 0
plt.imshow(mask)
plt.show()
center = (bg_img_bgr.shape[0] // 2 , bg_img_bgr.shape[1] // 2)
result = cv2.seamlessClone(bg_img_bgr, code_img_bgr_fill_to_bg, mask, center, cv2.MIXED_CLONE)
print(result.shape)
plt.imshow(result)
plt.show()
#In[]
print(cv2.seamlessClone.__doc__)
#In[]
import numpy as np
from PIL import Image  # Python image Library
from scipy import sparse
from scipy.sparse import linalg


class SeamlessEditingTool:

    def __init__(self, ref, target, mask):

        self.ref = np.array(Image.open(ref))
        self.target = np.array(Image.open(target))
        self.mask = np.array(Image.open(mask))
        self.height, self.width, blank = self.ref.shape
        # (width, height)-tuple
        self.newImage = Image.new('RGB', (self.width, self.height))
        # index of mask
        # map coordinate of pixels to be calculated to index_map according to
        # mask
        self.maskidx2Corrd = []
        # map coordinates of neigbourhoods to mask indices
        self.Coord2indx = -1 * np.ones([self.height, self.width])

        # True  if  q \in N_p \bigcap \Sigma
        # False elsewise
        # at boundary
        self.if_strict_interior = []  # left, right, top, botton
        idx = 0

        for i in range(self.height):
            for j in range(self.width):
                if self.mask[i, j, 0] == 255:
                    self.maskidx2Corrd.append([i, j])
                    self.if_strict_interior.append([
                        i > 0 and self.mask[i - 1, j, 0] == 255,
                        i < self.height - 1 and self.mask[i + 1, j, 0] == 255,
                        j > 0 and self.mask[i, j - 1, 0] == 255,
                        j < self.width - 1 and self.mask[i, j + 1, 0] == 255
                    ])
                    self.Coord2indx[i][j] = idx
                    idx += 1

        # number of mask
        N = idx
        self.b = np.zeros([N, 3])
        self.A = np.zeros([N, N])

    def create_possion_equation(self):

        # Using the finite difference method
        N = self.b.shape[0]
        for i in range(N):
            # for every pixel in interior and boundary
            self.A[i, i] = 4
            x, y = self.maskidx2Corrd[i]
            if self.if_strict_interior[i][0]:
                self.A[i, int(self.Coord2indx[x - 1, y])] = -1
            if self.if_strict_interior[i][1]:
                self.A[i, int(self.Coord2indx[x + 1, y])] = -1
            if self.if_strict_interior[i][2]:
                self.A[i, int(self.Coord2indx[x, y - 1])] = -1
            if self.if_strict_interior[i][3]:
                self.A[i, int(self.Coord2indx[x, y + 1])] = -1

        # Row-based linked list sparse matrix
        # This is an efficient structure for
        # constructing sparse matrices incrementally.
        self.A = sparse.lil_matrix(self.A, dtype=int)

        for i in range(N):
            flag = np.mod(
                np.array(self.if_strict_interior[i], dtype=int) + 1, 2)
            x, y = self.maskidx2Corrd[i]
            for j in range(3):
                self.b[i, j] = 4 * self.ref[x, y, j] - self.ref[x - 1, y, j] - \
                    self.ref[x + 1, y, j] - self.ref[x,
                                                     y - 1, j] - self.ref[x, y + 1, j]
                self.b[i, j] += flag[0] * self.target[x - 1, y, j] + \
                    flag[1] * self.target[x + 1, y, j] + flag[2] * \
                    self.target[x, y - 1, j] + \
                    flag[3] * self.target[x, y + 1, j]

    def possion_solver(self):

        self.create_possion_equation()

        # Use Conjugate Gradient iteration to solve A x = b
        x_r = linalg.cg(self.A, self.b[:, 0])[0]
        x_g = linalg.cg(self.A, self.b[:, 1])[0]
        x_b = linalg.cg(self.A, self.b[:, 2])[0]

        self.newImage = self.target

        for i in range(self.b.shape[0]):
            x, y = self.maskidx2Corrd[i]
            self.newImage[x, y, 0] = np.clip(x_r[i], 0, 255)
            self.newImage[x, y, 1] = np.clip(x_g[i], 0, 255)
            self.newImage[x, y, 2] = np.clip(x_b[i], 0, 255)

        self.newImage = Image.fromarray(self.newImage)
        return self.newImage


if __name__ == "__main__":

    test = 0
    if test == 1:
        ref = "mona-source.jpg"
        target = "mona-target.jpg"
        mask = "mona-mask.jpg"
        tools = SeamlessEditingTool(ref, target, mask)
        newImage = tools.possion_solver()
        newImage.save('mona-leber-final.jpg')
    else:
        ref = "sealion-source.jpg"
        target = "duck-target.jpg"
        mask = "duck-mask.jpg"
        tools = SeamlessEditingTool(ref, target, mask)
        newImage = tools.possion_solver()
        newImage.save('duck-sealion-final.jpg')