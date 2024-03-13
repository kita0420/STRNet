import numpy as np
import os
import cv2
import random
import imageio
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# binary image thinning (skeletonization) in-place.
# implements Zhang-Suen algorithm.
# http://agcggs680.pbworks.com/f/Zhan-Suen_algorithm.pdf
# @param im   the binary image
def thinningZS(im):
    prev = np.zeros(im.shape, np.uint8)
    while True:
        im = thinningZSIteration(im, 0)
        im = thinningZSIteration(im, 1)
        diff = np.sum(np.abs(prev - im))
        if not diff:
            break
        prev = im
    return im


# 1 pass of Zhang-Suen thinning
def thinningZSIteration(im, iter):
    marker = np.zeros(im.shape, np.uint8)
    for i in range(1, im.shape[0] - 1):
        for j in range(1, im.shape[1] - 1):
            p2 = im[(i - 1), j]
            p3 = im[(i - 1), j + 1]
            p4 = im[(i), j + 1]
            p5 = im[(i + 1), j + 1]
            p6 = im[(i + 1), j]
            p7 = im[(i + 1), j - 1]
            p8 = im[(i), j - 1]
            p9 = im[(i - 1), j - 1]
            A = (p2 == 0 and p3) + (p3 == 0 and p4) + \
                (p4 == 0 and p5) + (p5 == 0 and p6) + \
                (p6 == 0 and p7) + (p7 == 0 and p8) + \
                (p8 == 0 and p9) + (p9 == 0 and p2)
            B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            m1 = (p2 * p4 * p6) if (iter == 0) else (p2 * p4 * p8)
            m2 = (p4 * p6 * p8) if (iter == 0) else (p2 * p6 * p8)

        if (A == 1 and (B >= 2 and B <= 6) and m1 == 0 and m2 == 0):
            marker[i, j] = 1

    return np.bitwise_and(im, np.bitwise_not(marker))


def thinningSkimage(im):
    from skimage.morphology import skeletonize
    return skeletonize(im).astype(np.uint8)


def thinning(im):
    try:
        return thinningSkimage(im)
    except:
        return thinningZS(im)


# check if a region has any white pixel
def notEmpty(im, x, y, w, h):
    return np.sum(im) > 0


# merge ith fragment of second chunk to first chunk
# @param c0   fragments from first  chunk
# @param c1   fragments from second chunk
# @param i    index of the fragment in first chunk
# @param sx   (x or y) coordinate of the seam
# @param isv  is vertical, not horizontal?
# @param mode 2-bit flag,
#             MSB = is matching the left (not right) end of the fragment from first  chunk
#             LSB = is matching the right (not left) end of the fragment from second chunk
# @return     matching successful?
#
def mergeImpl(c0, c1, i, sx, isv, mode):
    B0 = (mode >> 1 & 1) > 0  # match c0 left
    B1 = (mode >> 0 & 1) > 0  # match c1 left
    mj = -1
    md = 4  # maximum offset to be regarded as continuous

    p1 = c1[i][0 if B1 else -1]

    if (abs(p1[isv] - sx) > 0):  # not on the seam, skip
        return False

    # find the best match
    for j in range(len(c0)):
        p0 = c0[j][0 if B0 else -1]
        if (abs(p0[isv] - sx) > 1):  # not on the seam, skip
            continue

        d = abs(p0[not isv] - p1[not isv])
        if (d < md):
            mj = j
            md = d

    if (mj != -1):  # best match is good enough, merge them
        if (B0 and B1):
            c0[mj] = list(reversed(c1[i])) + c0[mj]
        elif (not B0 and B1):
            c0[mj] += c1[i]
        elif (B0 and not B1):
            c0[mj] = c1[i] + c0[mj]
        else:
            c0[mj] += list(reversed(c1[i]))

        c1.pop(i)
        return True
    return False


HORIZONTAL = 1
VERTICAL = 2


# merge fragments from two chunks
# @param c0   fragments from first  chunk
# @param c1   fragments from second chunk
# @param sx   (x or y) coordinate of the seam
# @param dr   merge direction, HORIZONTAL or VERTICAL?
#
def mergeFrags(c0, c1, sx, dr):
    for i in range(len(c1) - 1, -1, -1):
        if (dr == HORIZONTAL):
            if (mergeImpl(c0, c1, i, sx, False, 1)): continue
            if (mergeImpl(c0, c1, i, sx, False, 3)): continue
            if (mergeImpl(c0, c1, i, sx, False, 0)): continue
            if (mergeImpl(c0, c1, i, sx, False, 2)): continue
        else:
            if (mergeImpl(c0, c1, i, sx, True, 1)): continue
            if (mergeImpl(c0, c1, i, sx, True, 3)): continue
            if (mergeImpl(c0, c1, i, sx, True, 0)): continue
            if (mergeImpl(c0, c1, i, sx, True, 2)): continue

    c0 += c1


# recursive bottom: turn chunk into polyline fragments
# look around on 4 edges of the chunk, and identify the "outgoing" pixels
# add segments connecting these pixels to center of chunk
# apply heuristics to adjust center of chunk
#
# @param im   the bitmap image
# @param x    left of   chunk
# @param y    top of    chunk
# @param w    width of  chunk
# @param h    height of chunk
# @return     the polyline fragments
#
def chunkToFrags(im, x, y, w, h):
    frags = []
    on = False  # to deal with strokes thicker than 1px
    li = -1
    lj = -1

    # walk around the edge clockwise
    for k in range(h + h + w + w - 4):
        i = 0
        j = 0
        if (k < w):
            i = y + 0
            j = x + k
        elif (k < w + h - 1):
            i = y + k - w + 1
            j = x + w - 1
        elif (k < w + h + w - 2):
            i = y + h - 1
            j = x + w - (k - w - h + 3)
        else:
            i = y + h - (k - w - h - w + 4)
            j = x + 0

        if (im[i, j]):  # found an outgoing pixel
            if (not on):  # left side of stroke
                on = True
                frags.append([[j, i], [x + w // 2, y + h // 2]])
        else:
            if (on):  # right side of stroke, average to get center of stroke
                frags[-1][0][0] = (frags[-1][0][0] + lj) // 2
                frags[-1][0][1] = (frags[-1][0][1] + li) // 2
                on = False
        li = i
        lj = j

    if (len(frags) == 2):  # probably just a line, connect them
        f = [frags[0][0], frags[1][0]]
        frags.pop(0)
        frags.pop(0)
        frags.append(f)
    elif (len(frags) > 2):  # it's a crossroad, guess the intersection
        ms = 0
        mi = -1
        mj = -1
        # use convolution to find brightest blob
        for i in range(y + 1, y + h - 1):
            for j in range(x + 1, x + w - 1):
                s = \
                    (im[i - 1, j - 1]) + (im[i - 1, j]) + (im[i - 1, j + 1]) + \
                    (im[i, j - 1]) + (im[i, j]) + (im[i, j + 1]) + \
                    (im[i + 1, j - 1]) + (im[i + 1, j]) + (im[i + 1, j + 1])
                if (s > ms):
                    mi = i
                    mj = j
                    ms = s
                elif (s == ms and abs(j - (x + w // 2)) + abs(i - (y + h // 2)) < abs(mj - (x + w // 2)) + abs(
                        mi - (y + h // 2))):
                    mi = i
                    mj = j
                    ms = s

        if (mi != -1):
            for i in range(len(frags)):
                frags[i][1] = [mj, mi]
    return frags

def traceSkeleton(im, x, y, w, h, csize, maxIter, rects):
    frags = []

    if (maxIter == 0):  # gameover
        return frags
    if (w <= csize and h <= csize):  # recursive bottom
        frags += chunkToFrags(im, x, y, w, h)
        return frags

    ms = im.shape[0] + im.shape[1]  # number of white pixels on the seam, less the better
    mi = -1  # horizontal seam candidate
    mj = -1  # vertical   seam candidate

    if (h > csize):  # try splitting top and bottom
        for i in range(y + 3, y + h - 3):
            if (im[i, x] or im[(i - 1), x] or im[i, x + w - 1] or im[(i - 1), x + w - 1]):
                continue

            s = 0
            for j in range(x, x + w):
                s += im[i, j]
                s += im[(i - 1), j]

            if (s < ms):
                ms = s
                mi = i
            elif (s == ms and abs(i - (y + h // 2)) < abs(mi - (y + h // 2))):
                # if there is a draw (very common), we want the seam to be near the middle
                # to balance the divide and conquer tree
                ms = s
                mi = i

    if (w > csize):  # same as above, try splitting left and right
        for j in range(x + 3, x + w - 2):
            if (im[y, j] or im[(y + h - 1), j] or im[y, j - 1] or im[(y + h - 1), j - 1]):
                continue

            s = 0
            for i in range(y, y + h):
                s += im[i, j]
                s += im[i, j - 1]
            if (s < ms):
                ms = s
                mi = -1  # horizontal seam is defeated
                mj = j
            elif (s == ms and abs(j - (x + w // 2)) < abs(mj - (x + w // 2))):
                ms = s
                mi = -1
                mj = j

    nf = []  # new fragments
    if (h > csize and mi != -1):  # split top and bottom
        L = [x, y, w, mi - y]  # new chunk bounding boxes
        R = [x, mi, w, y + h - mi]

        if (notEmpty(im, L[0], L[1], L[2], L[3])):  # if there are no white pixels, don't waste time
            if (rects != None): rects.append(L)
            nf += traceSkeleton(im, L[0], L[1], L[2], L[3], csize, maxIter - 1, rects)  # recurse

        if (notEmpty(im, R[0], R[1], R[2], R[3])):
            if (rects != None): rects.append(R)
            mergeFrags(nf, traceSkeleton(im, R[0], R[1], R[2], R[3], csize, maxIter - 1, rects), mi, VERTICAL)

    elif (w > csize and mj != -1):  # split left and right
        L = [x, y, mj - x, h]
        R = [mj, y, x + w - mj, h]
        if (notEmpty(im, L[0], L[1], L[2], L[3])):
            if (rects != None): rects.append(L)
            nf += traceSkeleton(im, L[0], L[1], L[2], L[3], csize, maxIter - 1, rects)

        if (notEmpty(im, R[0], R[1], R[2], R[3])):
            if (rects != None): rects.append(R)
            mergeFrags(nf, traceSkeleton(im, R[0], R[1], R[2], R[3], csize, maxIter - 1, rects), mj, HORIZONTAL)

    frags += nf
    if (mi == -1 and mj == -1):  # splitting failed! do the recursive bottom instead
        frags += chunkToFrags(im, x, y, w, h)

    return frags

def compute_thickness(mask, real):
    roi = mask * real
    deno = np.sum(np.sum(mask / 255.))  # the overall pixels
    return np.sum(np.sum(roi / 255.)) / deno


def cut_vessel(im0, polys, ratio=0.9, thickness=6):
    black_mask = np.zeros((im0.shape[0], im0.shape[1], 3)).astype(np.uint8)
    points = []
    thick_list = []
    for ind, l in enumerate(polys):
        temp_map = np.zeros((im0.shape[0], im0.shape[1], 3)).astype(np.uint8)
        for i in range(0, len(l) - 1):
            cv2.line(temp_map, (l[i][0], l[i][1]), (l[i + 1][0], l[i + 1][1]), [255, 255, 255], thickness)
        thick_list.append([compute_thickness(temp_map[:, :, 0], im0), ind])

    thick_list.sort(reverse=True)
    sort_index = []

    for x in thick_list:
        sort_index.append(x[1])
    poly_s = []

    for ind in range(len(polys)):
        poly_s.append(polys[sort_index[ind]])

    print(len(polys))

    # for ind in range(int(len(poly_s))):
    for ind in range(int(len(poly_s) * ratio)):
        l = poly_s[ind]
        if True:  # (ind <= int(len(poly_s)*ratio)) or ( (ind > int(len(poly_s)*ratio)) and (np.random.rand() <= 0.5)):
            for i in range(0, len(l) - 1):
                cv2.line(black_mask, (l[i][0], l[i][1]), (l[i + 1][0], l[i + 1][1]), [255, 255, 255], thickness)

    processed_map = im0 * (black_mask[:, :, 0] / 255.)
    return processed_map, poly_s

def count_ones_around(image, inp):
    count = 0
    y,x=inp
    for i in range(-2, 3):  # 遍历相邻5x5的区域
        for j in range(-2, 3):
            if 0 <= x + i < image.shape[0] and 0 <= y + j < image.shape[1]:
                if image[x + i, y + j] > 0:
                    count += 1

    return count
def img_skel(im0):
    im1 = (im0[:, :] > 0.5).astype(np.uint8)  # 黑白图0-1
    im = thinning(im1)  # 确实很瘦 0-1
    rects = []
    polys = traceSkeleton(im, 0, 0, im.shape[1], im.shape[0], 6, 999, rects)

    line = np.zeros((512, 512)).astype(np.uint8)
    line1 = np.zeros((512, 512)).astype(np.uint8)
    point = np.zeros((512, 512)).astype(np.uint8)
    # line = np.zeros((584,565)).astype(np.uint8)
    # line1 = np.zeros((584,565)).astype(np.uint8)
    # point = np.zeros((584,565)).astype(np.uint8)
    # line = np.zeros((300,300)).astype(np.uint8)
    # line1 = np.zeros((300,300)).astype(np.uint8)
    # point = np.zeros((300,300)).astype(np.uint8)
    # line = np.zeros((288,288)).astype(np.uint8)
    # line1 = np.zeros((288,288)).astype(np.uint8)
    # point = np.zeros((288,288)).astype(np.uint8)
    # line = np.zeros((960,999)).astype(np.uint8)
    # line1 = np.zeros((960,999)).astype(np.uint8)
    # point = np.zeros((960,999)).astype(np.uint8)
    # line = np.zeros((960,960)).astype(np.uint8)
    # line1 = np.zeros((960,960)).astype(np.uint8)
    # point = np.zeros((960,960)).astype(np.uint8)
    # line = np.zeros((605,700)).astype(np.uint8)
    # line1 = np.zeros((605,700)).astype(np.uint8)
    # point = np.zeros((605,700)).astype(np.uint8)


    # plt.imshow(im1, cmap='gray')

    #######这里是输出原来的分支图
    # for poly1 in polys:
    #     d = len(poly1)*1
    #
    #     for i in range(len(poly1) - 1):
    #         x1, y1 = poly1[i]
    #         x2, y2 = poly1[i + 1]
    #         color = 255/(len(poly1)*1)*d
    #         d-=1
    #         cv2.line(point, poly1[i], poly1[i + 1],color,1)
    #     for i in range(-1,1):#头尾
    #         x,y=poly1[i]
    #         point[y,x]=255
    #     # for i in poly1:#全部
    #     #     x,y=i
    #     #     point[y,x]=255

    i = 0
    ###########这里是对根进行排序
    for poly1 in polys:
        a = count_ones_around(im * 255, poly1[0])
        b = count_ones_around(im * 255, poly1[-1])
        if a >= b:  # 在这里我们设置，a是比较根部的部分
            i = i + 1
            continue
        else:
            # ploy1 = poly1[::-1]
            polys[i] = polys[i][::-1]
            # polys[i] = poly1
        i = i + 1

    for poly1 in polys:
        x,y = poly1[0]
        point[y, x] = 255
        # plt.scatter(x, y, marker='o', s=6, alpha=1, color='m')  # 调整圆点大小和颜色
        x, y = poly1[-1]
        point[y, x] = 255
        # plt.scatter(x, y, marker='o', s=6, alpha=1, color='b')  # 调整圆点大小和颜色




    ##########这里是输出排序后的分支图
    for poly1 in polys:
        d = len(poly1) * 1
        k = 0
        # for i in poly1:
        #     x,y=i
        #     point[y,x]=255
        for i in range(len(poly1) - 1):
            x1, y1 = poly1[i]
            x2, y2 = poly1[i + 1]
            color = 255 / (len(poly1) + 2 * 1) * (d + 2)  # 正
            color1 = 255 / (len(poly1) + 2 * 1) * (k + 2)  # 反
            d -= 1
            k = k + 1
            cv2.line(line, poly1[i], poly1[i + 1], color, 3)
            cv2.line(line1, poly1[i], poly1[i + 1], color1, 3)
    #
    #
    # imageio.imwrite('xor.png', cv2.bitwise_xor(line, point))  # 输出细血管mask
    # imageio.imwrite('test/skel.png', im*255)  # 输出细血管mask
    # imageio.imwrite('test/dian.png', point)#输出细血管mask
    # imageio.imwrite('test/xian.png', line)  # 输出细血管mask
    line = line * im1
    line1 = line1 * im1

    # plt.axis('off')
    # plt.savefig('test_xx.png', dpi=800, pad_inches=0, bbox_inches='tight')

    return line/255.,line1/255.,point/255.
if __name__ == "__main__":

    R = 0.6  # noise ratio
    TAU = 3  # thickness threshold


    import glob

    image_path = '/root/daima/YuzuSoft/dataset1/DRIVE/training/images'
    new_image_path = '../dataset1/DRIVEN/training/images'#粗
    os.makedirs(new_image_path, exist_ok=True)
    new_image_path1 = '../dataset1/DRIVEN/training/images1'#细
    os.makedirs(new_image_path1, exist_ok=True)

    all_file = glob.glob('/root/daima/YuzuSoft/log/visual_results/DRIVE/pre_3.png') #输入图像  *.gif
    for f in all_file:
        im0 = imageio.imread(f)
        # black_map = np.zeros((im0.shape[0], im0.shape[1], 3)).astype(np.uint8)
        im1 = (im0[:, :] > 128).astype(np.uint8) #黑白图0-1

        im = thinning(im1)  #确实很瘦 0-1
        rects = []


        polys = traceSkeleton(im, 0, 0, im.shape[1], im.shape[0], 6, 999, rects)
        temp = np.zeros((584,565)).astype(np.uint8)
        point = np.zeros((584,565)).astype(np.uint8)
        line = np.zeros((584, 565)).astype(np.uint8)
        line1 = np.zeros((584, 565)).astype(np.uint8)


        #######这里是输出原来的分支图
        # for poly1 in polys:
        #     d = len(poly1)*1
        #
        #     for i in range(len(poly1) - 1):
        #         x1, y1 = poly1[i]
        #         x2, y2 = poly1[i + 1]
        #         color = 255/(len(poly1)*1)*d
        #         d-=1
        #         cv2.line(point, poly1[i], poly1[i + 1],color,1)
        #     for i in range(-1,1):#头尾
        #         x,y=poly1[i]
        #         point[y,x]=255
        #     # for i in poly1:#全部
        #     #     x,y=i
        #     #     point[y,x]=255

        i = 0
        ###########这里是对根进行排序
        for poly1 in polys:
            a = count_ones_around(im*255,poly1[0])
            b = count_ones_around(im*255,poly1[-1])
            if a>=b:    #在这里我们设置，a是比较根部的部分
                i = i + 1
                continue
            else:
                # ploy1 = poly1[::-1]
                polys[i] = polys[i][::-1]
                # polys[i] = poly1
            i = i + 1
            x, y = poly1[0]
            point[y, x] = 255
            x, y = poly1[-1]
            point[y, x] = 255


        plt.imshow(im1, cmap='gray')
        for poly1 in polys:
            x, y = poly1[-1]
            point[y, x] = 255
            plt.scatter(x, y, marker='o', s=6, alpha=1, color='b')  # 调整圆点大小和颜色
            x, y = poly1[0]
            point[y, x] = 255
            plt.scatter(x, y, marker='o', s=6, alpha=1, color='m')  # 调整圆点大小和颜色

        plt.axis('off')
        # plt.savefig('test_xx.png', dpi=800, pad_inches=0, bbox_inches='tight')
        ##########这里是输出排序后的分支图
        for poly1 in polys:
            d = len(poly1)*1
            k=0
            # for i in poly1:
            #     x,y=i
            #     point[y,x]=255
            for i in range(len(poly1) - 1):
                x1, y1 = poly1[i]
                x2, y2 = poly1[i + 1]
                color = 255/(len(poly1)+1*1)*(d+1) #正
                color1 = 255/(len(poly1)+1*1)*(k+1) #反
                d-=1
                k=k+1
                cv2.line(line, poly1[i], poly1[i + 1],color,3)
                cv2.line(line1, poly1[i], poly1[i + 1], color1, 3)
        #
        #
        # imageio.imwrite('xor.png', cv2.bitwise_xor(line, point))  # 输出细血管mask
        # imageio.imwrite('test/skel.png', im*255)  # 输出细血管mask
        # imageio.imwrite('test/dian.png', point)#输出细血管mask
        # imageio.imwrite('test/xian.png', line)  # 输出细血管mask
        line = line * im1
        line1 = line1 * im1
        print(f)
        print(f.replace('input', 'output_root'))
        imageio.imwrite(f.replace('input', 'output_root').replace('pre_','').replace('.png','_.png'), line) # 输出root
        print(f.replace('input', 'output_edge'))
        imageio.imwrite(f.replace('input', 'output_edge'.replace('pre_','').replace('.png','_.png')), line1)  # 输出edge
        print(f.replace('input', 'point'))
        imageio.imwrite(f.replace('input', 'out_put_point'.replace('pre_', '').replace('.png', '_.png')), point)  # 输出edge


# #        polys = np.load(os.path.join(poly_file, f.split('/')[-1]), allow_pickle=True)
# #         np.save(os.path.join(poly_file, f.split('/')[-1].replace('gif', 'npy')), polys)
#         # Select the ratio
#         cut_map, poly_s = cut_vessel(im0, polys, R, TAU)
#         # np.save(os.path.join(polys_file, f.split('/')[-1].replace('gif', 'npy')), poly_s)
#         imageio.imwrite(os.path.join(output_file, f.split('/')[-1]), cut_map)#粗血管mask输出
#         cut_map = cut_map / 255
#
#         image = Image.open(os.path.join(image_path, f.split('/')[-1].replace('manual1.gif', 'training.tif')))
#
#         image = np.asarray(image).astype(np.uint8)
#         big_mask = 1 - cut_map
#         image[:, :, 0] = image[:, :, 0] * big_mask
#         image[:, :, 1] = image[:, :, 1] * big_mask
#         image[:, :, 2] = image[:, :, 2] * big_mask
#         image = Image.fromarray(image.astype(np.uint8))
#         image.save(os.path.join(new_image_path1, f.split('/')[-1].replace('manual1.gif', 'training.tif')))
#
#         print('cut_mpa!!',f.split('/')[-1])
#         mask = np.logical_xor(cut_map, im1)
#         mask = mask.astype(np.uint8)
#         mask_out = mask * 255
#         imageio.imwrite(os.path.join(mask_file, f.split('/')[-1]), mask_out)#输出细血管mask
#         image = Image.open(os.path.join(image_path, f.split('/')[-1].replace('manual1.gif', 'training.tif')))
#         image = np.asarray(image).astype(np.uint8)
#         mask = 1 - mask
#         image[:, :, 0] = image[:, :, 0] * mask
#         image[:, :, 1] = image[:, :, 1] * mask
#         image[:, :, 2] = image[:, :, 2] * mask
#         image = Image.fromarray(image.astype(np.uint8))
#         image.save(os.path.join(new_image_path, f.split('/')[-1].replace('manual1.gif', 'training.tif')))