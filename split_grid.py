import os
import cv2
import numpy as np

from matplotlib import pyplot as plt

def _find_euclidean_distance(pt1, pt2):
    pt1x = pt1[0]
    pt1y = pt1[1]
    pt2x = pt2[0]
    pt2y = pt2[1]
    dist = np.int0(np.round(np.sqrt((max(pt1x,pt2x) - min(pt1x, pt2x)) ** 2 + (max(pt1y,pt2y) - min(pt1y, pt2y)) ** 2)))

    return dist


def _auto_canny(image_mat):
    #automatically determines threshold for optimal edge detection
    med = np.median(image_mat)
    sigma = 0.33
    low = int(max(0, (1.0-sigma)*med))
    high = int(min(255, (1.0+sigma)*med))

    edges = cv2.Canny(image_mat, low, high)

    return edges


def eliminate_bg(edges):
    #Use Hough transform to obtain lines in image
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=20, minLineLength=20, maxLineGap=10)

    im_lines = np.zeros(shape=np.shape(im_gray), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(im_lines, (x1, y1), (x2, y2), (255, 255, 255), 2)

    #Dilate lines using large kernel to connect thin or broken lines
    kernel = np.ones((20, 20), np.uint8)
    im_lines_mod = cv2.dilate(im_lines, kernel)

    #Find contour of overall grid (largest contour with 4 points)
    _, contours, hierarchy = cv2.findContours(im_lines_mod.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        contour_pts = cv2.approxPolyDP(contour, 0.1 * perimeter, True)
        cv2.drawContours(edges, [contour_pts], -1, (255, 255, 255), 10)

        if len(contour_pts) == 4:
            grid_contour = contour_pts
            break

    #mask = np.zeros(shape=np.shape(im_gray), dtype=np.uint8)
    #cv2.drawContours(mask, [grid_contour], -1, (255, 255, 255), cv2.FILLED)

    #Determine bounding box for grid contour
    target_rect = cv2.minAreaRect(grid_contour)
    box = cv2.boxPoints(target_rect)
    box = np.int0(box)

    #determine Euclidean distance of points to obtain length and width
    height_grid = _find_euclidean_distance(box[3], box[0])
    width_grid = _find_euclidean_distance(box[3], box[2])

    #find homography to superimpose grid to front view
    h, mask = cv2.findHomography(box, np.array([[width_grid, height_grid], [0, height_grid], [0, 0], [width_grid, 0]]),
                                 cv2.RANSAC)

    #warp perspective to obtain front view of grid
    target = cv2.warpPerspective(im_gray, h, (width_grid, height_grid))

    return target

def extract_boxes(grid_mat):
    box_list = []

    grid_mat = cv2.adaptiveThreshold(grid_mat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 15)
    edges = _auto_canny(grid_mat)

    #Use Hough transform to obtain lines in image
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=0, minLineLength=0, maxLineGap=0)

    im_lines = np.zeros(shape=np.shape(grid_mat), dtype=np.uint8)

    for i, line in enumerate(lines):
        for x1, y1, x2, y2 in line:
            cv2.line(im_lines, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Dilate lines using large kernel to connect thin or broken lines
    kernel = np.ones((10, 10), np.uint8)
    im_lines_mod = cv2.dilate(im_lines, kernel)

    im_lines_mod = cv2.bitwise_not(im_lines_mod)

    # Find contour of overall grid (largest contour with 4 points)
    _, contours, hierarchy = cv2.findContours(cv2.bitwise_not(im_lines_mod.copy()), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    box_contours = []

    print(len(contours))
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        contour_pts = cv2.approxPolyDP(contour, 0.1 * perimeter, True)

        if len(contour_pts) == 4:
            box_contours.append(contour_pts)

    for contour in box_contours:
        cv2.drawContours(edges, [contour], -1, (255, 255, 255), 10)

    plt.figure()
    plt.subplot(131), plt.imshow(grid_mat, cmap='gray')
    plt.subplot(132), plt.imshow(edges, cmap='gray')
    plt.subplot(133), plt.imshow(cv2.bitwise_not(im_lines_mod), cmap='gray')

    plt.show()

    return box_list

    pass


if __name__ == "__main__":
    original_images = os.listdir("./images/original")

    for image_file in original_images:
        print(image_file)
        im = cv2.imread("./images/original/" + image_file)

        if im.size == 0:
            print("Could not load " + image_file)
            continue

        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # Intial noise adjustment
        im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

        edges = _auto_canny(im_gray)
        grid_mat = eliminate_bg(edges)
        boxes = extract_boxes(grid_mat)


        #plt.figure()
        #plt.subplot(131), plt.imshow(im_gray, cmap='gray')
        #plt.subplot(132), plt.imshow(edges, cmap='gray')
        #plt.subplot(133), plt.imshow(grid_mat, cmap='gray')

        #plt.show()