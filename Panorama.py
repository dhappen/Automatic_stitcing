import cv2
import numpy as np
def ransac(points1, points2, max_iters=1000, threshold=4.0):
    best_H = None
    best_inliers = 0

    for _ in range(max_iters):
        indices = np.random.choice(len(points1), 4, replace=False)
        A = []

        for i in indices:
            x1, y1 = points1[i]
            x2, y2 = points2[i]
            A.append([-x1, -y1, -1, 0, 0, 0, x2 * x1, x2 * y1, x2])
            A.append([0, 0, 0, -x1, -y1, -1, y2 * x1, y2 * y1, y2])

        A = np.array(A)
        _, _, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)

        inliers = 0

        for i in range(len(points1)):
            x1, y1 = points1[i]
            x2, y2 = points2[i]
            p1 = np.array([x1, y1, 1])
            p2_est = np.dot(H, p1)
            p2_est /= p2_est[2]
            err = np.sqrt((p2_est[0] - x2) ** 2 + (p2_est[1] - y2) ** 2)

            if err < threshold:
                inliers += 1

        if inliers > best_inliers:
            best_inliers = inliers
            best_H = H

    return best_H

def warp_image(image, homography, output_size):
    h, w = image.shape[:2]
    output_image = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
    inverse_homography = np.linalg.inv(homography)

    for y_out in range(output_size[1]):
        for x_out in range(output_size[0]):
            p_out = np.array([x_out, y_out, 1])
            p_in = np.dot(inverse_homography, p_out)
            p_in = p_in / p_in[2]

            x_in, y_in = int(p_in[0]), int(p_in[1])

            if 0 <= x_in < w and 0 <= y_in < h:
                if x_in < w - 1 and y_in < h - 1:
                    # Bilinear interpolation
                    dx = p_in[0] - x_in
                    dy = p_in[1] - y_in

                    pixel_value = (1 - dx) * (1 - dy) * image[y_in, x_in] + dx * (1 - dy) * image[y_in, x_in + 1] + (1 - dx) * dy * image[y_in + 1, x_in] + dx * dy * image[y_in + 1, x_in + 1]

                    output_image[y_out, x_out] = pixel_value.astype(np.uint8)
                else:
                    output_image[y_out, x_out] = image[y_in, x_in]

    return output_image

# prepare images
image1 = cv2.imread('image3.jpg')
image2 = cv2.imread('image4.jpg')
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Find keypoints and descriptors
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

# bfmatch
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# Sort them in ascending order of distance
matches = sorted(matches, key=lambda x: x.distance)

# Extract matched keypoints
matched_keypoints1 = np.array([keypoints1[match.queryIdx].pt for match in matches])
matched_keypoints2 = np.array([keypoints2[match.trainIdx].pt for match in matches])

# Use RANSAC to estimate the homography matrix
homography = ransac(matched_keypoints2, matched_keypoints1)

# Warp image2 to image1 using the homography matrix
output_size = (image1.shape[1] + image2.shape[1], image2.shape[0])
warp_image2 = warp_image(image2, homography, output_size)
warp_image2[0:image1.shape[0], 0:image1.shape[1]] = image1

# Save the panorama image
cv2.imwrite("result_34.jpg", warp_image2)

# Display the panorama image
# cv2.imshow("Panorama", warp_image2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
