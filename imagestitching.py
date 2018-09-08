from __future__ import print_function  #
import cv2
import argparse
import os
import numpy as np
import copy

file_name_mapping = {}

def up_to_step_1(imgs):
    """Complete pipeline up to step 3: Detecting features and descriptors"""
    # ... your code here ...
    new_imgs = []
    for filename in imgs:
        new_img = copy.deepcopy(file_name_mapping[filename])    
        grey_image= cv2.cvtColor(new_img,cv2.COLOR_BGR2GRAY)
        surf = cv2.xfeatures2d.SURF_create()
        surf.setHessianThreshold(1500)
        kp = surf.detect(grey_image,None)
        new_imgs.append((filename, cv2.drawKeypoints(grey_image,kp,new_img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)))
    return new_imgs


def save_step_1(imgs, output_path='./output/step1'):
    """Save the intermediate result from Step 1"""
    # ... your code here ...
    for image in imgs:
        cv2.imwrite(os.path.join(output_path,image[0]),image[1])
    pass


def calc_euc_dist(vec1, vec2):
    dist = np.linalg.norm(np.array(vec1) - np.array(vec2))
    return dist

def up_to_step_2(imgs, center_image=None):
    """Complete pipeline up to step 2: Calculate matching feature points"""
    # ... your code here ...
    imgs_after_step_1 = up_to_step_1(imgs)

    image_pairs = []
    match_list = []

    for idx_1, img_1 in enumerate(imgs_after_step_1):
        for idx_2, img_2 in enumerate(imgs_after_step_1):
            if idx_1 < idx_2:
                if center_image == None  or center_image == img_1[0] or center_image == img_2[0]:
                    image_pairs.append([img_1[0] + "_" + img_2[0],img_1,img_2])

    for image_pair in image_pairs:
        print("Image pair ", image_pair[0])
        surf = cv2.xfeatures2d.SURF_create()
        surf.setHessianThreshold(1500)
        img_1 = file_name_mapping[image_pair[1][0]]
        kp1, des1 = surf.detectAndCompute(img_1,None)
        img_2 = file_name_mapping[image_pair[2][0]]
        kp2, des2 = surf.detectAndCompute(img_2,None)
        image_pair.append([len(kp1),len(kp2)])
        n1 = len(kp1)
        n2 = len(kp2)
        kp1_kp2 = {}
        kp2_kp1 = {}

        for idx1 in range(n1):
            min_dist = None
            sec_min_dist = None
            for idx2 in range(n2):
                dist = calc_euc_dist(des1[idx1], des2[idx2])
                if type(min_dist) != list or min_dist[0] > dist:
                    if type(min_dist) == list:
                        sec_min_dist = min_dist
                    min_dist = [dist, idx2]
                if (type(sec_min_dist) != list or sec_min_dist[0] < dist < min_dist[0]) and min_dist[1] != idx2:
                    sec_min_dist = [dist, idx2]
            if type(min_dist) == list and type(sec_min_dist) == list and min_dist[0] < sec_min_dist[0] * 0.75:
                kp1_kp2[kp1[idx1]] = kp2[min_dist[1]]

        for idx2 in range(n2):
            min_dist = None
            sec_min_dist = None
            for idx1 in range(n1):
                dist = calc_euc_dist(des2[idx2], des1[idx1])
                if type(min_dist) != list or min_dist[0] > dist:
                    if type(min_dist) == list:
                        sec_min_dist = min_dist
                    min_dist = [dist, idx1]
                if (type(sec_min_dist) != list or sec_min_dist[0] < dist < min_dist[0]) and min_dist[1] != idx1:
                    sec_min_dist = [dist, idx1]
            if type(min_dist) == list and type(sec_min_dist) == list and min_dist[0] < sec_min_dist[0] * 0.75:
                kp2_kp1[kp2[idx2]] = kp1[min_dist[1]]

        matches = ([],[])
        for kp in kp1_kp2:
            closest = kp1_kp2[kp]
            if closest in kp2_kp1 and kp2_kp1[closest] == kp:
                matches[0].append(kp)
                matches[1].append(kp1_kp2[kp])

        match_list.append(matches)
    return image_pairs, match_list


def save_step_2(imgs, match_list, output_path="./output/step2"):
    """Save the intermediate result from Step 2"""
    # ... your code here ...

    for idx, img_pair in enumerate(imgs):
        img_1_name = img_pair[1][0]
        img_2_name = img_pair[2][0]
        total_kp1 = img_pair[3][0]
        total_kp2 = img_pair[3][1]

        image_1 = file_name_mapping[img_1_name]
        image_2 = file_name_mapping[img_2_name]

        combined_img = np.concatenate((image_1, image_2), axis=1)
        vert_len = len(image_1)
        horizon_len = len(image_1[0])
        kp1 = match_list[idx][0]
        kp2 = match_list[idx][1]
        num_kp = len(kp1)
        if num_kp < 30:
            continue
        for i in range(num_kp):
            x1 = int(kp1[i].pt[0])
            y1 = int(kp1[i].pt[1])
            x2 = int(kp2[i].pt[0]+horizon_len)
            y2 = int(kp2[i].pt[1])
            cv2.circle(combined_img,(x1,y1), 10, (255,255,0), -1)
            cv2.circle(combined_img,(x2,y2), 10, (255,255,0), -1)
            cv2.line(combined_img,(x1,y1),(x2,y2),(255,255,0),5)
            combined_img_name = img_1_name[:-4] + "_" + str(total_kp1) + "_" + img_2_name[:-4] + "_" + str(total_kp2) + "_" + str(num_kp) + ".jpg"
            cv2.imwrite(os.path.join(output_path,combined_img_name),combined_img)

    pass

def calc_homography_matrix(img1_pts, img2_pts):
    x11 = int(img1_pts[0].pt[0])
    x12 = int(img1_pts[1].pt[0])
    x13 = int(img1_pts[2].pt[0])
    x14 = int(img1_pts[3].pt[0])
    y11 = int(img1_pts[0].pt[1])
    y12 = int(img1_pts[1].pt[1])
    y13 = int(img1_pts[2].pt[1])
    y14 = int(img1_pts[3].pt[1])

    x21 = int(img2_pts[0].pt[0])
    x22 = int(img2_pts[1].pt[0])
    x23 = int(img2_pts[2].pt[0])
    x24 = int(img2_pts[3].pt[0])
    y21 = int(img2_pts[0].pt[1])
    y22 = int(img2_pts[1].pt[1])
    y23 = int(img2_pts[2].pt[1])
    y24 = int(img2_pts[3].pt[1])

    row1 = [x21, y21, 1, 0, 0, 0, -x21*x11, -y21*x11]
    row2 = [0, 0, 0, x21, y21, 1, -x21*y11, -y21*y11]
    row3 = [x22, y22, 1, 0, 0, 0, -x22*x12, -y22*x12]
    row4 = [0, 0, 0, x22, y22, 1, -x22*y12, -y22*y12]
    row5 = [x23, y23, 1, 0, 0, 0, -x23*x13, -y23*x13]
    row6 = [0, 0, 0, x23, y23, 1, -x23*y13, -y23*y13]
    row7 = [x24, y24, 1, 0, 0, 0, -x24*x14, -y24*x14]
    row8 = [0, 0, 0, x24, y24, 1, -x24*y14, -y24*y14]

    A =  np.array([row1,row2,row3,row4,row5,row6,row7,row8])


    if np.linalg.det(A) == 0:
        return np.zeros((3,3))
    B = np.array([[x11],[y11],[x12],[y12],[x13],[y13],[x14],[y14]])
    H = list(np.linalg.solve(A,B).T[0])
    H.append(1)
    H = np.array([H[:3],H[3:6],H[6:]])

    return H

def h_matrix_accuracy(homography_matrix, img1_coords, img2_coords):
    if np.linalg.det(homography_matrix) == 0:
        return 0

    total = len(img1_coords)
    count_accurate = 0
    count_inaccurate = 0
    inv_homography_matrix = np.linalg.inv(homography_matrix)

    for idx in range(total):
        x1 = int(img1_coords[idx].pt[0])
        y1 = int(img1_coords[idx].pt[1])
        x2 = int(img2_coords[idx].pt[0])
        y2 = int(img2_coords[idx].pt[1])
        I1 = np.array([[x1,y1,1]])
        I2 = np.dot(inv_homography_matrix, I1.T)
        x2_est = I2[0][0]/I2[2][0]
        y2_est = I2[1][0]/I2[2][0]
        distance = calc_euc_dist([x2,y2], [x2_est,y2_est])
        if distance <= 10:
            count_accurate += 1
    accuracy_percentage = float(count_accurate)/float(total)
    # BONUS MARK FEATURE: ALSO RETURN ACCURATE ESTIMATIONS AND TOTAL MATCHES FOR THIS IMAGE-HOMOGRAPHY COMBINATION
    return accuracy_percentage, count_accurate, total

def up_to_step_3(imgs):
    """Complete pipeline up to step 3: estimating homographies and warpings"""
    # ... your code here ...
    modified_imgs, match_list = up_to_step_2(imgs)
    imgs = []
    for idx, pair_matches in enumerate(match_list):
        print("Step ", idx + 1)
        filename1 = modified_imgs[idx][1][0]
        img1 = copy.deepcopy(file_name_mapping[filename1])
        filename2 = modified_imgs[idx][2][0]
        img2 = copy.deepcopy(file_name_mapping[filename2])

        img1_coords = pair_matches[0]
        img2_coords = pair_matches[1]

        best_accuracy = 0
        best_H_matrix = np.zeros((3,3))
        homography_matrix = np.zeros((3,3))
        count = 0
        while count < 100 or (count >=  100 and np.linalg.det(homography_matrix) == 0):
            count +=1 
            quad = np.random.choice(range(len(img1_coords)), 4)
            img1_quad = [img1_coords[quad[0]],img1_coords[quad[1]],img1_coords[quad[2]],img1_coords[quad[3]]]
            img2_quad = [img2_coords[quad[0]],img2_coords[quad[1]],img2_coords[quad[2]],img2_coords[quad[3]]]
            homography_matrix = calc_homography_matrix(img1_quad, img2_quad)

            if np.linalg.det(homography_matrix) != 0:
                accuracy, count_accurate, total = h_matrix_accuracy(homography_matrix, img1_coords, img2_coords)
                if accuracy > best_accuracy:
                    best_H_matrix = homography_matrix
                    best_accuracy = accuracy
                    print(best_accuracy)
        if np.linalg.det(homography_matrix) == 0:
            continue

        homography_matrix = best_H_matrix
        inv_homography_matrix = np.linalg.inv(homography_matrix)

        vert_len = len(img1)
        horizon_len = len(img1[0])
        new_img_1 = np.zeros((vert_len, horizon_len, 3), dtype=np.uint8)

        for x1 in range(horizon_len):
            for y1 in range(vert_len):
                I1 = np.array([[x1,y1,1]])
                I2 = np.dot(inv_homography_matrix, I1.T)
                x2 = int(I2[0]/I2[2])
                y2 = int(I2[1]/I2[2])
                if  0 <= x2 < horizon_len and 0 <= y2 < vert_len:
                    new_img_1[y2][x2] = img1[y1][x1]
                if new_img_1[y1][x1][0] == 0 and new_img_1[y1][x1][1] == 0 and new_img_1[y1][x1][2] == 0:
                    I1 = np.array([[x1,y1,1]])
                    I2 = np.dot(homography_matrix, I1.T)
                    x2 = int(I2[0]/I2[2])
                    y2 = int(I2[1]/I2[2])
                    if  0 <= x2 < horizon_len and 0 <= y2 < vert_len:
                        new_img_1[y1][x1] = img1[y2][x2]

        # new_im  = cv2.warpPerspective(img1, inv_homography_matrix, (img1.shape[0], img1.shape[1]))
        # cv2.imwrite(filename1[:-4] + "_" + filename2, new_im)

        vert_len = len(img2)
        horizon_len = len(img2[0])
        new_img_2 = np.zeros((vert_len, horizon_len, 3), dtype=np.uint8)

        for x1 in range(horizon_len):
            for y1 in range(vert_len):
                I1 = np.array([[x1,y1,1]])
                I2 = np.dot(homography_matrix, I1.T)
                x2 = int(I2[0]/I2[2])
                y2 = int(I2[1]/I2[2])
                if  0 <= x2 < horizon_len - 1 and 0 <= y2 < vert_len - 1:
                    new_img_2[y2][x2] = img2[y1][x1]
                if new_img_2[y1][x1][0] == 0 and new_img_2[y1][x1][1] == 0 and new_img_2[y1][x1][2] == 0:
                    I1 = np.array([[x1,y1,1]])
                    I2 = np.dot(inv_homography_matrix, I1.T)
                    x2 = int(I2[0]/I2[2])
                    y2 = int(I2[1]/I2[2])
                    if  0 <= x2 < horizon_len and 0 <= y2 < vert_len:
                        new_img_2[y1][x1] = img2[y2][x2]

        # new_im  = cv2.warpPerspective(img2, homography_matrix, (img2.shape[0], img2.shape[1]))
        # cv2.imwrite(filename2[:-4] + "_" + filename1, new_im)
        
        imgs.append([[filename1,new_img_1],[filename2,new_img_2]])

    return imgs


def save_step_3(img_pairs, output_path="./output/step3"):
    """Save the intermediate result from Step 3"""
    # ... your code here ...
    for img_pair in img_pairs:
        filename1 = img_pair[0][0]
        filename2 = img_pair[1][0]
        cv2.imwrite(os.path.join(output_path,"warped_"+filename1[:-4] + "_reference_" + filename2),img_pair[0][1])
        cv2.imwrite(os.path.join(output_path,"warped_"+filename2[:-4] + "_reference_" + filename1),img_pair[1][1])
    pass


def up_to_step_4(imgs):
    """Complete the pipeline and generate a panoramic image"""
    # ... your code here ...
    center_image_filename = imgs[len(imgs)/2]
    center_image = file_name_mapping[center_image_filename]

    vert_len = len(center_image)
    horizon_len = len(center_image[0])
    buffer_image = np.zeros((vert_len*3, horizon_len*3,3), dtype=np.uint8)

    modified_imgs, match_list = up_to_step_2(imgs, center_image=center_image_filename)
    imgs = []
    counter = 0
    for idx, pair_matches in enumerate(match_list):
        filename1 = modified_imgs[idx][1][0]
        img1 = copy.deepcopy(file_name_mapping[filename1])
        filename2 = modified_imgs[idx][2][0]
        img2 = copy.deepcopy(file_name_mapping[filename2])
        if center_image_filename != filename1 and center_image_filename!= filename2:
            continue
        counter += 1
        print("Image number ", counter)

        img1_coords = pair_matches[0]
        img2_coords = pair_matches[1]

        best_accuracy = 0
        best_H_matrix = np.zeros((3,3))
        homography_matrix = np.zeros((3,3))
        count = 0
        while count < 100 or (count >=  100 and np.linalg.det(homography_matrix) == 0):
            count +=1 
            quad = np.random.choice(range(len(img1_coords)), 4)
            img1_quad = [img1_coords[quad[0]],img1_coords[quad[1]],img1_coords[quad[2]],img1_coords[quad[3]]]
            img2_quad = [img2_coords[quad[0]],img2_coords[quad[1]],img2_coords[quad[2]],img2_coords[quad[3]]]
            homography_matrix = calc_homography_matrix(img1_quad, img2_quad)
            if np.linalg.det(homography_matrix) != 0:
                accuracy, count_accurate, total = h_matrix_accuracy(homography_matrix, img1_coords, img2_coords)
                if accuracy > best_accuracy:
                    best_H_matrix = homography_matrix
                    best_accuracy = accuracy
                    print(best_accuracy)
        if np.linalg.det(homography_matrix) == 0:
            continue

        homography_matrix = best_H_matrix
        inv_homography_matrix = np.linalg.inv(homography_matrix)   

        if center_image_filename == filename2:  
            max_y = None
            max_x = None
            min_y = None
            min_x = None

            vert_len = len(img1)
            horizon_len = len(img1[0])
            for x1 in range(horizon_len):
                for y1 in range(vert_len):
                    I1 = np.array([[x1,y1,1]])
                    I2 = np.dot(inv_homography_matrix, I1.T)
                    x2 = int(I2[0]/I2[2])
                    y2 = int(I2[1]/I2[2])
                    if  -vert_len < y2 < vert_len*2 and -horizon_len < x2 < horizon_len*2:
                        if max_x is None or x2 > max_x:
                            max_x = x2
                        if max_y is None or y2 > max_y:
                            max_y = y2
                        if min_x is None or x2 < min_x:
                            min_x = x2
                        if min_y is None or y2 < min_y:
                            min_y = y2
                        buffer_image[y2+vert_len][x2+horizon_len] = img1[y1][x1]

            for x1 in range(min_x, max_x):
                for y1 in range(min_y, max_y):
                    if x1 >= len(buffer_image[0]) or y1 >= len(buffer_image):
                        continue
                    if buffer_image[y1+vert_len][x1+horizon_len][0] == 0 and buffer_image[y1+vert_len][x1+horizon_len][1] == 0 and buffer_image[y1+vert_len][x1+horizon_len][2] == 0:
                        I1 = np.array([[x1,y1,1]])
                        I2 = np.dot(homography_matrix, I1.T)
                        x2 = int(I2[0]/I2[2])
                        y2 = int(I2[1]/I2[2])
                        if  0 <= x2 < horizon_len and 0 <= y2 < vert_len:
                            buffer_image[y1+vert_len][x1+horizon_len] = img1[y2][x2]

        else:
            max_y = None
            max_x = None
            min_y = None
            min_x = None

            vert_len = len(img2)
            horizon_len = len(img2[0])
            for x1 in range(horizon_len):
                for y1 in range(vert_len):
                    I1 = np.array([[x1,y1,1]])
                    I2 = np.dot(homography_matrix, I1.T)
                    x2 = int(I2[0]/I2[2])
                    y2 = int(I2[1]/I2[2])
                    if  -vert_len < y2 < vert_len*2 and -horizon_len < x2 < horizon_len*2:
                        if max_x is None or x2 > max_x:
                            max_x = x2
                        if max_y is None or y2 > max_y:
                            max_y = y2
                        if min_x is None or x2 < min_x:
                            min_x = x2
                        if min_y is None or y2 < min_y:
                            min_y = y2
                        buffer_image[y2+vert_len][x2+horizon_len] = img2[y1][x1]

            for x1 in range(min_x, max_x):
                for y1 in range(min_y, max_y):
                    if x1 >= len(buffer_image[0]) or y1 >= len(buffer_image):
                        continue
                    if buffer_image[y1+vert_len][x1+horizon_len][0] == 0 and buffer_image[y1+vert_len][x1+horizon_len][1] == 0 and buffer_image[y1+vert_len][x1+horizon_len][2] == 0:
                        I1 = np.array([[x1,y1,1]])
                        I2 = np.dot(inv_homography_matrix, I1.T)
                        x2 = int(I2[0]/I2[2])
                        y2 = int(I2[1]/I2[2])
                        if  0 <= x2 < horizon_len and 0 <= y2 < vert_len:
                            buffer_image[y1+vert_len][x1+horizon_len] = img2[y2][x2]

    buffer_image[vert_len:2*vert_len, horizon_len:2*horizon_len] = center_image

    return buffer_image


def save_step_4(panoramic_img, output_path="./output/step4"):
    """Save the intermediate result from Step 4"""
    # ... your code here ...
    cv2.imwrite(os.path.join(output_path,"panoramic_img.jpg"), panoramic_img)

    pass

def up_to_step_5(imgs):
    """Complete the pipeline and generate a panoramic image"""
    # ... your code here ...
    center_image_filename = imgs[len(imgs)/2]
    center_image = file_name_mapping[center_image_filename]

    vert_len = len(center_image)
    horizon_len = len(center_image[0])
    buffer_image = np.zeros((vert_len*3, horizon_len*3,3), dtype=np.uint8)

    modified_imgs, match_list = up_to_step_2(imgs, center_image=center_image_filename)
    imgs = []
    counter = 0
    total_accurate_matches = 0
    total_matches_checked = 0
    for idx, pair_matches in enumerate(match_list):
        filename1 = modified_imgs[idx][1][0]
        img1 = copy.deepcopy(file_name_mapping[filename1])
        filename2 = modified_imgs[idx][2][0]
        img2 = copy.deepcopy(file_name_mapping[filename2])
        if center_image_filename != filename1 and center_image_filename!= filename2:
            continue
        counter += 1

        img1_coords = pair_matches[0]
        img2_coords = pair_matches[1]

        best_accuracy = 0
        best_H_matrix = np.zeros((3,3))
        # BONUS MARK FEATURE: TRACKER FOR NUMBER OF MATCHES CHECKED AND THE NUMBER OF ACCURATE MATCHES
        best_count_accurate = 0
        best_total = 0
        homography_matrix = np.zeros((3,3))
        count = 0

        while count < 100 or (count >=  100 and np.linalg.det(homography_matrix) == 0):
            count +=1 
            quad = np.random.choice(range(len(img1_coords)), 4)
            img1_quad = [img1_coords[quad[0]],img1_coords[quad[1]],img1_coords[quad[2]],img1_coords[quad[3]]]
            img2_quad = [img2_coords[quad[0]],img2_coords[quad[1]],img2_coords[quad[2]],img2_coords[quad[3]]]
            homography_matrix = calc_homography_matrix(img1_quad, img2_quad)
            if np.linalg.det(homography_matrix) != 0:
                accuracy, count_accurate, total = h_matrix_accuracy(homography_matrix, img1_coords, img2_coords)
                if accuracy > best_accuracy:
                    best_H_matrix = homography_matrix
                    best_accuracy = accuracy
                    # BONUS MARK FEATURE: UPDATE THE NUMBER OF MATCHES CHECKED AND THE NUMBER OF ACCURATE MATCHES FOR CURRENT IMAGE USING BEST HOMOGRAPHY
                    best_count_accurate = count_accurate
                    best_total = total
        # BONUS MARK FEATURE: UPDATE THE TOTAL NUMBER OF MATCHES CHECKED AND THE NUMBER OF ACCURATE MATCHES
        total_accurate_matches += count_accurate
        total_matches_checked += total
        if np.linalg.det(homography_matrix) == 0:
            continue

        homography_matrix = best_H_matrix
        inv_homography_matrix = np.linalg.inv(homography_matrix)   

        if center_image_filename == filename2:  
            max_y = None
            max_x = None
            min_y = None
            min_x = None

            vert_len = len(img1)
            horizon_len = len(img1[0])
            for x1 in range(horizon_len):
                for y1 in range(vert_len):
                    I1 = np.array([[x1,y1,1]])
                    I2 = np.dot(inv_homography_matrix, I1.T)
                    x2 = int(I2[0]/I2[2])
                    y2 = int(I2[1]/I2[2])
                    if  -vert_len < y2 < vert_len*2 and -horizon_len < x2 < horizon_len*2:
                        if max_x is None or x2 > max_x:
                            max_x = x2
                        if max_y is None or y2 > max_y:
                            max_y = y2
                        if min_x is None or x2 < min_x:
                            min_x = x2
                        if min_y is None or y2 < min_y:
                            min_y = y2
                        buffer_image[y2+vert_len][x2+horizon_len] = img1[y1][x1]

            for x1 in range(min_x, max_x):
                for y1 in range(min_y, max_y):
                    if x1 >= len(buffer_image[0]) or y1 >= len(buffer_image):
                        continue
                    if buffer_image[y1+vert_len][x1+horizon_len][0] == 0 and buffer_image[y1+vert_len][x1+horizon_len][1] == 0 and buffer_image[y1+vert_len][x1+horizon_len][2] == 0:
                        I1 = np.array([[x1,y1,1]])
                        I2 = np.dot(homography_matrix, I1.T)
                        x2 = int(I2[0]/I2[2])
                        y2 = int(I2[1]/I2[2])
                        if  0 <= x2 < horizon_len and 0 <= y2 < vert_len:
                            buffer_image[y1+vert_len][x1+horizon_len] = img1[y2][x2]

        else:
            max_y = None
            max_x = None
            min_y = None
            min_x = None

            vert_len = len(img2)
            horizon_len = len(img2[0])
            for x1 in range(horizon_len):
                for y1 in range(vert_len):
                    I1 = np.array([[x1,y1,1]])
                    I2 = np.dot(homography_matrix, I1.T)
                    x2 = int(I2[0]/I2[2])
                    y2 = int(I2[1]/I2[2])
                    if  -vert_len < y2 < vert_len*2 and -horizon_len < x2 < horizon_len*2:
                        if max_x is None or x2 > max_x:
                            max_x = x2
                        if max_y is None or y2 > max_y:
                            max_y = y2
                        if min_x is None or x2 < min_x:
                            min_x = x2
                        if min_y is None or y2 < min_y:
                            min_y = y2
                        buffer_image[y2+vert_len][x2+horizon_len] = img2[y1][x1]

            for x1 in range(min_x, max_x):
                for y1 in range(min_y, max_y):
                    if x1 >= len(buffer_image[0]) or y1 >= len(buffer_image):
                        continue
                    if buffer_image[y1+vert_len][x1+horizon_len][0] == 0 and buffer_image[y1+vert_len][x1+horizon_len][1] == 0 and buffer_image[y1+vert_len][x1+horizon_len][2] == 0:
                        I1 = np.array([[x1,y1,1]])
                        I2 = np.dot(inv_homography_matrix, I1.T)
                        x2 = int(I2[0]/I2[2])
                        y2 = int(I2[1]/I2[2])
                        if  0 <= x2 < horizon_len and 0 <= y2 < vert_len:
                            buffer_image[y1+vert_len][x1+horizon_len] = img2[y2][x2]


    # BONUS MARK FEATURE: CODE FOR PRINTING OUT STITCHING QUALITY ASSESSMENT
    print("ASSESSMENT OF STITCHING QUALITY\n")
    print("NUMBER OF MATCHING POINTS BETWEEN IMAGES CHECKED: ", total_matches_checked, "\n")
    print("NUMBER OF WARPED COORDINATES WITHIN 10px OF ITS MATCHING POINT: ", total_accurate_matches, "\n")
    if total_matches_checked != 0:
        print("PREDICTION ACCURACY: ", "{:.0%}".format(float(total_accurate_matches)/float(total_matches_checked)))

    buffer_image[vert_len:2*vert_len, horizon_len:2*horizon_len] = center_image

    return buffer_image


def save_step_5(panoramic_img, output_path="./output/step4"):
    """Save the intermediate result from Step 4"""
    # ... your code here ...
    cv2.imwrite(os.path.join(output_path,"panoramic_img.jpg"), panoramic_img)

    pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "step",
        help="compute image stitching pipeline up to this step",
        type=int
    )

    parser.add_argument(
        "input",
        help="a folder to read in the input images",
        type=str
    )

    parser.add_argument(
        "output",
        help="a folder to save the outputs",
        type=str
    )

    args = parser.parse_args()

    filenames = [] #filenames
    for filename in os.listdir(args.input):
        print(filename)
        img = cv2.imread(os.path.join(args.input, filename))
        file_name_mapping[filename] = img
        filenames.append(filename)
    if args.step == 1:
        print("Running step 1")
        modified_imgs = up_to_step_1(filenames)
        save_step_1(modified_imgs, args.output)
    elif args.step == 2:
        print("Running step 2")
        modified_imgs, match_list = up_to_step_2(filenames)
        save_step_2(modified_imgs, match_list, args.output)
    elif args.step == 3:
        print("Running step 3")
        img_pairs = up_to_step_3(filenames)
        save_step_3(img_pairs, args.output)
    elif args.step == 4:
        print("Running step 4")
        panoramic_img = up_to_step_4(filenames)
        save_step_4(panoramic_img, args.output)
    elif args.step == 5:
        print("Running step 5")
        # BONUS MARK FEATURE: STEP 5
        # FORMAT: PRINTED TO STANDARD OUTPUT
        # EVALUATING FEATURE MATCHING PERFORMANCE
        # METRIC: %TP, the percentage of correctly matched keypoints out of the total number of matches, also referred to as precision
        # REFERENCE: Page 2138, last paragraph, report: http://aeriksson.net/papers/marmol-etal-ral-2017.pdf
        panoramic_img = up_to_step_5(filenames)
        save_step_5(panoramic_img, args.output)