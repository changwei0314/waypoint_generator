import cv2
import math
import queue

import numpy as np
from rdp import rdp


########## Check if AB degree less than BC degree ##########

def calculate_if_degree_is_smaller(A, B, C, white):

    # Define vectors A, B, and C as lists of their components
    # A: start to last point in the path
    # B: start to end point
    # C: start to mid

    # Calculate the dot products of A and B, and C and B
    dot_product_AB = sum([A[i] * B[i] for i in range(len(A))])
    dot_product_CB = sum([C[i] * B[i] for i in range(len(C))])

    # Calculate the magnitudes (lengths) of vectors A, B, and C
    magnitude_A = math.sqrt(sum([A[i] ** 2 for i in range(len(A))]))
    magnitude_B = math.sqrt(sum([B[i] ** 2 for i in range(len(B))]))
    magnitude_C = math.sqrt(sum([C[i] ** 2 for i in range(len(C))]))

    # Calculate the cosines of the angles between A and B, and C and B
    if (magnitude_A * magnitude_B) == 0 or (magnitude_C * magnitude_B) == 0:
        print(white)

    cos_theta_AB = dot_product_AB / (magnitude_A * magnitude_B)
    cos_theta_CB = dot_product_CB / (magnitude_C * magnitude_B)

    # Calculate the angles in radians
    theta_radians_AB = math.acos(cos_theta_AB)
    theta_radians_CB = math.acos(cos_theta_CB)

    # Convert the angles to degrees
    theta_degrees_AB = math.degrees(theta_radians_AB)
    theta_degrees_CB = math.degrees(theta_radians_CB)

    # Compare the angles
    if theta_degrees_AB >= theta_degrees_CB:
        return True
    else:
        return False


########## Count all white points (all centerline in map) ##########


def iter_all_white_points(img):

    target_color = (255, 255, 255)
    indices = np.where(np.all(img == target_color, axis=-1))
    white_cnt = list(zip(indices[0], indices[1]))

    # w, h, _ = img.shape
    # for i in range(w):
    #     for j in range(h):
    #         pt = img[i][j]

    #         if (pt == np.array([255, 255, 255])).all():
    #             img[i][j] = np.array([0, 0, 255])

    return white_cnt

########## Find Path ##########


def find_path(start, end, white_cnt):

    cache = queue.Queue()
    cache.put([start])

    final_path = []

    while not cache.empty():

        parent_path = cache.get()
        print(parent_path)
        candidates = []

        for white in white_cnt:

            last_point = parent_path[-1]
            # print(parent_path)

            new_x = last_point[0] - white[0]
            new_y = last_point[1] - white[1]
            dis = (new_x**2 + new_y**2)**0.5

            # if white point have existed in the path
            if white in parent_path:
                continue

            curr_dis2end = ((last_point[0] - end[0])
                            ** 2 + (last_point[1] - end[1])**2)**0.5
            next_dis2end = ((white[0] - end[0])**2 +
                            (white[1] - end[1])**2)**0.5

            path = parent_path.copy()
            path.append(white)

            if dis < 7 and next_dis2end < 2:
                print("Get!!!")
                final_path.append(path)
                cache = queue.Queue()
                candidates = []
                break

            if dis < 7 and next_dis2end < curr_dis2end:
                candidates.append(path)
                # cache.put(new_path)

        # select one path from all candidate paths, then add to the cache
        # print("="*20)
        # for p in candidates:
        #     print(p)
        # print("="*20)

        if len(candidates) > 0:
            win_path = candidates[0]
            for p in candidates[1:]:
                tp = p[-1]
                cur_s2tp = ((tp[0] - start[0])**2 + (tp[1] - start[1])**2)**0.5
                cur_tp2e = ((tp[0] - end[0])**2 + (tp[1] - end[1])**2)**0.5
                cur_total = cur_s2tp + cur_tp2e

                tp2 = win_path[-1]
                pre_s2tp = ((tp2[0] - start[0])**2 +
                            (tp2[1] - start[1])**2)**0.5
                pre_tp2e = ((tp2[0] - end[0])**2 + (tp2[1] - end[1])**2)**0.5
                pre_total = pre_s2tp + pre_tp2e

                if cur_total < pre_total:
                    win_path = p

            cache.put(win_path)

    for i, p in enumerate(final_path):
        print(p)

    return final_path[0]


def new_find_path(start, end, white_cnt):

    cache = queue.Queue()
    cache.put([start])

    final_path = []

    while not cache.empty():

        parent_path = cache.get()
        candidates = []
        # print(parent_path)

        for white in white_cnt:

            last_point = parent_path[-1]

            new_x = last_point[0] - white[0]
            new_y = last_point[1] - white[1]
            dis = (new_x**2 + new_y**2)**0.5

            # if white point have existed in the path
            if white in parent_path:
                continue

            curr_dis2end = ((last_point[0] - end[0])
                            ** 2 + (last_point[1] - end[1])**2)**0.5
            next_dis2end = ((white[0] - end[0])**2 +
                            (white[1] - end[1])**2)**0.5

            path = parent_path.copy()
            path.append(white)

            if dis < 7 and next_dis2end < 5:  # town 4: 5
                print("Get!!!")
                final_path.append(path)
                cache = queue.Queue()
                candidates = []
                break

            if dis < 7 and next_dis2end < curr_dis2end:
                candidates.append(path)
                # cache.put(new_path)

        # select one path from all candidate paths, then add to the cache
        # print("="*20)
        # for p in candidates:
        #     print(p)
        # print("="*20)

        # if len(candidates) > 0:
        #     win_path = candidates[0].copy()
        #     for p in candidates[1:]:
        #         tp = p[-1]
        #         cur_s2tp = ((tp[0] - start[0])**2 + (tp[1] - start[1])**2)**0.5
        #         cur_tp2e = ((tp[0] - end[0])**2 + (tp[1] - end[1])**2)**0.5
        #         cur_total = cur_s2tp + cur_tp2e

        #         tp2 = win_path[-1]
        #         pre_s2tp = ((tp2[0] - start[0])**2 +
        #                     (tp2[1] - start[1])**2)**0.5
        #         pre_tp2e = ((tp2[0] - end[0])**2 + (tp2[1] - end[1])**2)**0.5
        #         pre_total = pre_s2tp + pre_tp2e

        #         if cur_total < pre_total:
        #             win_path = p

        #     cache.put(win_path)

        if len(parent_path) == 1:
            cache.put(candidates[0])

        elif len(candidates) > 0:

            prev_pt = parent_path[-1]
            prev_prev_pt = parent_path[-2]

            # Straight
            if prev_pt[0] - prev_prev_pt[0] == 0 or prev_pt[1] - prev_prev_pt[1] == 0:
                win_path = candidates[0].copy()
                for p in candidates[1:]:
                    tp = p[-1]
                    cur_s2tp = ((tp[0] - start[0])**2 +
                                (tp[1] - start[1])**2)**0.5
                    cur_tp2e = ((tp[0] - end[0])**2 + (tp[1] - end[1])**2)**0.5
                    cur_total = cur_s2tp + cur_tp2e

                    tp2 = win_path[-1]
                    pre_s2tp = ((tp2[0] - start[0])**2 +
                                (tp2[1] - start[1])**2)**0.5
                    pre_tp2e = ((tp2[0] - end[0])**2 +
                                (tp2[1] - end[1])**2)**0.5
                    pre_total = pre_s2tp + pre_tp2e

                    if cur_total < pre_total:
                        win_path = p

                cache.put(win_path)
            # Turn
            else:
                pre_m = (prev_pt[1] - prev_prev_pt[1]) / \
                    (prev_pt[0] - prev_prev_pt[0])
                win_path = candidates[0].copy()
                for p in candidates[1:]:
                    tp = p[-1]
                    cur_m = (p[-2][1] - tp[1]) / (p[-2][0] - tp[0])
                    cur_diff = cur_m - pre_m

                    tp2 = win_path[-1]
                    win_m = (p[-2][1] - tp2[1]) / (p[-2][0] - tp2[0])
                    win_diff = win_m - pre_m

                    if abs(cur_diff) < abs(win_diff):
                        win_path = p

                cache.put(win_path)

    for i, p in enumerate(final_path):
        print(p)

    return final_path[0]


########## Draw waypoint ##########


def draw_waypoint(img, start, end, final_path):
    draw_img = img.copy()
    for x, y in final_path:
        draw_img[x][y] = np.array([255, 255, 0])

    draw_img[start[0]][[start[1]]] = np.array([255, 0, 255])
    draw_img[end[0]][[end[1]]] = np.array([255, 0, 255])

    return draw_img

########## rdp algorithm ##########


def rdp_algorithm(draw_img, final_path):

    width = 36

    rdp_img = draw_img.copy()
    shortened_route = rdp(final_path, epsilon=0.5)  # 0.75

    all_segments = []

    for i in range(len(shortened_route)-1):
        # Calculate half of the width
        half_width = width // 2

        middle_down_point = shortened_route[i]
        middle_top_point = shortened_route[i+1]
        # Calculate the angle between the two points

        angle = math.atan2(
            middle_top_point[1] - middle_down_point[1], middle_top_point[0] - middle_down_point[0])
        angle = math.pi - angle

        # Calculate the coordinates of the other three corners of the rectangle
        # Assuming the middle down point is the origin
        top_left = [int(middle_down_point[1] - half_width * math.cos(angle)),
                    int(middle_down_point[0] - half_width * math.sin(angle))]
        top_right = [int(middle_down_point[1] + half_width * math.cos(angle)),
                     int(middle_down_point[0] + half_width * math.sin(angle))]
        bottom_right = [int(middle_top_point[1] + half_width * math.cos(angle)),
                        int(middle_top_point[0] + half_width * math.sin(angle))]
        bottom_left = [int(middle_top_point[1] - half_width * math.cos(angle)),
                       int(middle_top_point[0] - half_width * math.sin(angle))]

        # print("="*20)
        # print(middle_down_point)
        # print(middle_top_point)
        # print(top_left)
        # print(top_right)
        # print(bottom_left)
        # print(bottom_right)
        # print(angle)
        # print(math.cos(angle))
        # print(math.sin(angle))

        pts = np.array([top_left, top_right, bottom_right, bottom_left])
        all_segments.append(pts)

        pts = pts.reshape((-1, 1, 2))
        rdp_img = cv2.polylines(rdp_img, [pts], True, (255, 255, 0))

    return rdp_img, all_segments


########## main ##########
town = 1
# 1, 4

if town == 4:
    img = cv2.imread("Town04.png", 1)
    all_whites_pos = iter_all_white_points(img)
    start = (596, 1067)
    end = (447, 1261)

elif town == 1:
    img = cv2.imread("Town01.png", 1)
    all_whites_pos = iter_all_white_points(img)
    # select start point and goal
    start = [3653, 2559][::-1]
    end = [3948, 2187][::-1]

    start = (2559, 3653)
    end = (2187, 3948)

new_img = img.copy()
new_img[start[0]][start[1]] = np.array([0, 0, 255])
new_img[end[0]][end[1]] = np.array([0, 0, 255])

cv2.circle(new_img, start, radius=5, color=(255, 0, 0), thickness=3)
cv2.circle(new_img, end, radius=5, color=(255, 0, 0), thickness=3)


print("Finding Path...")
# final_path = find_path(start, end, all_whites_pos)
final_path = new_find_path(start, end, all_whites_pos)

# cv2.imwrite("se.png", new_img)

print("Drawing Waypoint...")
draw_img = draw_waypoint(img, start, end, final_path)

print("Executing RDP algorithm to get route segment...")
rdp_img, _ = rdp_algorithm(draw_img, final_path)

# cv2.imwrite(f"Town{town}_route_result.png", rdp_img)
cv2.imwrite(f"Town{town}_route_{start}_{end}.png", rdp_img)
