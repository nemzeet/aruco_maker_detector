import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# ----- 지침
# 다양한 각도로 체스보드 16장 찍기 (체스보드가 평평하게 놓여 있다는 가정 하에)
# -> 모니터에 띄우두고 카메라로 찍으면 훨씬 편함
# -> 휴대폰으로 찍을 때 광량, 포커싱, 조리개 값 (보통 작게) 고정, ISO 값 높이기 -> 이값들 고정 시키고 촬영해야함
# 영상 해상도 1024 * 768 HD
# 격자의 크기를 자로 정확하게 제기 (mm기준)


def plot_img(rows, cols, index, img, title, axis='on'):
    ax = plt.subplot(rows, cols, index)
    if(len(img.shape)==3):
        ax_img = plt.imshow(img[...,::-1])
    else:
        ax_img = plt.imshow(img, cmap='gray')
    
    plt.axis(axis)
    if(title != None): plt.title(title)
    return  ax_img, ax

def display_untilKey(Pimgs, Titles, file_out = False):
    for img, title in zip(Pimgs, Titles):
        cv.imshow(title, img)
        if file_out == True:
            cv.imwrite(title + '.jpeg', img)
    cv.waitKey(0)

if __name__ == '__main__':
    
    fr = cv.FileStorage('./camera_parameters.txt', cv.FileStorage_READ)
    intrisic_mtx = fr.getNode('camera intrinsic matrix').mat()
    dist_coefs = fr.getNode('distortion coefficient').mat()
    newcameramtx = fr.getNode('camera new intrinsic matrix').mat()
    print('camera intrinsic matrix:\n', intrisic_mtx)
    print('distortion coefficients: ', dist_coefs.ravel())
    print('camera new intrinsic matrix: \n', newcameramtx)
    fr.release()
    
    # pose test img 12
    pattern_size = (9,6)
    square_size = 22 # mm 기준 
    
    pattern_points = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    
    pattern_points *= square_size
    
    import glob
    image_names = glob.glob('./my_calib_resize_img/*.jpeg')
    # image_names[11]
    img = cv.imread(image_names[5])
    img_undist_test = cv.undistort(img, intrisic_mtx, dist_coefs, None, newcameramtx)
    img_undist_test_gray = cv.cvtColor(img_undist_test, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(img_undist_test_gray, pattern_size)
    
    if ret:
        criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT, 30, 0.1) # 0.001
        corners2 = cv.cornerSubPix(img_undist_test_gray, corners, (5,5), (-1,-1), criteria)
        ret, rvecs, tvecs = cv.solvePnP(pattern_points, corners2, newcameramtx, None)
        print('rvecs:\n', rvecs)
        print('tvecs:\n', tvecs)
        
        axis = np.float32([[square_size, 0, 0], [0, square_size, 0], [0, 0, square_size]]).reshape(-1, 3)
        # project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, newcameramtx, dist_coefs)
        
        axis_center = tuple(corners2[0].ravel().astype(int))
        cv.line(img_undist_test, axis_center, tuple(imgpts[0].ravel().astype(int)), (0,0,255), 5)
        cv.line(img_undist_test, axis_center, tuple(imgpts[1].ravel().astype(int)), (0,255,0), 5)
        cv.line(img_undist_test, axis_center, tuple(imgpts[2].ravel().astype(int)), (255,0,0), 5)
        display_untilKey([img_undist_test], ['pose'])
        
        
        
    