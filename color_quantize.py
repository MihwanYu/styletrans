from xml.sax.handler import DTDHandler
import cv2
import numpy as np
import matplotlib.pyplot as plt
def quantimage(image,k):
    i = np.float32(image).reshape(-1,3)
    condition = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,20,1.0)
    ret,label,center = cv2.kmeans(i, k , None, condition,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    final_img = center[label.flatten()]
    final_img = final_img.reshape(image.shape)
    return final_img


def findContourOf(src): #지금 안쓰는거
    dst = src.copy()
    ''''''
    whiteboard = np.ones(src.shape, dtype=np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    #hsv = cv2.cvtColor(src, cv2.COLOR_RGB2HSV)
    #cv2.imshow("hsv", hsv)
    # ret, binary = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV) 지금까지 중 최선
    # ret, binary = cv2.threshold(gray,100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv2.imshow("binary", binary)

    # morp = cv2.dilate(binary, kernel, anchor=(-1,-1))
    #print('hsv: ', hsv[:,:,2].shape)
    print('gray: ',gray.shape)
    morp = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=2) #open/close 두개-> open 이거다
    morp = cv2.dilate(morp, kernel, anchor=(-1,-1)) #모폴로지 팽창을 한번 했을 때 결괏값 bound가 더 얇아짐
    
    # image = cv2.bitwise_not(morp) #여기서 binary 결과랑 img 결과랑 달라지게 만드는거같음
    image = morp.copy()
    cv2.imshow("img", image)
    
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    cv2.drawContours(dst, contours, -1, (0,0,255), 1)
    cv2.drawContours(whiteboard, contours, -1, (0,0,0), 1)
    cv2.imshow("whiteboard", whiteboard)
    whiteboard = cv2.cvtColor(whiteboard, cv2.COLOR_RGB2GRAY)
    return dst, whiteboard


def contOfBinary(gray, kernel, th1):
    # ret, binary = cv2.threshold(gray,th1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) #이거 잘만 되다가 갑자기 왜안되지(5.4)
    ret, binary = cv2.threshold(gray,th1, 255, cv2.THRESH_BINARY)
    binary = cv2.bilateralFilter(binary,10,75,75)
    morp = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2) #open/close 두개-> open 이거다
    morp = cv2.dilate(morp, kernel, anchor=(-1,-1)) #모폴로지 팽창을 한번 했을 때 결괏값 bound가 더 얇아짐
    contours, hierarchy = cv2.findContours(morp, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    # cv2.imshow(str(th1), binary)
    return contours

def findContourOf2(src):
    dst = src.copy()
    whiteboard = np.ones(src.shape, dtype=np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    # cv2.imshow("gray", gray)
    '''
    cv2.imshow("gray", gray)
    # ret, binary = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV) 지금까지 중 최선
    ret, binary = cv2.threshold(gray,100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow("binary", binary)
    morp = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2) #open/close 두개-> open 이거다
    morp = cv2.dilate(morp, kernel, anchor=(-1,-1)) #모폴로지 팽창을 한번 했을 때 결괏값 bound가 더 얇아짐

    # image = cv2.bitwise_not(morp) #여기서 binary 결과랑 img 결과랑 달라지게 만드는거같음
    image = morp.copy()
    cv2.imshow("img", image)
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    '''

    
    cont127 = contOfBinary(gray, kernel, 127)
    cont100 = contOfBinary(gray, kernel, 100)
    cont150 = contOfBinary(gray, kernel, 150)
    cont175 = contOfBinary(gray, kernel, 175)
    cont183 = contOfBinary(gray, kernel, 183)
    
    
    # cv2.drawContours(dst, contours, -1, (0,0,255), 1)
    cv2.drawContours(whiteboard, cont127, -1, (0,0,0), 1)
    cv2.drawContours(whiteboard, cont100, -1, (0,0,0), 1)
    cv2.drawContours(whiteboard, cont150, -1, (0,0,0), 1)
    cv2.drawContours(whiteboard, cont183, -1, (0,0,0), 1)
    cv2.drawContours(whiteboard, cont175, -1, (0,0,0), 1)
    
    # cv2.imshow("whiteboard", whiteboard)
    whiteboard = cv2.cvtColor(whiteboard, cv2.COLOR_RGB2GRAY)
    return dst,whiteboard

def saturate_contrast2(p, num):
    pic = p.copy()
    pic = pic.astype('int32')
    pic = np.clip(pic+(pic-128)*num, 0, 255)
    pic = pic.astype('uint8')
    return pic


def originEdge(src):
    cv2.imshow("EDGE_src", src)
    gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray,5,75,75)
    cv2.imshow("EDGE_bilateralFilter", gray)
    # gray = cv2.GaussianBlur(gray, (3,3), 0)
    # cv2.imshow("EDGE_GaussianBlur", gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    gray = cv2.dilate(gray, kernel, anchor=(-1,-1))
    gray = cv2.erode(gray, kernel, iterations=2)

    cv2.imshow("EDGE_grayscale", gray) #<--여기서 픽셀 깨짐이 많이 발생
    '''
    #canny edge 이용
    dst = cv2.Canny(src, 80,200)
    
    #가우시안 - laplacian 시용
    # mask1 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    # mask3 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    # gaussian = cv2.GaussianBlur(gray, (3, 3), 0)
    # LoG = cv2.filter2D(gaussian, -1, mask1)
    '''
    #라플라시안
    dst = cv2.Laplacian(gray, -1)
    cv2.imshow("Laplacian", dst)
    '''
    #scharr 사용
    # dst_x = cv2.Scharr(gray, -1, 1, 0)
    # dst_y = cv2.Scharr(gray, -1, 0, 1)
    '''

    '''
    #canny edge 사용
    dst = cv2.Canny(dst, 100, 255)
    cv2.imshow("Canny after Laplacian", dst)
    '''
    dst=cv2.bitwise_not(dst)

    '''
    sharpening_mask1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpening_mask2 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpening_out1 = cv2.filter2D(dst, -1, sharpening_mask1)
    sharpening_out1 = cv2.filter2D(sharpening_out1, -1, sharpening_mask1)
    sharpening_out1 = cv2.bilateralFilter(sharpening_out1,5,75,75)
    # sharpening_out2 = cv2.filter2D(dst, -1, sharpening_mask2)

    cv2.imshow("EDGE_dst", dst)
    cv2.imshow("EDGE_sharpening_out1", sharpening_out1)
    # cv2.imshow("EDGE_sharpening_out2", sharpening_out2)
    '''

    '''
    kernel = np.ones((5,5), np.uint8)
    dst = cv2.erode(dst, kernel, iterations=1)
    dst = cv2.dilate(dst,kernel, iterations=1)
    kernel = np.ones((3,3), np.uint8)
    dst = cv2.erode(dst, kernel, iterations=2)
    dst = cv2.dilate(dst,kernel, iterations=2)
    '''
    # dst = saturate_contrast2(dst,2)
    # dst = saturate_contrast2(dst,3)
    # alpha = 1.0
    # dst = np.clip((1+alpha)*dst - 254*alpha, 0, 255).astype(np.uint8)
    cv2.imshow("before binarization", dst)
    ret, dst = cv2.threshold(dst,254, 255, cv2.THRESH_BINARY) #254, 200 등 뭐 넣는지에 따라 boundary 결과가 달라짐
    cv2.imshow("after binarization", dst)
    # dst = cv2.medianBlur(dst, 5)
    # dst = cv2.blur(dst, (3,3))
    # ret, dst = cv2.threshold(dst,181, 255, cv2.THRESH_BINARY)
    # dst = cv2.fastNlMeansDenoising(dst, 150, 1, 11)
    cv2.denoise_TVL1(dst, dst, 10.0, 5)
    
    cv2.imshow("after denoising", dst)
    #생각해보니까 그냥 이진화를 하면 되었던 것이었음
    return dst


imgpath = 'base18_style4_MST.jpg'
image = cv2.imread('images/dfr_afterstyle/'+imgpath)
imageorigin = cv2.imread('images/content/'+imgpath.split('_')[0]+'.jpg')

print('styled image: ','images/dfr_afterstyle/'+imgpath)
print('origin image: ','images/content/'+imgpath.split('_')[0]+'.jpg')
# width 을 600 으로 할때
ratio = 600.0 / image.shape[1]
dim = (600, int(image.shape[0] * ratio))

imageorigin = cv2.resize(imageorigin, dim)
image = cv2.resize(image, dim)
# image = cv2.bilateralFilter(image,10,75,75)

cv2.imshow("original", image)


image3 = quantimage(image,8)
# image3 = cv2.bilateralFilter(image3,10,50,50) #필터 적용 했을 때랑 안했을 때랑 결과도 차이가 큼, 지금은 안하는게 나음
cv2.imshow(imgpath.split('_')[0]+'col35', image3)

dst, whiteboard = findContourOf2(image3)

edge = originEdge(image3)
# cv2.imshow('edge', edge)

print('img size: ', whiteboard.shape, edge.shape)
result = cv2.bitwise_and(whiteboard, edge)
# cv2.imshow('result', result)

'''
kernel = np.ones((3,3), np.uint8)
result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=2)
result = cv2.dilate(result, kernel, iterations=3)
# cv2.imshow('result after dilation', result)
'''

cv2.waitKey(0)
cv2.destroyAllWindows()

