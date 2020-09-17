import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt


img = np.zeros([300,300],dtype='uint8')*255
img[:,:]
window_name = "Canvas"
cv2.namedWindow(window_name)

drawing = False # true if mouse is pressed
ix,iy = -1,-1

def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img,(x,y),5,(255,255,255),10)

    elif event == cv2.EVENT_LBUTTONUP:
            drawing = False



cv2.setMouseCallback(window_name,draw_circle) #Shape is sub method called in setMouseCallback method

m_new = tf.keras.models.load_model('model_digit.h5')
        
while True:
    cv2.imshow(window_name,img)
    key = cv2.waitKey(1)
    if key == ord('q'):     #ord - ordinal function, returns unicode value
        break
    elif key == ord('c'):
        img[:,:]=0
    elif key == ord('p'):
        image_test_resize = cv2.resize(img,(28,28)).reshape(1,28,28)
        detect_digit = m_new.predict_classes(image_test_resize)
        print("The pridected number is: ",detect_digit)
cv2.destroyAllWindows()

