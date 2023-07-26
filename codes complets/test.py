import pyfirmata
board = pyfirmata.Arduino('COM7') # Windows
import time
import numpy as np
"""
with open('data.npy', 'wb') as f:
    np.save(f, np.array(1))
    np.save(f, np.array([1, 3]))
with open('data.npy', 'rb') as f:
    a = np.load(f)
    b=np.load(f)
print(a,b)
"""
"""
engine = board.digital[3]
engine.mode=pyfirmata.PWM
engine.write(1)

engine.write(1)
time.sleep(2)
engine.write(0)

in1= board.digital[13] 
in2= board.digital[12]
in3= board.digital[11]
in4= board.digital[10]

in1.write(1)
in2.write(1)
in3.write(1)
in4.write(1)

in1.write(0)
time.sleep(2)
in1.write(1)

in2.write(0)
time.sleep(2)
in2.write(1)

in3.write(0)
time.sleep(2)
in3.write(1)

in4.write(0)
time.sleep(2)
in4.write(1)
"""

in3= board.digital[10] #arriere
in4= board.digital[11] #avant
#in3.write(1)
#time.sleep(0.5)
#in3.write(0)

#in4.write(1)
#time.sleep(1)
#in4.write(0)    
#in3.mode=pyfirmata.PWM
#in3.write(1)

def move(sens,temps):
    """
    0=arriere
    1= avant
    temps = 1/tau
    """
    if sens == 1:
        in4.write(1)
        time.sleep(temps)
        in4.write(0)
    else:
        in3.write(1)
        time.sleep(temps)
        in3.write(0)

def create_wave(temps,amplitude):
    t=time.time()
    while time.time()<t+temps:
        move(0,amplitude)
        move(1,amplitude)
#time.sleep(10)
move(1,0.55)
#move(1,2)
#create_wave(3,0.5)
#move(1,0.1)
#in2=board.digital[12]
#in2.write(1)
#time.sleep(0.1)
#in2.write(0)

#move(1,0.5)
#temps=0.25
#move(1,0.5)
#for i in range(7):
#    move(0,temps)
#    move(1,temps)


"""
#in3.mode=pyfirmata.PWM
#in3.write(1)
#in4.write(1)
#time.sleep(2)

#in4.write(0)"""
"""
"""
"""import time
import cv2 as cv
cv.namedWindow("preview")
vc = cv.VideoCapture(0)
t=time.perf_counter()
while time.perf_counter()-t<20:
    _, frame = vc.read()
    cv.imshow("preview",frame) 
    k = cv.waitKey(0)
print(frame)
#cv.destroyWindow("preview")
#vc.release()
"""

"""import cv2

def countCameras():

    camCount = 0

    while True:
        cam = cv2.VideoCapture(camCount)

        if cam.isOpened():
            print("Found camera", camCount)
            cam.release()
            camCount += 1

        else:
            return camCount

countCameras()"""