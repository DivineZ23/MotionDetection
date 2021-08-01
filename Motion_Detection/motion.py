import cv2, time, pandas
from datetime import datetime

staticframe = None
flag_list=[None,None]
times=[]
data=pandas.DataFrame(columns=["Object Entered","Object Left"])

vid = cv2.VideoCapture(0)

while True:
    check, frame = vid.read()
    flag=0
    curr = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    curr = cv2.GaussianBlur(curr,(21,21),0)
    if staticframe is None:
        staticframe=curr
        continue

    delta=cv2.absdiff(staticframe,curr)
    ret,th_frame=cv2.threshold(delta,45,255,cv2.THRESH_BINARY)
    th_frame=cv2.dilate(th_frame,None,iterations=2)

    (cont,_) = cv2.findContours(th_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for contour in cont:
        if cv2.contourArea(contour) < 10000:
            continue
        flag=1
        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),4)

    flag_list.append(flag)
    if flag_list[-1] == 1 and flag_list[-2] == 0:
        times.append(datetime.now())
    if flag_list[-1] == 0 and flag_list[-2] == 1:
        times.append(datetime.now())


    cv2.imshow("Gray",curr)
    cv2.imshow("delta",delta)
    cv2.imshow("threshold frame",th_frame)
    cv2.imshow("final",frame)

    key=cv2.waitKey(1)
    if key==ord('q'):
        if flag == 1:
            times.append(datetime.now())
        break
for time in range(0,len(times),2):
    data=data.append({"Object Entered": times[time],"Object Left": times[time+1]},ignore_index=True)

data.to_csv("Timestamp.csv")

vid.release()
cv2.destroyAllWindows()
