import os
import cv2

sample = cv2.imread("SOCOFing/Altered/Altered-Hard/150__M_Right_index_finger_Obl.BMP")
best_score=0
filename=None
image=None
kp1,kp2,mp=None,None,None


counter=0
for file in [file for file in os.listdir("SOCOFing/Real")][:1000]:
   # if counter %10==0:
       # print(counter)
      #  print(file)
   # counter+=1

    fingerprint_image=cv2.imread("SOCOFing/Real/"+file)
    sift=cv2.SIFT_create()
    keyoints_1,descriptor_1=sift.detectAndCompute(sample,None)
    keyoints_2,descriptor_2 = sift.detectAndCompute(fingerprint_image,None)
    matches=cv2.FlannBasedMatcher({'algorithm':1,'trees':2},{}).knnMatch(descriptor_1,descriptor_2,k=2)


    match_point=[]

    for p,q in matches:
        if p.distance<0.1*q.distance:
            match_point.append(p)


    keypoints=0
    if len(keyoints_1)< len(keyoints_2):
            keypoints=len(keyoints_1)

    else:
            keypoints=len(keyoints_2)
    if len(match_point)/keypoints *100>best_score:
        best_score=len(match_point)/ keypoints*100
        filename=file
        image=fingerprint_image
        kp1,kp2,mp=keyoints_1,keyoints_2,match_point


print("Best Match :"+filename)
print("score :"+str(best_score))
result=cv2.drawMatches(sample,kp1,image,kp2,mp,None)
cv2.resize(result,None,fx=100,fy=100)
cv2.imshow("Result",result)
cv2.waitKey(0)
cv2.destroyWindow()
