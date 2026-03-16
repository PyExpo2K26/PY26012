import cv2
import numpy as np

def detect_copymove(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0, 0.0


        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(img, None)
        
        if descriptors is None or len(descriptors) < 2:
            return 0, 0.0
            
  
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(descriptors, descriptors, k=2)
        good_matches = []
        for m, n in matches:
  
            if m.distance < 0.75 * n.distance:
  
                pt1 = keypoints[m.queryIdx].pt
                pt2 = keypoints[n.queryIdx].pt
                
                spatial_dist = np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)
                
   
                if spatial_dist > 20: 
                    good_matches.append(m)
        
        count = len(good_matches)
        
     
        score = min(count / 50.0, 1.0) 
        
        return count, float(score)

    except Exception as e:
        print(f"Copy-Move Error: {e}")
        return 0, 0.0

        
