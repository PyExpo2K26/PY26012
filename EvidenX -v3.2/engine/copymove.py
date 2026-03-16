import cv2
import numpy as np

def detect_copymove(image_path):
    """
    Detects potential copy-move forgery using ORB feature matching.
    Returns:
        matches_count: Number of similar keypoints found (higher is suspicious).
        score: Normalized risk score (0-1).
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0, 0.0

        # ORB Detector
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(img, None)
        
        if descriptors is None or len(descriptors) < 2:
            return 0, 0.0
            
        # Match features against themselves
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(descriptors, descriptors, k=2)
        
        good_matches = []
        for m, n in matches:
            # Distance check (Lowe's ratio test logic, but strictly for self-similarity)
            # We want to find points that are very similar but not the excat same point (distance > 0)
            if m.distance < 0.75 * n.distance:
                # Ensure they are not the same physical point (spatial distance check)
                pt1 = keypoints[m.queryIdx].pt
                pt2 = keypoints[n.queryIdx].pt
                
                spatial_dist = np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)
                
                # If they are spatially separated but visually similar, it's a candidate for copy-move
                if spatial_dist > 20: 
                    good_matches.append(m)
        
        count = len(good_matches)
        
        # Heuristic score: if many matches found, high risk
        score = min(count / 50.0, 1.0) # Cap at 1.0
        
        return count, float(score)

    except Exception as e:
        print(f"Copy-Move Error: {e}")
        return 0, 0.0
