import cv2
import numpy as np
import base64

def detect_copymove(image_path):
    """
    Detects copy-move forgery using SIFT feature matching (more accurate than ORB).
    Also generates a visual heatmap overlay showing matched/suspicious regions.

    Returns:
        matches_count : int   — number of suspicious match pairs found
        score         : float — normalized risk score (0–1)
        heatmap_b64   : str   — base64 PNG heatmap with matched region dots drawn
    """
    try:
        img_color = cv2.imread(image_path)
        if img_color is None:
            return 0, 0.0, ""

        img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        # ── Use SIFT (better descriptor quality than ORB) ──────────────────
        # Fall back to ORB if SIFT is unavailable (older OpenCV builds)
        try:
            detector = cv2.SIFT_create(nfeatures=1500)
            norm_type = cv2.NORM_L2
        except AttributeError:
            detector = cv2.ORB_create(nfeatures=1500)
            norm_type = cv2.NORM_HAMMING

        keypoints, descriptors = detector.detectAndCompute(img, None)

        if descriptors is None or len(descriptors) < 4:
            return 0, 0.0, ""

        # ── k-NN self-matching ─────────────────────────────────────────────
        bf = cv2.BFMatcher(norm_type, crossCheck=False)
        matches = bf.knnMatch(descriptors, descriptors, k=3)

        good_matches = []
        for match_list in matches:
            for m in match_list[1:]:  # skip self-match (index 0, distance ~0)
                pt1 = keypoints[m.queryIdx].pt
                pt2 = keypoints[m.trainIdx].pt
                spatial_dist = np.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])
                if spatial_dist > 20:
                    good_matches.append((pt1, pt2))

        count = len(good_matches)

        # ── Score ──────────────────────────────────────────────────────────
        score = float(min(count / 80.0, 1.0))

        # ── Heatmap visualization ──────────────────────────────────────────
        heatmap_b64 = ""
        try:
            overlay = img_color.copy()
            heat = np.zeros(img_color.shape[:2], dtype=np.float32)
            h, w = heat.shape

            for (pt1, pt2) in good_matches:
                x1, y1 = int(pt1[0]), int(pt1[1])
                x2, y2 = int(pt2[0]), int(pt2[1])
                # Draw connecting line (subtle)
                cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
                # Accumulate heat at both ends
                for xc, yc in [(x1, y1), (x2, y2)]:
                    cv2.circle(heat, (xc, yc), 12, 1.0, -1)

            # Smooth and colourise heat
            heat = cv2.GaussianBlur(heat, (31, 31), 0)
            heat_norm = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
            blended = cv2.addWeighted(overlay, 0.6, heat_color, 0.4, 0)

            _, buf = cv2.imencode('.png', blended)
            heatmap_b64 = base64.b64encode(buf).decode()
        except Exception as viz_err:
            print(f"Copy-Move viz error: {viz_err}")

        return count, score, heatmap_b64

    except Exception as e:
        print(f"Copy-Move Error: {e}")
        return 0, 0.0, "
