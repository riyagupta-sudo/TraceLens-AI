import numpy as np
import traceback
import os

cache_path = r"C:\Users\riya2\OneDrive\Desktop\TraceLens AI\backend\app\uploads\features\upload_1782378715.99713_features.npz"
print("File exists:", os.path.exists(cache_path))
if os.path.exists(cache_path):
    try:
        with np.load(cache_path, allow_pickle=True) as data:
            print("Keys:", list(data.keys()))
            orb_kp_data = data["orb_kp"]
            print("Loaded orb_kp_data")
            # Let's try deserialize
            import cv2
            def deserialize_keypoints(serialized):
                kps = []
                if not serialized:
                    return kps
                for item in serialized:
                    kp = cv2.KeyPoint(
                        x=float(item[0]), y=float(item[1]),
                        size=float(item[2]), angle=float(item[3]),
                        response=float(item[4]), octave=int(item[5]),
                        class_id=int(item[6])
                    )
                    kps.append(kp)
                return kps
            orb_kp = deserialize_keypoints(orb_kp_data)
            print("Deserialized orb_kp")
            orb_des = data["orb_des"]
            print("Loaded orb_des, type:", type(orb_des), "shape:", getattr(orb_des, 'shape', None))
            if len(orb_des) == 0:
                print("len(orb_des) == 0 is True")
                orb_des = None
            else:
                print("len(orb_des) == 0 is False")
    except Exception as e:
        traceback.print_exc()
