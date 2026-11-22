"""Download the MediaPipe Hand Landmarker model (about 7.5 MB) into models/.

Source: Google's official MediaPipe model bucket, version 1, float16.
Usage:  python scripts\\download_model.py
"""
import os
import sys
import urllib.request

URL = ("https://storage.googleapis.com/mediapipe-models/"
       "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
DEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "hand_landmarker.task")


def main() -> int:
    if os.path.isfile(DEST) and os.path.getsize(DEST) > 1_000_000:
        print(f"Model already present: {DEST}")
        return 0
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    print(f"Downloading {URL}\n -> {DEST}")
    tmp = DEST + ".part"
    urllib.request.urlretrieve(URL, tmp)
    os.replace(tmp, DEST)
    print(f"Done ({os.path.getsize(DEST) / 1e6:.1f} MB).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
