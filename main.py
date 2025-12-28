import cv2
import argparse
import numpy as np
from detector import GateDetector
from crossing import CrossingDetector
from track import Track
from race import RaceFSM
import audio

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["calib", "learn", "race"], required=True)
parser.add_argument("--gates", type=int, default=5)
parser.add_argument("--video", type=str, default=None,
                    help="Optional MP4 file instead of live camera")
args = parser.parse_args()

cap = cv2.VideoCapture(args.video if args.video else 0)

detector = GateDetector(model_path="runs/detect/train4/weights/best.pt", conf=0.65)
crossing = CrossingDetector()

if args.mode == "learn":
    track = Track(args.gates)
elif args.mode == "race":
    track = Track.load()
    race = RaceFSM(track)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.detect(frame)

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        if crossing.check_crossing(d["bbox"]):
            entry_vector = np.array([0, 0, 1])

            if args.mode == "learn":
                track.add_gate(d, entry_vector)
                audio.ok()
                print(f"Learned gate {len(track.gates)}")

                if track.complete():
                    track.save()
                    print("Track saved")
                    exit(0)

            elif args.mode == "race":
                result = race.on_gate(d, entry_vector)
                print(result)
                audio.ok() if result == "OK" else audio.wrong()

    cv2.imshow("fpv-spotter", frame)
    if cv2.waitKey(1) == 27:
        break