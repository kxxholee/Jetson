from ultralytics import YOLO
import cv2

model = YOLO("yolov8s.pt")
results = model("./test.png")
plots = results[0].plot()
cv2.imshow("plot", plots)
cv2.waitKey(0)
cv2.destroyAllWindows()
