import time

from openpose import pyopenpose as op
import cv2 as cv

fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('dataset/BEHAVE/output.avi',fourcc, 20.0, (640,480))

cap = cv.VideoCapture("dataset/BEHAVE/fight_margaret_1_24_01_2007.wmv")
cap.set(cv.CAP_PROP_POS_FRAMES, 60683)
fgbg = cv.createBackgroundSubtractorMOG2()

params = dict()
params["model_folder"] = "models/openpose/"

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()
datum = op.Datum()

t = time.clock()
# Process Image
while True:
    print('{} fps'.format(1 / (time.clock() - t)))
    t = time.clock()

    _, frame = cap.read()
    fgmask = fgbg.apply(frame)
    frame = cv.bitwise_and(frame, frame, mask = fgmask)

    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])

    # Display Image
    print("Body keypoints: \n" + str(datum.poseKeypoints))
    cv.imshow("OpenPose", datum.cvOutputData)
    out.write(datum.cvOutputData)
    if cv.waitKey(1) == 27:
        break

out.release()
