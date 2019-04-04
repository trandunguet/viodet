import sys
import cv2

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Invalid argument!")
        print("Usage: {} VIDEO START_POS LENGTH OUTPUT".format(sys.argv[0]))
        exit()
    
    cap = cv2.VideoCapture(sys.argv[1])
    start = float(sys.argv[2])
    length = int(sys.argv[3])
    output_path = sys.argv[4]

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (640,480))

    cap.set(cv2.CAP_PROP_POS_MSEC, start)
    for i in range(length):
        ret, frame = cap.read()
        if ret:
            out.write(frame)

            cv2.imshow('frame', frame)
            cv2.waitKey(30)
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
