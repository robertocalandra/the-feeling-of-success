#!/usr/bin/python
import press, cv2, aolib.util as ut, pylab, numpy as np, sys

#drop_rate = 50
drop_rate = None

def main():
  #train_path = sys.argv[1]
  model_path = sys.argv[1]
  #vid_file = '/home/ao/Videos/Webcam/train/2_2017-06-01-170024.mp4')
  if len(sys.argv) > 2:
    vid_file = sys.argv[2]
  else:
    vid_file = 0
  #net = press.NetClf(train_path)
  #net = press.NetClf(train_path)
  net = press.NetClf(model_path)
  #cap = cv2.VideoCapture(0)
  cap = cv2.VideoCapture(vid_file)
  #cap = cv2.VideoCapture('/home/ao/Videos/Webcam/train/1_2017-05-31-232750.mp4')
  #cap = cv2.VideoCapture('/home/ao/Videos/Webcam/train/2017-05-31-232750.mp4')
  num_frames = 0
  cmap = pylab.cm.RdYlGn
  i = 0
  while True:
    ret, frame = cap.read()
    if i == 0:
      im0 = frame
    i += 1
    if drop_rate is not None and (i != 1 and i % drop_rate != 0):
      continue
    if not ret:
      print 'Could not read from camera'
      break
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
    num_frames += 1
    #frame = cv2.imread('/home/ao/Dropbox/gelsight-grasp/results/press-data-v1/ims/00000_00073_00011_0.png')
    #frame = cv2.imread('/home/ao/Dropbox/gelsight-grasp/results/press-data-v1/ims/00000_00007_00002_0.png')
    prob = net.predict(frame, im0)
    color = map(int, 255*np.array(cmap(prob))[:3])
    cv2.putText(frame, ut.f2(prob), (0, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color)
    cv2.imshow('frame', frame)

  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()

