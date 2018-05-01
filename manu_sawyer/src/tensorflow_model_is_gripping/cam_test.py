#!/usr/bin/python
import cv2, time

def main():
  print 'Initializing'
  cap = cv2.VideoCapture(0)
  fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
  print 'camera fps =', fps
  start = time.time()
  num_frames = 0
  while True:
    #print 'Reading frame'
    ret, frame = cap.read()
    if not ret:
      print 'Could not read from camera'
      break
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
    if num_frames % 30 == 0:
      fps = num_frames / float(time.time() - start)
      print 'FPS: %.2f' % fps
    print frame.shape
    cv2.imshow('frame', frame)
    num_frames += 1

  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()

