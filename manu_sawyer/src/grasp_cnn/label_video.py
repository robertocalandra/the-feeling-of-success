#!/usr/bin/python
# don't use anaconda python, since its opencv version
# doesn't work with this video

import sys, os, cv2, time

#fps = 29.97
#fps = 250.
buf = 0.25

def main():
  print "Mark positives with a space and negatives with 'n'. z = undo"
  in_vid_file = sys.argv[1]
  print 'Loading:', in_vid_file
  cap = cv2.VideoCapture(in_vid_file)
  fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
  print 'fps =', fps

  start_time = 0. if len(sys.argv) <= 2 else float(sys.argv[2])

  press_time = None
  neg_time = None

  out_fname = '.'.join(in_vid_file.split('.')[:-1]) + '.txt'
  with open(out_fname, 'a') as out_file:
    idx = 0
    while True:
      ret, frame = cap.read()
      if not ret:
        print 'Could not read from camera'
        print 'ret =', ret
        break
      k = cv2.waitKey(1)
      t = idx / float(fps)
      idx += 1

      if t < start_time:
        if idx % 100 == 0:
          print 'Skipping', t, 'start =', start_time
        continue

      is_pressing = press_time is not None and press_time < t < press_time + buf
      is_neg = neg_time is not None and neg_time < t < neg_time + buf

      sleep_dur = 0. if is_neg else 3*1./fps
      time.sleep(sleep_dur)

      k = k & 0xFF
      if k == ord('q'):
        break
      elif k == ord(' ') or k == ord('p'):
        if press_time is None or t > press_time + buf:
          if press_time is not None:
            out_file.write('p %.4f\n' % press_time)
            out_file.flush()
        press_time = t
        print 'Press:', press_time
      elif k == ord('n'):
        if neg_time is None or t > neg_time + buf:
          if neg_time is not None:
            out_file.write('n %.4f\n' % neg_time)
            out_file.flush()
        neg_time = t
        print 'Neg:', neg_time
      elif k == ord('z'):
        print 'Undoing last press and neg'
        press_time = None
        neg_time = None
        
      if is_pressing:
        color = (0, 255, 0)
      elif is_neg:
        color = (0, 0, 255)
      else:
        color = (255, 0, 0)

      frame = cv2.resize(frame, (frame.shape[1]/2, frame.shape[0]/2))
      secs = t
      mins = int(secs / 60.)
      secs = secs - mins*60
      text = '%02d:%02d' % (mins, secs)
      cv2.putText(frame, text, (0, 50), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1, color)

      cv2.imshow('frame', frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
  main()




