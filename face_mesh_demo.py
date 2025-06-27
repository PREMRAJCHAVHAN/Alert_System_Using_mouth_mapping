import cv2
import mediapipe as mp
import winsound
import os
def main():
    cap = cv2.VideoCapture(0)
    face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
    if not cap.isOpened():print("cannot open camera");return
    mouth_open = False
    alert_played = False
    UPPER_LIP = 13
    LOWER_LIP = 14
    while True:
        ret, frame = cap.read()
        if not ret:break
        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        face_count = 0
        if results.multi_face_landmarks:
            face_count = len(results.multi_face_landmarks)
            for lm in results.multi_face_landmarks:
                h, w,_ =frame.shape
                upper_lip = lm.landmark[UPPER_LIP]
                lower_lip = lm.landmark[LOWER_LIP]
                x1, y1 =int(upper_lip.x *w),int(upper_lip.y*h)
                x2, y2 =int(lower_lip.x *w),int(lower_lip.y*h)
                cv2.circle(frame,(x1,y1),3,(255,0,0),-1)
                cv2.circle(frame,(x2,y2),3,(0,255,0),-1)
                mouth_distance = abs(y2-y1)
                cv2.putText(frame,f'mouth:{mouth_distance}',(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
                if mouth_distance>25:
                    mouth_open =True
                else:
                    mouth_open =False
                if mouth_open and not alert_played:
                    alert_path =os.path.join(os.path.dirname(__file__),'alert.wav')
                    winsound.PlaySound(alert_path,winsound.SND_FILENAME| winsound.SND_ASYNC)
                    alert_played = True
                elif not mouth_open:
                    alert_played=False
                for pt in lm.landmark:
                    x, y =int(pt.x*w),int(pt.y*h)
                    cv2.circle(frame,(x, y),2,(0,0,255),-1)
        cv2.putText(frame,f'faces:{face_count}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imshow('Camera',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):break
    cap.release(); cv2.destroyAllWindows()
if __name__=="__main__":main() 

    