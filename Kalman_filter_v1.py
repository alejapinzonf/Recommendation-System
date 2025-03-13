from ultralytics import YOLO, solutions
from ultralytics.utils.plotting import Annotator, colors
import numpy as np
import cv2
import pyttsx3
import threading
import time
#from pykalman import KalmanFilter 

engine = pyttsx3.init()
engine_lock = threading.Lock()

#kalman_dict = {}

model = YOLO('/home/aleja/tesis/13112024/runs/segment/train/weights/best.pt')
path = '/home/aleja/tesis/videos/video5.mp4'

class_colors = {
    'bolardo': (255, 165, 0), 'charco': (0, 255, 255), 'ciclista': (0, 255, 0),
    'cicloruta': (100, 0, 0), 'hueco': (0, 255, 128), 'cruce': (128, 128, 128),
    'desnivel': (64, 224, 208), 'motociclista': (0, 0, 255), 'pare': (255, 255, 0),
    'peaton': (0, 0, 128), 'poste': (255, 192, 203), 'vehiculo': (128, 0, 128)
}

alerted_ids = set()
last_alert_time = {}
kalman_f = {}  
#kf = KalmanFilter(initial_state_mean=[0], n_dim_obs=1)
#kf = KalmanFilter(initial_state_mean=[0, 0], n_dim_obs=2)

#def init_kalman():
    #kalman = cv2.KalmanFilter(4, 2)  # 4 estados (x, y, dx, dy), 2 mediciones (x, y)
    #kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    #kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    #kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
    #return kalman

class KalmanFilter:
    def __init__(self, initial_y):
        self.state = np.array([[initial_y], [0.0]])  
        self.covariance = np.eye(2) * 100            
        self.process_noise = np.eye(2) * 0.1         # Q
        self.measurement_noise = np.array([[10]])    # R
        
        self.F = np.array([[1, 1], [0, 1]])  # M transiciOn
        self.H = np.array([[1, 0]])           # M observaciON

    def predict(self):
        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.process_noise

    def update(self, measurement):
        z = np.array([[measurement]])  # Conver matriz 2D
        y = z - self.H @ self.state    
        S = self.H @ self.covariance @ self.H.T + self.measurement_noise
        K = self.covariance @ self.H.T @ np.linalg.inv(S)  # G

        self.state = self.state + K @ y
        self.covariance = (np.eye(2) - K @ self.H) @ self.covariance

        return self.state[0][0]  

def cal_distancia(y_pixeles):
    return 9098.4 * (y_pixeles ** -1.24)

def play_audio(alert_text):
    voice_id = 'es-419'
    engine.setProperty('voice', voice_id)
    with engine_lock:  
        engine.say(alert_text)
        engine.runAndWait()

def should_alert(track_id, class_name, delay=10):
    current_time = time.time()
    if track_id in alerted_ids:
        return False  
    if class_name in last_alert_time and current_time - last_alert_time[class_name] < delay:
        return False  
    last_alert_time[class_name] = current_time
    return True

cap = cv2.VideoCapture(path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
videoseg = '/home/aleja/tesis/13112024/result4.mp4'
out = cv2.VideoWriter(videoseg, cv2.VideoWriter_fourcc(*'mp4v'), 120, (width, height))

while True:
    ret, im0 = cap.read()  
    if not ret:
        break  
    
    annotator = Annotator(im0, line_width=2)
    results = model.track(im0, persist=True)
    mask_cicloruta = np.zeros((height, width), dtype=np.uint8)

    if results[0].boxes.id is not None and results[0].masks is not None:
        masks = results[0].masks.xy
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()  
        class_names = [model.names[class_id] for class_id in class_ids]  
        boxes = results[0].boxes.xyxy.cpu().numpy()

        for mask, track_id, class_name, box in zip(masks, track_ids, class_names, boxes):
            if mask.size == 0:  
                continue 

            y_pos = box[3]  
            #kf = kf.em([[x_pos, y_pos]], n_iter=5)
            #(x_pred, y_pred) = kf.filter([[x_pos, y_pos]])[0][-1]

            if track_id not in kalman_f:
                kalman_f[track_id] = KalmanFilter(y_pos)  
                y_pos_filtrado = y_pos 
            else:
                kalman_f[track_id].predict()  # Predicción
                y_pos_filtrado = kalman_f[track_id].update(y_pos)  
            #distancia = cal_distancia(y_pred)

            distancia_sin_filtro = cal_distancia(y_pos)  
            distancia_con_filtro = cal_distancia(y_pos_filtrado)  

            
            print(f"ID: {track_id}, Clase: {class_name}")
            print(f"  Y sin filtro: {y_pos:.2f}, Distancia sin filtro: {distancia_sin_filtro:.2f} m")
            print(f"  Y con filtro: {y_pos_filtrado:.2f}, Distancia con filtro: {distancia_con_filtro:.2f} m")

            #----------------------------------CICLORUTA--------------------------------------------------
            if class_name == 'cicloruta':
                color = class_colors.get(class_name, (255, 255, 255))  
                mask_cicloruta = cv2.fillPoly(mask_cicloruta, np.int32([mask]), 255)  
                continue  
            
            #if track_id not in kalman_dict:
                #kalman_dict[track_id] = init_kalman()
            #kalman = kalman_dict[track_id]

            #measurement = np.array([[np.float32(box[0])], [np.float32(box[1])]])
            #kalman.correct(measurement)
            #prediction = kalman.predict()

            #x_pred, y_pred = int(prediction[0]), int(prediction[1])

            color = class_colors.get(class_name, (255, 255, 255))
            label = f"{class_name} {track_id} {distancia_con_filtro:.2f} m"
            im0 = cv2.fillPoly(im0, np.int32([mask]), color)
            annotator.box_label(box, label, color=color)

            intersection = cv2.bitwise_and(cv2.fillPoly(np.zeros_like(mask_cicloruta), np.int32([mask]), 255), mask_cicloruta)

            if np.any(intersection > 0):  
                if should_alert(track_id, class_name):
                    alert = f"Alerta: {class_name} está sobre la cicloruta a {distancia_con_filtro:.2f} metros."
                    print(f"-------- ALERTA-----------ALERTA----------ALERTA----------------\n")
                    print(alert)
                    print(f"ID: {track_id}, Y: {y_pos_filtrado:.2f}, Distancia: {distancia_con_filtro:.2f} metros")
                    threading.Thread(target=play_audio, args=(alert,)).start()
                    alerted_ids.add(track_id)

                    #----------------------------------BOLARDO--------------------------------------------------
                    if class_name == 'bolardo':
                        if box[0] > (width * 0.75):
                            alert = "Se acerca el final o un cambio de la cicloruta."
                            print(alert)
                            threading.Thread(target=play_audio, args=(alert,)).start()

                    #----------------------------------CHARCO--------------------------------------------------
                    elif class_name == 'charco':
                        alert = f"ID: {track_id}, Distancia: {distancia_con_filtro:.2f} metros. Reduce la velocidad."
                        alert1 = f"Hay un {class_name} a {distancia_con_filtro:.2f} metros. Reduce la velocidad."
                        print(alert)
                        threading.Thread(target=play_audio, args=(alert1,)).start()

                    #----------------------------------CICLISTA--------------------------------------------------
                    elif class_name == 'ciclista':
                        alert = f"ID: {track_id}, Distancia: {distancia_con_filtro:.2f} metros. Mantenga la distancia."
                        alert1 = f"Hay un {class_name} a {distancia_con_filtro:.2f} metros. Mantenga la distancia."
                        print(alert)
                        threading.Thread(target=play_audio, args=(alert1,)).start()

                    #----------------------------------CRUCE--------------------------------------------------
                    elif class_name == 'cruce':
                        alert = f"ID: {track_id}, Distancia: {distancia_con_filtro:.2f} metros. Reduzca la velocidad."
                        alert1 = f"Hay un {class_name} a {distancia_con_filtro:.2f} metros. Reduzca la velocidad."
                        print(alert)
                        threading.Thread(target=play_audio, args=(alert1,)).start()

                    #---------------------------------DESNIVEL--------------------------------------------------
                    elif class_name == 'desnivel':
                        alert = f"ID: {track_id}, Distancia: {distancia_con_filtro:.2f} metros. Maneje con cuidado."
                        alert1 = f"Hay un {class_name} a {distancia_con_filtro:.2f} metros. Maneje con cuidado."
                        print(alert)
                        threading.Thread(target=play_audio, args=(alert1,)).start()

                    #----------------------------------HUECO--------------------------------------------------
                    elif class_name == 'hueco':
                        if box[0] > (width * 0.5):  # A la derecha
                            alert = f"ID: {track_id}, Distancia: {distancia_con_filtro:.2f} metros. Gire a la izquierda con precaución."
                            alert1 = f"Hay un {class_name} a {distancia_con_filtro:.2f} metros. Gire a la izquierda con precaución."
                        else:  # A la izquierda
                            alert = f"ID: {track_id}, Distancia: {distancia_con_filtro:.2f} metros. Gire a la derecha con precaución."
                            alert1 = f"Hay un {class_name} a {distancia_con_filtro:.2f} metros. Gire a la derecha con precaución."
                        print(alert)
                        threading.Thread(target=play_audio, args=(alert1,)).start()

                    #----------------------------------PARE--------------------------------------------------
                    elif class_name == 'pare':
                        alert = f"ID: {track_id}, Distancia: {distancia_con_filtro:.2f} metros. Deténgase por completo."
                        alert1 = f"Hay un {class_name} a {distancia_con_filtro:.2f} metros. Deténgase por completo."
                        print(alert)
                        threading.Thread(target=play_audio, args=(alert1,)).start()

                    #----------------------------------PEATON--------------------------------------------------
                    elif class_name == 'peaton':
                        if box[0] > (width * 0.5):  # A la derecha
                            alert = f"ID: {track_id}, Distancia: {distancia_con_filtro:.2f} metros. El peatón está a la derecha, maneje con cuidado."
                            alert1 = f"Hay un {class_name} a {distancia_con_filtro:.2f} metros. El peatón está a la derecha, maneje con cuidado."
                        else:  # A la izquierda
                            alert = f"ID: {track_id}, Distancia: {distancia_con_filtro:.2f} metros. El peatón está a la izquierda, maneje con cuidado."
                            alert1 = f"Hay un {class_name} a {distancia_con_filtro:.2f} metros. El peatón está a la izquierda, maneje con cuidado."
                        print(alert)
                        threading.Thread(target=play_audio, args=(alert1,)).start()

                    #----------------------------------POSTE--------------------------------------------------
                    elif class_name == 'poste':
                        if box[0] > (width * 0.5):  # A la derecha
                            alert = f"ID: {track_id}, Distancia: {distancia_con_filtro:.2f} metros. Gire levemente a la izquierda."
                            alert1 = f"Hay un {class_name} a {distancia_con_filtro:.2f} metros. Gire levemente a la izquierda."
                            print(alert)
                            threading.Thread(target=play_audio, args=(alert1,)).start()

                    #----------------------------------VEHICULO MOTOCICLISTA--------------------------------------------------
                    elif class_name in ['vehiculo', 'motociclista']:
                        alert = f"ID: {track_id}, Clase: {class_name}, Distancia: {distancia_con_filtro:.2f} metros. Reduzca la velocidad."
                        alert1 = f"Hay un {class_name} a {distancia_con_filtro:.2f} metros. Reduzca la velocidad."
                        print(alert)
                        threading.Thread(target=play_audio, args=(alert1,)).start()

    
    expanded_mask = cv2.dilate(mask_cicloruta, np.ones((30, 30), np.uint8), iterations=1)
    overlay = im0.copy()
    overlay[expanded_mask > 0] = (0, 255, 255)
    combined = cv2.addWeighted(im0, 0.7, overlay, 0.3, 0)
    out.write(combined)
    cv2.imshow("video", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()




