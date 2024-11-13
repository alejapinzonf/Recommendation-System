from ultralytics import YOLO, solutions
from ultralytics.utils.plotting import Annotator, colors
import numpy as np
import cv2
import pyttsx3
import threading
import time

engine = pyttsx3.init() 

engine_lock = threading.Lock() 

model = YOLO('/home/aleja/tesis/23092024/runs/segment/train4/weights/best.pt')
path = '/home/aleja/tesis/videos/video3.mp4'

class_colors = {'bolardo':(255,165,0) ,'charco': (0, 255, 255),'ciclista': (0,255,0),  
                'cicloruta': (100, 0, 0), 'hueco': (0,255,128),
                'cruce': (128, 128, 128), 'desnivel': (64, 224, 208), 'motociclista': (0, 0, 255),
                'pare': (255,255,0), 'peaton': (0,0,128),'poste': (255, 192, 203), 'vehiculo': (128, 0, 128)}

alerted_ids = set()
last_alert_time = {}

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
videoseg = '/home/aleja/tesis/12102024/result4.mp4'
out = cv2.VideoWriter(videoseg, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

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
            distancia = cal_distancia(y_pos)

            #----------------------------------CICLORUTA--------------------------------------------------
            if class_name == 'cicloruta':
                color = class_colors.get(class_name, (255, 255, 255))  
                mask_cicloruta = cv2.fillPoly(mask_cicloruta, np.int32([mask]), 255)  
                continue  

            color = class_colors.get(class_name, (255, 255, 255))
            label = f"{class_name} {track_id} {distancia:.2f} m"
            im0 = cv2.fillPoly(im0, np.int32([mask]), color)
            annotator.box_label(box, label, color=color)

            intersection = cv2.bitwise_and(cv2.fillPoly(np.zeros_like(mask_cicloruta), np.int32([mask]), 255), mask_cicloruta)

            if np.any(intersection > 0):  
                if should_alert(track_id, class_name):
                    alert = f"Alerta: {class_name} está sobre la cicloruta a {distancia:.2f} metros."
                    print(f"-------- ALERTA-----------ALERTA----------ALERTA----------------\n")
                    print(alert)
                    #threading.Thread(target=play_audio, args=(alert,)).start()
                    alerted_ids.add(track_id)
            
                    #if should_alert(track_id, class_name):  

                    #----------------------------------BOLARDO--------------------------------------------------
                    if class_name == 'bolardo':
                        if box[0] > (width * 0.75):
                            alert = "Se acerca el final o un cambio de la cicloruta."
                            print(alert)
                            threading.Thread(target=play_audio, args=(alert1,)).start()

                    #----------------------------------CHARCO--------------------------------------------------
                    elif class_name == 'charco':
                        alert = f"ID: {track_id}, Distancia: {distancia:.2f} metros. Reduce la velocidad."
                        alert1 = f"Hay un {class_name} a {distancia:.2f} metros. Reduce la velocidad."
                        print(alert)
                        threading.Thread(target=play_audio, args=(alert1,)).start()

                    #----------------------------------CICLISTA--------------------------------------------------
                    elif class_name == 'ciclista':
                        alert = f"ID: {track_id}, Distancia: {distancia:.2f} metros. Mantenga la distancia."
                        alert1 = f"Hay un {class_name} a {distancia:.2f} metros. Mantenga la distancia."
                        print(alert)
                        threading.Thread(target=play_audio, args=(alert1,)).start()

                    #----------------------------------CRUCE--------------------------------------------------
                    elif class_name == 'cruce':
                        alert = f"ID: {track_id}, Distancia: {distancia:.2f} metros. Reduzca la velocidad."
                        alert1 = f"Hay un {class_name} a {distancia:.2f} metros. Reduzca la velocidad."
                        print(alert)
                        threading.Thread(target=play_audio, args=(alert1,)).start()

                    #---------------------------------DESNIVEL--------------------------------------------------
                    elif class_name == 'desnivel':
                        alert = f"ID: {track_id}, Distancia: {distancia:.2f} metros. Maneje con cuidado."
                        alert1 = f"Hay un {class_name} a {distancia:.2f} metros. Maneje con cuidado."
                        print(alert)
                        threading.Thread(target=play_audio, args=(alert1,)).start()
                    #----------------------------------HUECO--------------------------------------------------
                    elif class_name == 'hueco':
                        if box[0] > (width * 0.5):  # A la derecha
                            alert = f"ID: {track_id}, Distancia: {distancia:.2f} metros. Gire a la izquierda con precaución."
                            alert1 = f"Hay un {class_name} a {distancia:.2f} metros. Gire a la izquierda con precaución."
                        else:  # A la izquierda
                            alert = f"ID: {track_id}, Distancia: {distancia:.2f} metros. Gire a la derecha con precaución."
                            alert1 = f"Hay un {class_name} a {distancia:.2f} metros. Gire a la derecha con precaución."
                        print(alert)
                        threading.Thread(target=play_audio, args=(alert1,)).start()

                    #----------------------------------PARE--------------------------------------------------
                    elif class_name == 'pare':
                        alert = f"ID: {track_id}, Distancia: {distancia:.2f} metros. Deténgase por completo."
                        alert1 = f"Hay un {class_name} a {distancia:.2f} metros. Deténgase por completo."
                        print(alert)
                        threading.Thread(target=play_audio, args=(alert1,)).start()

                    #----------------------------------PEATON--------------------------------------------------
                    elif class_name == 'peaton':
                        if box[0] > (width * 0.5):  # A la derecha
                            alert = f"ID: {track_id}, Distancia: {distancia:.2f} metros. El peatón está a la derecha, maneje con cuidado."
                            alert1 = f"IHay un {class_name} a {distancia:.2f} metros. El peatón está a la derecha, maneje con cuidado."
                        else:  # A la izquierda
                            alert = f"ID: {track_id}, Distancia: {distancia:.2f} metros. El peatón está a la izquierda, maneje con cuidado."
                            alert1 = f"Hay un {class_name} a {distancia:.2f} metros. El peatón está a la izquierda, maneje con cuidado."
                        print(alert)
                        threading.Thread(target=play_audio, args=(alert1,)).start()

                    #----------------------------------POSTE--------------------------------------------------
                    elif class_name == 'poste':
                        if box[0] > (width * 0.5):  # A la derecha
                            alert = f"ID: {track_id}, Distancia: {distancia:.2f} metros. Gire levemente a la izquierda."
                            alert1 = f"Hay un {class_name} a {distancia:.2f} metros. Gire levemente a la izquierda."
                            print(alert)
                            threading.Thread(target=play_audio, args=(alert1,)).start()

                    #----------------------------------VEHICULO MOTOCICLISTA--------------------------------------------------
                    elif class_name in ['vehiculo', 'motociclista']:
                        alert = f"ID: {track_id}, Clase: {class_name}, Distancia: {distancia:.2f} metros. Reduzca la velocidad."
                        alert1 = f"Hay un {class_name} a {distancia:.2f} metros. Reduzca la velocidad."
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
