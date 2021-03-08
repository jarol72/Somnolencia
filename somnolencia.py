# importar las librerías necesarias
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import imutils
import time
import dlib
import cv2

# define los valores límites para la apertura de los ojos y la boca,
# así como el número de frames que la apertura de los ojos debe estar
# por debajo del límite de apertura, para disparar la alarma.
EYE_AR_THRESH = 0.25
MOUTH_AR_THRESH = 0.40
EYE_AR_CONSEC_FRAMES = 30

# inicializar el contador de frames y una variable para determinar si la
# alarma está encendida
COUNTER = 0
ALARM_ON = False

# Valor de corrección de gamma de la imagen recibida por la cámara
gamma = 0.6


# verificar si la alarma está encendida
def check_alarm(alarm_sound):
    global ALARM_ON
    if not ALARM_ON:
        ALARM_ON = True
        # crea un nuevo hilo para que la alarma se reproduzca
        # sin interrumpir la ejecución del programa
        t = Thread(target=alarm,
                   args=(alarm_sound,))
        t.daemon = True
        t.start()


# reproduce el sonido de la alarma
def alarm(path):
    playsound.playsound(path)


# calcula la relación de aspecto del ojo
def eye_aspect_ratio(eye):
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])

    c = dist.euclidean(eye[0], eye[3])

    ear = (a + b) / (2.0 * c)

    return ear


# calcula la relación de aspecto de la boca
def mouth_aspect_ratio(mouth):
    a = dist.euclidean(mouth[13], mouth[19])
    b = dist.euclidean(mouth[15], mouth[17])

    c = dist.euclidean(mouth[0], mouth[6])

    mar = (a + b) / (2.0 * c)

    return mar


# realiza la corrección de brillo y contraste de la imagen de video
def gammaCorrection():
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

    res = cv2.LUT(frame, lookUpTable)

    img_gamma_corrected = cv2.hconcat([res])
    return img_gamma_corrected


# inicializa las librerías de reconocimiento facial
print("[INFO] cargando librería de detección facial...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# extrae los índices de correspondientes a los ojos y la boca
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# iniciar la captura de video
print("[INFO] inciando video...")
vs = VideoStream(0).start()
time.sleep(1.0)

# procesar los frames recibidos desde la cámara
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    frame = gammaCorrection()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detecta las caras en la imagen
    rects = detector(gray, 0)

    # procesa cada cara reconocida, identificando los puntos de referencia
    # y extrayendo las áreas de los ojos y boca
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        mouth_AR = mouth_aspect_ratio(mouth)

        ear_avg = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        # Verifica la relación de aspecto de la boca
        if mouth_AR > MOUTH_AR_THRESH:
            # draw an alarm on the frame
            check_alarm("ding.wav")
            cv2.putText(frame, "NECESITA DESCANSAR?", (frame.shape[1]//4, frame.shape[0]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Verifica la relación de aspecto de los ojos
        if ear_avg < EYE_AR_THRESH: # or mouth_AR > MOUTH_AR_THRESH:#
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                check_alarm("alarm.wav")
                cv2.putText(frame, "ALERTA DE SOMNOLENCIA!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0
            ALARM_ON = False

        cv2.putText(frame, "E.A.R.: {:.2f}".format(ear_avg), (320, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(frame, "M.A.R.: {:.2f}".format(mouth_AR), (320, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    # muestra la imagen procesada
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # cierra el rograma si se presiona la tecla `q`
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
