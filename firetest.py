import cv2
import supervision as sv
from ultralytics import YOLO
import typer
import numpy as np
import logging
import os
import pandas as pd
from datetime import datetime
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\asp\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

plate_model = YOLO("lprbest.pt")
helmet_model = YOLO("best.pt")    

app = typer.Typer()



import firebase_admin
from firebase_admin import credentials, firestore
import smtplib
from email.mime.text import MIMEText

def send_email(to_email, subject, body):
  
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    sender_email = "zaidinnovates@gmail.com"
    sender_password = "heme tnjp zmqf qhqf" 

    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = to_email

        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, to_email, msg.as_string())
        server.quit()
        print(f"✅ Email sent successfully to {to_email}")

    except Exception as e:
        print(f"❌ Failed to send email: {e}")

def fetch_email_and_send_notification(reg_number):
    try:
        cred = credentials.Certificate("serviceAccountkey.json")
        firebase_admin.initialize_app(cred)

        db = firestore.client()


        docs = db.collection("bikeRegistrations").where("regNumber", "==", reg_number).stream()

        matched = False
        for doc in docs:
            data = doc.to_dict()
            email = data.get("email")
            if email:
                matched = True
                print(f"✅ Found email for {reg_number}: {email}")

            
                subject = "Violation Notice"
                body = f"""Dear User,

                This is to inform you that your vehicle bearing registration number {reg_number} was observed in violation of traffic safety regulations due to the rider not wearing a helmet.

                Wearing a helmet is mandatory under the Motor Vehicle Act and is essential for your safety.

                You are kindly advised to comply with traffic rules to avoid further penalties.

                Regards,  
                Traffic Enforcement Department
                """

                send_email(email, subject, body)
                break

        if not matched:
            print(f"❌ No matching record found for registration number: {reg_number}")

    except Exception as e:
        print(f"🔥 Error: {e}")



















def log_violation_to_excel(plate_text):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = "violations.xlsx"

    df = pd.DataFrame([[plate_text, timestamp]], columns=["License Plate", "Date & Time"])

    if os.path.exists(log_path):

        existing = pd.read_excel(log_path)
        updated = pd.concat([existing, df], ignore_index=True)
        updated.to_excel(log_path, index=False)
    else:
        df.to_excel(log_path, index=False)
    print(f"[Excel] Logged: {plate_text} at {timestamp}")


def perform_ocr_tesseract(image_array):
    if image_array is None or image_array.size == 0:
        logging.warning("Empty image provided for OCR")
        return ""
    try:

        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
      
        gray = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
      
        text = pytesseract.image_to_string(gray, config='--psm 8')
        return text.strip()
    except Exception as e:
        logging.error(f"OCR error: {e}")
        return ""


def process_webcam(output_file="output.mp4"):
    cap = cv2.VideoCapture("demo.mp4")

    if not cap.isOpened():
        logging.error("Could not open video file.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logging.info(f"Input resolution: {width}x{height}, FPS: {fps}")

    target_width = 1280
    target_height = 720
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, fps, (target_width, target_height))

    if not out.isOpened():
        logging.error("Could not initialize VideoWriter.")
        cap.release()
        return

    os.makedirs("output", exist_ok=True)
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        logging.info(f"Processing frame {frame_count}")
        frame = cv2.resize(frame, (target_width, target_height))
        annotated_frame = frame.copy()

       
        helmet_results = helmet_model(frame)[0]
        helmet_detections = sv.Detections.from_ultralytics(helmet_results)

        without_helmet_boxes = []
        for xyxy, class_id in zip(helmet_detections.xyxy, helmet_detections.class_id):
            class_name = helmet_results.names[class_id]
            if class_name.lower() == "without helmet":
                without_helmet_boxes.append(xyxy)


        annotated_frame = box_annotator.annotate(annotated_frame, helmet_detections)
        annotated_frame = label_annotator.annotate(annotated_frame, helmet_detections)

     
        plate_results = plate_model(frame)[0]
        plate_detections = sv.Detections.from_ultralytics(plate_results)

        for i, (xyxy, confidence, class_id) in enumerate(zip(plate_detections.xyxy, plate_detections.confidence, plate_detections.class_id)):
            x1, y1, x2, y2 = map(int, xyxy)
            plate_center_x = (x1 + x2) / 2
            plate_center_y = (y1 + y2) / 2

       
            matched = False
            for helmet_xyxy in without_helmet_boxes:
                hx1, hy1, hx2, hy2 = map(int, helmet_xyxy)

                if (
                    hx1 - 30 <= plate_center_x <= hx2 + 30 and  
                    hy2 <= plate_center_y <= hy2 + 250                         ):
                    matched = True
                    break

            if matched:
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

 
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"plate_{timestamp}.jpg"
                plate_path = os.path.join("output", filename)
                cv2.imwrite(plate_path, crop)
                logging.info(f"✅ Saved plate image: {plate_path}")

         
                text = perform_ocr_tesseract(crop)
                if text:
                    logging.info(f"Detected plate: {text}")


                    log_violation_to_excel(text)  
                    fetch_email_and_send_notification(text)

               
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


        out.write(annotated_frame)
        cv2.imshow("Helmet & Plate Detection", annotated_frame)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    logging.info(f"Video saved to: {output_file}")

@app.command()
def webcam(output_file: str = "output1.avi"):
    typer.echo("Starting video processing...")
    process_webcam(output_file)

if __name__ == "__main__":
    app()





