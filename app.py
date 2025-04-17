from flask import Flask, render_template, request, send_file, send_from_directory, Response
import cv2
import os
import numpy as np
import base64
from ultralytics import YOLO


image_folder = "./images/"
disease_info = {
    'Early Blight': {
        'cause': 'Fungal disease caused by *Alternaria solani*. Favored by warm temperatures and high humidity.',
        'remedy': 'Spray with fungicides like Score 250 EC or Dithane M-45. Practice crop rotation and proper spacing.'
    },
    'Healthy': {
        'cause': 'No disease detected. Plant appears healthy.',
        'remedy': 'Continue good with the practices. Preventive use of organic sprays like Neem oil can help maintain plant health.'
    },
    'Late Blight': {
        'cause': 'Caused by the oomycete *Phytophthora infestans*. Thrives in cool, wet conditions.',
        'remedy': 'Apply fungicides like Milraz, Acrobat, or Curzate. Avoid overhead irrigation and ensure good drainage.'
    },
    'Leaf Miner': {
        'cause': 'Insect pests (*Liriomyza spp.*) that tunnel through leaf tissues, leaving serpentine trails.',
        'remedy': 'Use insecticides like Tracer, Duduthrin, or Diazinon. Remove and destroy affected leaves.'
    },
    'Leaf Mold': {
        'cause': 'Fungal infection caused by *Passalora fulva*. It thrives in humid greenhouses.',
        'remedy': 'Spray with fungicides such as Copper Oxychloride or Daconil. Ensure proper air circulation.'
    },
    'Mosaic Virus': {
        'cause': 'Caused by Tomato Mosaic Virus (ToMV). Often spread by human handling or contaminated tools.',
        'remedy': 'No chemical cure. Remove and destroy infected plants. Disinfect tools and avoid tobacco use near plants.'
    },
    'Septoria': {
        'cause': 'Fungal disease caused by *Septoria lycopersici*. Common during wet weather.',
        'remedy': 'Apply fungicides like Mancozeb or Copper-based sprays. Remove infected leaves and improve air flow.'
    },
    'Spider Mites': {
        'cause': 'Tiny arachnids (*Tetranychus urticae*) that suck sap. It causes speckled leaves and webbing.',
        'remedy': 'Use acaricides like Agrimec (Abamectin) or natural oils like Neem oil. Spray undersides of leaves.'
    },
    'Yellow Leaf Curl Virus': {
        'cause': 'Transmitted by whiteflies.It causes leaf curling and stunted growth.',
        'remedy': 'Use insecticides like Actara or Imidacloprid to control whiteflies. Use resistant varieties like "Tylka F1".'
    }
}

app = Flask(__name__)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory('static', 'favicon.ico', mimetype='image/vnd.microsoft.icon')


@app.route("/", methods = ["GET"])
def home():
    return render_template("home.html")

@app.route("/", methods = ["POST"])
def predict():
    imagefile = request.files["imagefile"]
    image_path = image_folder + imagefile.filename
    imagefile.save(image_path)

    #loading YOLOnv8 trained model
    model = YOLO("best.pt")

    #obtain prediction
    results = model(image_path)
    if results[0].boxes:

        # Extract first detection details
        boxes = results[0].boxes.xyxy.tolist()  # Bounding box coordinates
        class_indices = results[0].boxes.cls.tolist()  # Class indices
        confidences = results[0].boxes.conf.tolist()  # Confidence scores

        #load the image
        image_annotated = cv2.imread(image_path)
        
        for i, box in enumerate(boxes):
            x1,y1,x2,y2 = map(int, box)  # Convert integers
            class_name = model.names[int(class_indices[i])]
            confidence = confidences[i]

            #Draw bounding boxes
            cv2.rectangle(image_annotated, (x1,y1), (x2,y2), (255,255,0),2)

            #Add label text
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(image_annotated, label, (x1,y1 -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,0), 2)

        # Convert the annotated image to base64 string to display directly in HTML
        _, buffer = cv2.imencode('.jpg', image_annotated)
        annotated_image_b64 = base64.b64encode(buffer).decode('utf-8')
        annotated_image_data = f"data:image/jpeg;base64,{annotated_image_b64}"

        info = disease_info.get(class_name, {})
        return render_template("home.html", 
                               prediction_text = class_name,
                               cause = info.get("cause", "Not Available"),
                               remedy = info.get("remedy", "Not Available"),
                               annotated_image_data = annotated_image_data
                               )
    else:  
        return render_template("home.html", prediction_text= "No class detected", annotated_image_data = None)

@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_file(os.path.join(image_folder, filename), mimetype = "image/jpeg")


if __name__ == "__main__":
    app.run()