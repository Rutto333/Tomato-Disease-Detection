from flask import Flask, render_template, request, send_file, send_from_directory
import cv2
import os
import gc  # For garbage collection
import torch
from ultralytics import YOLO

app = Flask(__name__)

# Load model only when needed (lazy loading)
model = None
image_folder = "./images/"

# Disease information dictionary stored outside of request handling
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

def load_model():
    """Load model only when needed and configure for low memory usage"""
    global model
    if model is None:
        # Set torch to use less memory
        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark = False
        
        # Load model with minimal memory footprint
        model = YOLO("best.pt")
        
        # Use half precision if possible
        if torch.cuda.is_available():
            model.half()  # Use FP16 instead of FP32
        
        # Apply quantization for CPU
        else:
            # Only quantize non-CUDA models
            torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
            )

@app.route('/favicon.ico')
def favicon():
    return send_from_directory('static', 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")

@app.route("/", methods=["POST"])
def predict():
    # Load model only when needed
    load_model()
    
    imagefile = request.files["imagefile"]
    image_path = os.path.join(image_folder, imagefile.filename)
    imagefile.save(image_path)

    try:
        # Process with minimal memory
        results = model(image_path, verbose=False)  # Turn off verbose to save memory
        
        if results[0].boxes:
            # Extract first detection details - convert to Python types immediately
            box = results[0].boxes.xyxy[0].cpu().numpy()  # Get only first box
            class_idx = int(results[0].boxes.cls[0].item())  # Convert to Python int
            confidence = float(results[0].boxes.conf[0].item())  # Convert to Python float
            class_name = model.names[class_idx]
            
            # Free tensor memory explicitly
            del results
            
            # Load image at a reduced size for annotation
            image_annotated = cv2.imread(image_path)
            # Optionally resize to reduce memory
            # image_annotated = cv2.resize(image_annotated, (640, 480))
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image_annotated, (x1, y1), (x2, y2), (255, 255, 0), 2)
            
            # Add label text
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(image_annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Save the annotated image
            annotated_filename = f"annotated_{imagefile.filename}"
            annotated_path = os.path.join(image_folder, annotated_filename)
            cv2.imwrite(annotated_path, image_annotated)
            
            # Get disease info
            info = disease_info.get(class_name, {})
            
            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return render_template("home.html",
                                  prediction_text=class_name,
                                  cause=info.get("cause", "Not Available"),
                                  remedy=info.get("remedy", "Not Available"),
                                  annotated_image=annotated_filename)
        else:
            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return render_template("home.html", prediction_text="No class detected", annotated_image=None)
    
    except Exception as e:
        # On error, release memory
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return render_template("home.html", prediction_text=f"Error: {str(e)}", annotated_image=None)

@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_file(os.path.join(image_folder, filename), mimetype="image/jpeg")

if __name__ == "__main__":
    # Create images directory if it doesn't exist
    os.makedirs(image_folder, exist_ok=True)
    # Run with minimal threads
    app.run(threaded=False)