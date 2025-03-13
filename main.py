from ultralytics import YOLO
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox
from PyQt6.QtGui import QDesktopServices, QIcon
from PyQt6.QtCore import QUrl
from PIL import Image
import fitz, joblib, shutil, os, sys, cv2

# To convert pages of PDF to Image
def pdf_to_images(pdf_path):
    print("Creating Image of Each page to Predict from YOLO Model")
    output_folder = f'tmp'
    # Extract PDF name
    pdf_name = os.path.basename(pdf_path).split(".")[0]
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    # Hide Cursor and
    print("\033[?25l",end="")
    # To create a Loading Effect
    spinner = ["-", "/", "|", "\\"]
    i = -1
    for page_num in range(len(doc)):
        i += 1
        # \r Replaces the already written text
        print(f"\r{spinner[ i % 4 ]}",end="")
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300)
        output_path = f"{output_folder}/{pdf_name}/page_{page_num + 1}.png"
        # Create output dir if not exist
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        if not os.path.exists(f"{output_folder}/{pdf_name}"):
            os.mkdir(f"{output_folder}/{pdf_name}")
        pix.save(output_path)
    # Show Cursor
    print("\033[?25h\rAll Pages Exported Successfully!")
    return [total_pages, f"{output_folder}/{pdf_name}"]

# It is used to import Already trained Models and the Categories globally
def import_requirements():
    global log_model, vectorizer, yolo_model, secondary_classes, primary_classes
    print("\nLoading Trained Models\n")
    log_model = joblib.load("Models/logistic_regression_model.pkl")
    yolo_model = YOLO("Models/yolo_model.pt")
    vectorizer = joblib.load("Models/tfidf_vectorizer.pkl")
    # Classes of Caption Classification
    secondary_classes = [
                      "Contour_Maps",
                      "Drilling_Plots",
                      "Geological_Map",
                      "Geotechnical_Order",
                      "Location_Map",
                      "Log_Motif",
                      "Remote_Sensing_Image",
                      "Seismic_Section",
                      "Stratigraphy_and_Casing_Plot",
                      "Structural_Map",
                      "Well_Construction_Diagram",
                      "Well_Schematic_Diagram",
                      "Others"
    ]
    # Classes of Yolo Image Detections
    primary_classes = [
                "figure_with_label",
                "figure_without_label",
                "graph"
    ]

# Used to delete Images of PDF pages at End
def delete_unused_images(path):
    print("Deleting Image of each Page!")
    # To recursively delete
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)

# Used to Predict Caption Classification Class from NLP Model
def predict_class(caption):
    # Convert text to TF-IDF features
    caption_tfidf = vectorizer.transform([caption])
    # Feed these features to Model for Prediction
    predicted_index = log_model.predict(caption_tfidf)[0]
    return [predicted_index, secondary_classes[predicted_index]]

# Function to Export Image and capture Caption if Detected
def detect_and_classify(pdf_path):
    total_pages, pages_path = pdf_to_images(pdf_path)
    pdf_name = os.path.basename(pdf_path).split(".")[0]
    output_dir = f"captured_images/{pdf_name}"
    # List of relative secondary_class directories
    dirs = [f"{primary_classes[0]}/{cls}" for cls in secondary_classes]
    dirs += [primary_classes[1], primary_classes[2]]
    for subfolder in dirs:
        os.makedirs(os.path.join(output_dir, subfolder), exist_ok=True)
    doc = fitz.open(pdf_path)
    for page in range(total_pages):
        page_path = f"{pages_path}/page_{page + 1}.png"
        # Predict from YOLO Model
        results = yolo_model(page_path)
        for idx, box in enumerate(results[0].boxes.xyxy):  # Bounding boxes (x0, y0, x1, y1)
            x0, y0, x1, y1 = map(int, box)  # Convert to integers
            cropped_img = cv2.imread(page_path)[y0:y1, x0:x1]  # Crop the detected object
            pdf_page = doc.load_page(page)
            page_size_yolo = list(Image.open(page_path).size)
            page_size_fitz = [pdf_page.rect.width, pdf_page.rect.height]
            # Factor to convert Yolo coordinate to fitz
            height_scale = page_size_fitz[1]/page_size_yolo[1]
            cls = int(results[0].boxes.cls[idx])
            primary_class = results[0].names[cls]
            caption = ""
            secondary_class = ""
            # Capture Caption
            if primary_class == "figure_with_label":
                # Area to Search Caption
                caption_area = fitz.Rect(
                            0,                           # left most of Page
                            int(y0*height_scale),        # Top border of captured image
                            page_size_fitz[0],           # right most of Page
                            min(page_size_fitz[1], int(y1*height_scale) + 50)
                            # Lower border of captured image + 50 extra | Page Bottom
                        )
                # Extract text blocks
                text_blocks = pdf_page.get_text("blocks")
                # Indicate that current text block to be captured or not
                continued = True
                # Indicate Caption capturing is started or not 
                started = False
                for block in text_blocks:
                    block_bbox = fitz.Rect(block[:4])
                    if caption_area.contains(block_bbox):
                        if block[4].strip()[:3].lower() == "fig":
                            started = True
                        if continued and started:
                            caption[ : len(caption) - 1]
                            caption = caption + block[4].strip()
                        if started and block[4].strip() and block[4].strip()[-1] != "-":
                            continued = False
                if caption:
                    secondary_class = predict_class(caption)[1]+"/"
                    caption_path = f"{output_dir}/{primary_class}/{secondary_class}page_{page + 1}_object_{idx + 1}.txt"
                    with open(caption_path, "w", encoding="utf-8") as f:
                        f.write(caption)
            # Save the cropped image
            save_path = f"{output_dir}/{primary_class}/{secondary_class}page_{page + 1}_object_{idx + 1}.png"
            cv2.imwrite(save_path, cropped_img)
    delete_unused_images(pages_path.split("/")[0])
    success_popup(output_dir)

def browse_pdf():
    print("Please! Select the PDFs from the Popup..")
    files_path, _ = QFileDialog.getOpenFileNames(None, "Select PDFs to Extract and Classify Image in it!", "", "PDF Files(*pdf)")
    return files_path

def success_popup(output_directory):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Icon.Information)
    msg.setWindowIcon(QIcon("logo.png"))
    msg.setWindowTitle("Success")
    msg.setText("Success! Extraction and Classification Completed.\nClick! OK to Open the Output.")
    msg.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
    if msg.exec() == QMessageBox.StandardButton.Ok:
        QDesktopServices.openUrl(QUrl.fromLocalFile(output_directory))

def execute():
    paths = browse_pdf()
    import_requirements()
    for pdf_path in paths:
        print(f'Extracting from {pdf_path}')
        print("-----------------------------------------------------------------------------")
        detect_and_classify(pdf_path)
        print("-----------------------------------------------------------------------------")
        print("Extraction & Classification Completed Successfully!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    execute()