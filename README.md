# 🥦🍎 Fruits & Vegetables Detection App  

[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-red?logo=pytorch)](https://pytorch.org/)  
[![Flask](https://img.shields.io/badge/Flask-Backend-black?logo=flask)](https://flask.palletsprojects.com/)  
[![YOLOv5](https://img.shields.io/badge/YOLOv5-Object%20Detection-green)](https://github.com/ultralytics/yolov5)  

The **Fruits & Vegetables Detection App** is a **Flask web application** powered by a **YOLOv5 deep learning model** trained with PyTorch.  
It detects **32+ classes of fruits and vegetables** from images or webcam input and provides **calorie information** for each item.  

This project highlights skills in **Computer Vision, Flask Web Development, and AI-powered applications**.  

---

## ✨ Features  

- 🖼️ **Upload Image Detection** – Upload an image and get bounding boxes with detected fruits/vegetables.  
- 🎥 **Live Webcam Detection** – Real-time detection directly in the browser.  
- 📦 **32+ Supported Classes** – Apple, Banana, Tomato, Mango, Potato, Strawberry, and more.  
- 🔎 **YOLOv5 Model** – Custom-trained weights (`best.pt`) for accurate detection.  
- 📊 **Calorie Information** – Each detected item shows nutritional values.  
- 🎨 **Flask + HTML Templates** – Lightweight UI with Bootstrap/Tailwind-ready design.  

---

## 🛠️ Tech Stack  

- **PyTorch** – Deep learning framework  
- **YOLOv5** – Object detection model  
- **Flask** – Web framework for serving the app  
- **OpenCV / NumPy** – Image processing utilities  
- **HTML + Jinja2** – Templates for UI rendering  

---

## 🚀 Getting Started  

### 1. Clone the Repository  
```
git clone https://github.com/your-username/fruits-vegetables-detection.git
cd fruits-vegetables-detection
```
2. Install Dependencies
```
pip install -r requirements.txt
```
4. Run the Flask App
```
python app.py
The app will start on http://127.0.0.1:5000/
```
📂 Project Structure
```
fruits-vegetables-detection/
├── images/                # Static images (hero, backgrounds, etc.)
│   └── hero-bg.jpg
│
├── templates/             # HTML templates
│   ├── index.html         # Home page (upload detection)
│   ├── live.html          # Live webcam detection
│   └── result.html        # Display results
│
├── weights/               # Model weights
│   └── best.pt            # YOLOv5 trained model
│
├── app.py                 # Flask application
├── data.yaml              # Dataset configuration file
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```
📈 Optimizations
```
Custom YOLOv5 training with 32 classes of fruits/vegetables.

Lightweight Flask API with minimal latency.

Reusable calorie lookup function.

Separate HTML templates for clean UI organization.
```
🌐 Deployment (Future Scope)
```
Backend (Flask) → Deploy on Render, Heroku, or AWS EC2.

Frontend Templates → Can be extended with React / modern UI later.

Configure environment variables for production.
```
🤝 Contributing
Contributions are welcome! Fork this repo, create a branch, and submit a PR.

📜 License
This project is licensed under the MIT License.

👨‍💻 Author
Built with ❤️ by Syed Abdul Qadeer
Currently pursuing Full-Stack Web Development @ Masai School
#dailylearning #masaiverse

🔖 Tags
Flask · PyTorch · YOLOv5 · Computer Vision · Machine Learning · Deep Learning · Object Detection · Fruits Detection · Vegetables Detection · Calorie Tracking · Portfolio Project
