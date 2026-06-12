import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from docx import Document
import matplotlib.pyplot as plt
import numpy as np
import io

def generate_graphs():
    # Data extracted from CSE498V2.ipynb and Custom CNN.ipynb
    models = ['DenseNet121', 'ResNet50', 'EfficientNet-B0', 'MobileNetV3', 'ViT Small']
    accuracy = [0.8982, 0.8965, 0.8949, 0.8678, 0.8235]
    f1_score = [0.8974, 0.8957, 0.8944, 0.8660, 0.8214]
    
    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.style.use('dark_background')
    
    bars1 = plt.bar(x - width/2, accuracy, width, label='Accuracy', color='#00d2ff')
    bars2 = plt.bar(x + width/2, f1_score, width, label='F1-Score', color='#3a7bd5')

    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, models)
    plt.legend()
    plt.ylim(0, 1.0)
    
    # Add text on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom', fontsize=9)
            
    img_stream = io.BytesIO()
    plt.savefig(img_stream, format='png', bbox_inches='tight', dpi=300)
    img_stream.seek(0)
    plt.close()
    return img_stream

def create_presentation(output_path, img_stream):
    prs = Presentation()
    
    # Title Slide
    title_slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(title_slide_layout)
    title = slide.shapes.title
    subtitle = slide.placeholders[1]
    title.text = "CSE498R Research Presentation"
    subtitle.text = "Eye Disease Image Classification using Custom DenseNet121\nand API Deployment"

    # Slide 2: Introduction
    bullet_slide_layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide(bullet_slide_layout)
    shapes = slide.shapes
    title_shape = shapes.title
    body_shape = shapes.placeholders[1]
    title_shape.text = "Introduction & Objective"
    tf = body_shape.text_frame
    tf.text = "Objective: Develop a robust, high-performance Deep Learning model for classifying eye diseases from images."
    p = tf.add_paragraph()
    p.text = "Significance: Early and accurate detection of eye diseases can prevent vision loss."
    p = tf.add_paragraph()
    p.text = "Approach: Evaluated multiple state-of-the-art architectures (ResNet50, EfficientNet, ViT, DenseNet) and designed a custom CNN (DenseEyeNet) based on DenseNet121."

    # Slide 3: Model Architectures Evaluated
    slide = prs.slides.add_slide(bullet_slide_layout)
    shapes = slide.shapes
    title_shape = shapes.title
    body_shape = shapes.placeholders[1]
    title_shape.text = "Model Architectures Evaluated"
    tf = body_shape.text_frame
    tf.text = "We comprehensively trained and tested several models:"
    p = tf.add_paragraph()
    p.text = "ResNet50: Solid baseline with skip connections."
    p = tf.add_paragraph()
    p.text = "EfficientNet-B0: Optimized for accuracy and computational efficiency."
    p = tf.add_paragraph()
    p.text = "MobileNetV3-Small: Lightweight model suitable for mobile deployment."
    p = tf.add_paragraph()
    p.text = "ViT Small: Vision Transformer capturing global image context."
    p = tf.add_paragraph()
    p.text = "DenseNet121 (DenseEyeNet): Custom CNN leveraging dense connections, which emerged as our best performing model."

    # Slide 4: Performance Comparison
    blank_slide_layout = prs.slide_layouts[5]
    slide = prs.slides.add_slide(blank_slide_layout)
    slide.shapes.title.text = "Model Performance Comparison"
    slide.shapes.add_picture(img_stream, Inches(1), Inches(1.5), width=Inches(8))

    # Slide 5: API Integration & Deployment
    slide = prs.slides.add_slide(bullet_slide_layout)
    shapes = slide.shapes
    title_shape = shapes.title
    body_shape = shapes.placeholders[1]
    title_shape.text = "API Integration & Deployment"
    tf = body_shape.text_frame
    tf.text = "A robust backend was built using FastAPI for real-time model inference."
    p = tf.add_paragraph()
    p.text = "Framework: FastAPI (main.py) ensures high concurrency and fast responses."
    p = tf.add_paragraph()
    p.text = "Endpoints: '/models' lists available architectures; '/predict' accepts an image upload and returns class probabilities."
    p = tf.add_paragraph()
    p.text = "Image Processing: image_utils.py handles preprocessing (resizing, normalization) seamlessly."
    p = tf.add_paragraph()
    p.text = "Model Loader: model_loader.py dynamically loads the pre-trained DenseNet121 PyTorch bundle to GPU/CPU."

    # Slide 6: Conclusion
    slide = prs.slides.add_slide(bullet_slide_layout)
    shapes = slide.shapes
    title_shape = shapes.title
    body_shape = shapes.placeholders[1]
    title_shape.text = "Conclusion & Future Work"
    tf = body_shape.text_frame
    tf.text = "Conclusion:"
    p = tf.add_paragraph()
    p.text = "DenseNet121 achieved the highest accuracy (89.82%) and F1-score (89.74%)."
    p = tf.add_paragraph()
    p.text = "The FastAPI backend successfully bridges the gap between deep learning research and practical clinical application."
    p = tf.add_paragraph()
    p.text = "Future Work:"
    p = tf.add_paragraph()
    p.text = "Further expansion using Federated Learning techniques to ensure data privacy."
    p = tf.add_paragraph()
    p.text = "Integration of Explainable AI (XAI) to provide clinicians with interpretable visual heatmaps."
    
    prs.save(output_path)

def create_script(output_path):
    doc = Document()
    doc.add_heading('CSE498R Research Presentation - Script', 0)

    doc.add_heading('Slide 1: Title Slide', level=2)
    doc.add_paragraph("Welcome everyone to our presentation on CSE498R Research. Today, we will be discussing our work on 'Eye Disease Image Classification using a Custom DenseNet121 model and its API Deployment'.")

    doc.add_heading('Slide 2: Introduction & Objective', level=2)
    doc.add_paragraph("Our primary objective for this research was to develop a robust and highly accurate Deep Learning model capable of classifying various eye diseases from medical images. The significance of this project lies in the fact that early and accurate detection can be crucial in preventing severe vision loss in patients. To achieve this, we evaluated several state-of-the-art architectures and ultimately focused on designing a custom CNN, which we named DenseEyeNet, built upon the DenseNet121 architecture.")

    doc.add_heading('Slide 3: Model Architectures Evaluated', level=2)
    doc.add_paragraph("During our experimentation phase, we trained and compared a diverse set of models to establish a strong baseline and find the optimal architecture. We tested ResNet50 for its reliable skip connections, EfficientNet-B0 for its balance of accuracy and efficiency, MobileNetV3-Small for potential mobile applications, and even Vision Transformers (ViT Small) to see if global attention mechanisms would yield better results. However, our Custom CNN based on DenseNet121, which utilizes dense connectivity to encourage feature reuse, consistently outperformed the rest.")

    doc.add_heading('Slide 4: Model Performance Comparison', level=2)
    doc.add_paragraph("As you can see in this performance comparison chart, the DenseNet121 model achieved the highest metrics across the board, with an accuracy of 89.82% and an F1-score of 89.74%. ResNet50 and EfficientNet-B0 also performed admirably, scoring around 89.6% and 89.4% respectively, while the Vision Transformer and MobileNet lagged slightly behind. These statistics clearly validate our choice of DenseNet121 as the primary model for deployment.")

    doc.add_heading('Slide 5: API Integration & Deployment', level=2)
    doc.add_paragraph("To make our research applicable in a real-world scenario, we developed a robust backend API using FastAPI. This API allows for real-time model inference. It features endpoints to list the available models and a prediction endpoint where users can upload an image and receive the top disease classifications along with their probabilities. Under the hood, we implemented efficient image preprocessing and dynamic model loading to ensure the system is both fast and scalable.")

    doc.add_heading('Slide 6: Conclusion & Future Work', level=2)
    doc.add_paragraph("In conclusion, our custom DenseNet121 architecture provided the best performance for eye disease classification, and our FastAPI deployment successfully bridges the gap between research and practical use. Moving forward, we plan to enhance this system by integrating Federated Learning to address data privacy concerns, and Explainable AI (XAI) to generate heatmaps, giving clinicians better insight into how the model makes its decisions. Thank you.")

    doc.save(output_path)

if __name__ == '__main__':
    out_dir = r"D:\CSE498R Presentation"
    os.makedirs(out_dir, exist_ok=True)
    
    img_stream = generate_graphs()
    
    ppt_path = os.path.join(out_dir, "CSE498R presentation.pptx")
    create_presentation(ppt_path, img_stream)
    print(f"Presentation saved to {ppt_path}")
    
    docx_path = os.path.join(out_dir, "Script.docx")
    create_script(docx_path)
    print(f"Script saved to {docx_path}")
