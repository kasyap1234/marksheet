import cv2
import torch
import re
import json
import base64
import numpy as np
from RealESRGAN import RealESRGAN
from transformers import AutoModel, AutoTokenizer, pipeline

# ======================
# 1. Preprocessing Module
# ======================
class MarksheetPreprocessor:
    def __init__(self, device="cuda"):
        self.sr_model = RealESRGAN(device=device, scale=4)
        self.sr_model.load_weights('weights/RealESRGAN_x4.pth')
        
    def _deskew(self, image):
        """Correct document skew using contour analysis"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, 
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[-1]
        angle = angle - 90 if angle < -45 else angle
        M = cv2.getRotationMatrix2D(rect[0], angle, 1.0)
        return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    def process(self, image_path):
        """Full preprocessing pipeline"""
        img = cv2.imread(image_path)
        
        # Super-Resolution (4x)
        sr_img = self.sr_model.predict(img)
        
        # Deskewing
        deskewed = self._deskew(sr_img)
        
        # Contrast Enhancement
        lab = cv2.cvtColor(deskewed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        limg = cv2.merge([clahe.apply(l), a, b])
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

# ====================
# 2. OCR Module (GOT-OCR2.0)
# ====================
class GOTOCRProcessor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Ucas-HaoranWei/GOT-OCR2.0", 
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            "Ucas-HaoranWei/GOT-OCR2.0",
            device_map="auto",
            torch_dtype=torch.float16
        )
    
    def image_to_base64(self, image):
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')
    
    def extract_text(self, image):
        img_b64 = self.image_to_base64(image)
        result = self.model.chat(
            self.tokenizer,
            img_b64,
            ocr_type="format",
            render=True
        )
        return result

# ====================
# 3. LLM Correction Module
# ====================
class LLMCorrector:
    def __init__(self):
        self.pipe = pipeline(
            "text-generation",
            model="microsoft/Phi-3-mini-4k-instruct",
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=True
        )
    
    def create_prompt(self, ocr_text):
        return f"""<|system|>
        You are a document correction expert. Fix these OCR errors:
        1. Correct roll numbers (e.g., 'O'→'0', 'S'→'5')
        2. Validate marks format (e.g., '95/100')
        3. Return JSON with: roll_no, subjects (dict), total_marks
        
        <|user|>
        OCR Output: {ocr_text}
        <|assistant|>
        """

    def correct(self, ocr_text):
        response = self.pipe(
            self.create_prompt(ocr_text),
            max_new_tokens=400,
            temperature=0.1,
            do_sample=False
        )
        return response[0]['generated_text']

# ====================
# 4. Validation Module
# ====================
class MarksheetValidator:
    @staticmethod
    def sanitize_output(text):
        try:
            # Extract JSON from LLM output
            json_str = text.split("{", 1)[1].rsplit("}", 1)[0]
            return json.loads("{" + json_str + "}")
        except:
            return MarksheetValidator.fallback_validation(text)
    
    @staticmethod
    def fallback_validation(text):
        """Rule-based fallback when LLM fails"""
        roll_no = re.search(r'Roll No[.:]?\s*(\d{6,12})', text)
        marks = re.findall(r'(\b[A-Za-z ]+\b)\s*:\s*(\d+)\s*/\s*(\d+)', text)
        
        return {
            "roll_no": roll_no.group(1) if roll_no else None,
            "subjects": {subj.strip(): f"{score}/{total}" for subj, score, total in marks},
            "total_marks": f"{sum(int(s) for _,s,_ in marks)}/{sum(int(t) for _,_,t in marks)}"
        }

# ====================
# Main Workflow
# ====================
def process_marksheet(image_path):
    # 1. Preprocess
    preprocessor = MarksheetPreprocessor()
    processed_img = preprocessor.process(image_path)
    
    # 2. OCR
    ocr_engine = GOTOCRProcessor()
    raw_text = ocr_engine.extract_text(processed_img)
    
    # 3. LLM Correction
    corrector = LLMCorrector()
    corrected_text = corrector.correct(raw_text)
    
    # 4. Validation
    return MarksheetValidator.sanitize_output(corrected_text)

if __name__ == "__main__":
    result = process_marksheet("blurred_marksheet.jpg")
    print(json.dumps(result, indent=2))