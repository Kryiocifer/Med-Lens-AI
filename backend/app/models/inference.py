import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import base64
from io import BytesIO
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


class MedicalInference:
    def __init__(self, weights_dir="../weights"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_dir = weights_dir
        self.confidence_threshold = 70.0

        # Orders MUST match training
        self.class_names = {
            "brain_tumor": ["glioma", "meningioma", "no_tumor", "pituitary"],
            "chest_xray": ["Bacterial Pneumonia", "Normal", "Virus Pneumonia"],
        }

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.models = {}
        self._load_models()

    def _load_checkpoint(self, filename):
        path = os.path.join(self.weights_dir, filename)
        ckpt = torch.load(path, map_location=self.device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            return ckpt["model"]
        return ckpt

    def _make_convnext_dropout(self, num_classes):
        model = models.convnext_small(weights=None)
        in_f = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_f, num_classes),
        )
        return model

    def _load_models(self):
        # Brain — ConvNeXt-Small with Dropout head (this training run)
        brain_model = self._make_convnext_dropout(4)
        brain_model.load_state_dict(self._load_checkpoint("brain_convnext_small_best.pth"))
        brain_model.to(self.device)
        brain_model.eval()
        self.models["brain_tumor"] = brain_model

        # Chest — first-run weights used a plain Linear head
        chest_model = models.convnext_small(weights=None)
        chest_model.classifier[2] = nn.Linear(chest_model.classifier[2].in_features, 3)
        chest_model.load_state_dict(self._load_checkpoint("chest_convnext_small_best.pth"))
        chest_model.to(self.device)
        chest_model.eval()
        self.models["chest_xray"] = chest_model

        print("Both ConvNeXt-Small models loaded")

    def predict(self, image_path: str, scan_type: str):
        print(f"DEBUG: predict() called with scan_type = {scan_type}")

        if scan_type == "auto":
            brain_result = self._single_predict(image_path, "brain_tumor")
            chest_result = self._single_predict(image_path, "chest_xray")

            print(f"Brain Conf: {brain_result['confidence']}% | Chest Conf: {chest_result['confidence']}%")

            brain_conf = brain_result["confidence"]
            chest_conf = chest_result["confidence"]

            gap = abs(brain_conf - chest_conf)
            winner_is_brain = brain_conf >= chest_conf
            winner_conf = max(brain_conf, chest_conf)

            # both weak
            if brain_conf < 70 and chest_conf < 70:
                return {
                    "scan_type": "auto",
                    "prediction": "Uncertain",
                    "confidence": winner_conf,
                    "warning": "Both models have low confidence. Please consult a doctor.",
                    "used_model": "none",
                }

            # pick a winner if the lead is clear enough
            # 90%+ can win with an 8pt lead; otherwise need 12pts
            strong_enough = (
                (winner_conf >= 90 and gap >= 8)
                or (winner_conf >= 70 and gap >= 12)
            )

            if not strong_enough:
                return {
                    "scan_type": "auto",
                    "prediction": "Uncertain",
                    "confidence": winner_conf,
                    "warning": "Uncertain result. The scan doesn't clearly match any condition.",
                    "used_model": "none",
                }

            result = brain_result if winner_is_brain else chest_result
            result["used_model"] = "brain_tumor" if winner_is_brain else "chest_xray"
            result["scan_type"] = "auto"
            return result

        return self._single_predict(image_path, scan_type)

    def _single_predict(self, image_path: str, scan_type: str):
        if scan_type not in self.models:
            raise ValueError("Invalid scan_type")

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        model = self.models[scan_type]
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        class_name = self.class_names[scan_type][predicted.item()]
        confidence_score = round(confidence.item() * 100, 2)

        warning = None
        if confidence_score < self.confidence_threshold:
            warning = f"Low confidence ({confidence_score}%). Please consult a doctor."

        return {
            "scan_type": scan_type,
            "prediction": class_name,
            "class_index": predicted.item(),
            "confidence": confidence_score,
            "warning": warning,
        }

    def generate_gradcam(self, image_path: str, scan_type: str):
        if scan_type not in self.models:
            raise ValueError("Invalid scan_type")

        model = self.models[scan_type]
        target_layers = [model.features[-1]]

        image = Image.open(image_path).convert("RGB")
        rgb_img = np.array(image.resize((224, 224))) / 255.0
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(self._single_predict(image_path, scan_type)["class_index"])]

        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        buffered = BytesIO()
        Image.fromarray(visualization).save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return f"data:image/png;base64,{img_str}"
