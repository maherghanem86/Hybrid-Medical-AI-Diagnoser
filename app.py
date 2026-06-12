import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import gradio as gr
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from huggingface_hub import hf_hub_download, login

# ==========================================
# 1. المصادقة وإعداد الجهاز
# ==========================================
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("تنبيه: لم يتم العثور على HF_TOKEN في إعدادات Secrets.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 2. إعدادات الفئات والأسماء
# ==========================================
skin_classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
skin_desc = {
    'akiec': 'Actinic Keratoses (محتمل التسرطن)',
    'bcc': 'Basal Cell Carcinoma (سرطان الخلايا القاعدية)',
    'bkl': 'Benign Keratosis (آفة حميدة)',
    'df': 'Dermatofibroma (ورم ليفي جلدي)',
    'mel': 'Melanoma (ميلانوما - سرطان خبيث)',
    'nv': 'Melanocytic Nevi (شامة طبيعية)',
    'vasc': 'Vascular Lesion (آفة وعائية)'
}

xray_classes = ['Normal', 'Pneumonia']
xray_desc = {
    'Normal': 'طبيعي (لا يوجد التهاب)',
    'Pneumonia': 'التهاب رئوي (Pneumonia)'
}

# ==========================================
# 3. الصنف المعماري للنظام الهجين التكيفي
# ==========================================
class EntropyDynamicEnsemble(nn.Module):
    def __init__(self, model1, model2, model3):
        super(EntropyDynamicEnsemble, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        
        # تجميد الأوزان أثناء التجميع
        for param in self.parameters():
            param.requires_grad = False

    def compute_entropy(self, probs):
        return -torch.sum(probs * torch.log(probs + 1e-6), dim=1)

    def forward(self, x):
        probs1 = F.softmax(self.model1(x), dim=1)
        probs2 = F.softmax(self.model2(x), dim=1)
        probs3 = F.softmax(self.model3(x), dim=1)

        ent1 = self.compute_entropy(probs1)
        ent2 = self.compute_entropy(probs2)
        ent3 = self.compute_entropy(probs3)

        entropies = torch.stack([ent1, ent2, ent3], dim=1)
        weights = F.softmax(-entropies, dim=1)

        w1 = weights[:, 0].unsqueeze(1)
        w2 = weights[:, 1].unsqueeze(1)
        w3 = weights[:, 2].unsqueeze(1)

        dynamic_probs = (w1 * probs1) + (w2 * probs2) + (w3 * probs3)
        return dynamic_probs

# ==========================================
# 4. بناء النماذج وتحميل الأوزان
# ==========================================
def load_base_models(repo_id, num_classes, prefix=""):
    print(f"جاري تحميل أوزان {repo_id}...")
    
    # تحميل الأوزان
    res_path = hf_hub_download(repo_id=repo_id, filename=f"resnet50_{prefix}best_weights.pth")
    dense_path = hf_hub_download(repo_id=repo_id, filename=f"densenet121_{prefix}best_weights.pth")
    eff_path = hf_hub_download(repo_id=repo_id, filename=f"efficientnet_b0_{prefix}best_weights.pth")
    
    # ResNet-50
    m_res = models.resnet50(weights=None)
    m_res.fc = nn.Linear(m_res.fc.in_features, num_classes)
    m_res.load_state_dict(torch.load(res_path, map_location=device))
    m_res = m_res.to(device).eval()
    
    # DenseNet-121
    m_dense = models.densenet121(weights=None)
    m_dense.classifier = nn.Linear(m_dense.classifier.in_features, num_classes)
    m_dense.load_state_dict(torch.load(dense_path, map_location=device))
    m_dense = m_dense.to(device).eval()
    
    # EfficientNet-B0
    m_eff = models.efficientnet_b0(weights=None)
    m_eff.classifier[1] = nn.Linear(m_eff.classifier[1].in_features, num_classes)
    m_eff.load_state_dict(torch.load(eff_path, map_location=device))
    m_eff = m_eff.to(device).eval()
    
    # تجميع النظام الهجين (هذه الخطوة ستقوم بتجميد كافة الأوزان)
    hybrid = EntropyDynamicEnsemble(m_res, m_dense, m_eff).to(device).eval()
    
    # تفعيل التدرجات لنموذج ResNet حصراً من أجل Grad-CAM بعد تجميع النظام
    for param in m_res.parameters():
        param.requires_grad = True

    return hybrid, m_res

# تحميل نماذج الجلد (7 فئات)
skin_hybrid, skin_explainer = load_base_models("maherghanem86/skin-cancer-models", 7, "skin_")

# تحميل نماذج الأشعة (فئتان)
xray_hybrid, xray_explainer = load_base_models("maherghanem86/chest-xray-models", 2, "")

# ==========================================
# 5. دوال التوقع والتفسير البصري (Grad-CAM)
# ==========================================
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def generate_gradcam(img_tensor, original_img_array, explainer_model, target_class):
    target_layers = [explainer_model.layer4[-1]]
    cam = GradCAM(model=explainer_model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_class)]
    
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    visualization = show_cam_on_image(original_img_array, grayscale_cam, use_rgb=True)
    return visualization

def predict_hybrid(image, task_type):
    if image is None:
        return None, None
        
    # تجهيز الصورة
    image = image.convert('RGB')
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    
    img_array = np.array(image.resize((224, 224)), dtype=np.float32) / 255.0
    
    if task_type == "skin":
        model = skin_hybrid
        explainer = skin_explainer
        classes = skin_classes
        desc_dict = skin_desc
    else:
        model = xray_hybrid
        explainer = xray_explainer
        classes = xray_classes
        desc_dict = xray_desc

    with torch.no_grad():
        probs = model(img_tensor).squeeze().cpu().numpy()
        
    top_class_idx = np.argmax(probs)
    
    # تجهيز قاموس الاحتمالات للواجهة
    result_dict = {desc_dict[classes[i]]: float(probs[i]) for i in range(len(classes))}
    
    # توليد الخريطة الحرارية (Grad-CAM)
    explainer.zero_grad()
    heatmap_img = generate_gradcam(img_tensor, img_array, explainer, top_class_idx)
    
    return result_dict, heatmap_img

# ==========================================
# 6. بناء واجهة Gradio
# ==========================================
with gr.Blocks(title="نظام الذكاء الاصطناعي الطبي الهجين") as interface:
    gr.Markdown("<h1 style='text-align: center;'>نظام الذكاء الاصطناعي الهجين (Dynamic Ensemble) للتشخيص الطبي</h1>")
    gr.Markdown("<p style='text-align: center;'>هذا النظام يدمج قرارات ثلاث شبكات عصبية (ResNet, DenseNet, EfficientNet) بشكل ديناميكي يعتمد على حساب الإنتروبيا (معدل الشك)، ويقدم تفسيراً بصرياً لقراره لتوضيح مناطق الإصابة.</p>")

    with gr.Tabs():
        # التبويب الأول: الأمراض الجلدية
        with gr.TabItem("🔬 تشخيص الأمراض الجلدية (Skin Cancer)"):
            with gr.Row():
                with gr.Column():
                    skin_in = gr.Image(type="pil", label="قم برفع صورة الآفة الجلدية (Dermoscopy)")
                    skin_btn = gr.Button("تحليل الصورة الجلدية", variant="primary")
                with gr.Column():
                    skin_out_label = gr.Label(num_top_classes=3, label="القرار المدمج (أعلى احتمالات)")
                    skin_out_img = gr.Image(label="خريطة التفسير الحرارية (Grad-CAM)")
            skin_btn.click(lambda img: predict_hybrid(img, "skin"), inputs=skin_in, outputs=[skin_out_label, skin_out_img])
            
        # التبويب الثاني: الأشعة السينية
        with gr.TabItem("🩻 تشخيص الأشعة السينية (Chest X-Ray)"):
            with gr.Row():
                with gr.Column():
                    xray_in = gr.Image(type="pil", label="قم برفع صورة الأشعة السينية للصدر")
                    xray_btn = gr.Button("تحليل صورة الأشعة", variant="primary")
                with gr.Column():
                    xray_out_label = gr.Label(label="القرار المدمج (التهاب رئوي أم طبيعي)")
                    xray_out_img = gr.Image(label="خريطة التفسير الحرارية (Grad-CAM)")
            xray_btn.click(lambda img: predict_hybrid(img, "xray"), inputs=xray_in, outputs=[xray_out_label, xray_out_img])

interface.launch()