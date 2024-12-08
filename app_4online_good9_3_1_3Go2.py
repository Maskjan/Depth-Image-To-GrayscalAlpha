import glob
import gradio as gr
import matplotlib
import numpy as np
from PIL import Image, ImageOps
import torch
import tempfile
import cv2
from gradio_imageslider import ImageSlider
import webbrowser
from pyngrok import ngrok, conf  # นำเข้าไลบรารี ngrok และการตั้งค่า

from depth_anything_v2.dpt import DepthAnythingV2

# การตั้งค่าการล็อกเพื่อเอาข้อความ "xFormers not available" ออก
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

css = """
#img-display-container {
    max-height: 100vh;
}
#img-display-input {
    max-height: 80vh;
}
#img-display-output {
    max-height: 80vh;
}
#download {
    height: 62px;
    margin-top: 20px;  # เพิ่มระยะห่างด้านบนของปุ่มดาวน์โหลด
}
"""
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}
encoder = 'vitl'
checkpoint_path = 'D:/00224499/Depth-Anything-V2-main/checkpoints/depth_anything_v2_vitl.pth'
model = DepthAnythingV2(**model_configs[encoder])
state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
model.load_state_dict(state_dict)
model = model.to(DEVICE).eval()

title = "# แปลงภาพเป็นดีฟแม๊ป เกรสเกล "
description = """### Depth Map picture"""

def predict_depth(image):
    return model.infer_image(image)

def enhance_image(img, scale_factor):
    height, width = img.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(resized, -1, sharpen_kernel)
    
    denoised = cv2.fastNlMeansDenoisingColored(sharpened, None, 10, 10, 7, 21)
    
    return denoised

def convert_to_grayscale_and_adjust(image):
    grayscale = ImageOps.grayscale(image)
    grayscale_np = np.array(grayscale)
    return grayscale_np

def convert_webp_to_png(image_path):
    with Image.open(image_path) as img:
        png_path = image_path.replace(".webp", ".png")
        img.save(png_path, "PNG")
    return png_path

def on_submit(image, smooth_factor):
    original_image = Image.fromarray(image)

    # แปลงภาพเป็นเกรสเกลและปรับความคมชัด
    grayscale = convert_to_grayscale_and_adjust(original_image)
    grayscale_image = Image.fromarray(grayscale)

    depth = predict_depth(image[:, :, ::-1])

    raw_depth = Image.fromarray(depth.astype('uint16'))
    tmp_raw_depth = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    raw_depth.save(tmp_raw_depth.name)

    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    depth_colored = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
    colored_depth = (cmap(depth)[:, :, :3] * 255).astype(np.uint8)

    gray_depth = Image.fromarray(depth)
    tmp_gray_depth = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    gray_depth.save(tmp_gray_depth.name)

    # ตรวจสอบและปรับขนาดของ depth ให้ตรงกับ grayscale
    depth_resized = cv2.resize(depth_colored, (grayscale.shape[1], grayscale.shape[0]))

    # ตรวจสอบให้แน่ใจว่าขนาดของภาพตรงกัน
    if grayscale.shape != depth_resized.shape[:2]:
        depth_resized = cv2.resize(depth_resized, (grayscale.shape[1], grayscale.shape[0]))

    # ซ้อนทับภาพเกรสเกลกับ Grayscale depth map รอบแรก
    overlay = cv2.addWeighted(grayscale, 0.7, depth_resized[:, :, 0], 0.3, 0)
    
    # ซ้อนทับภาพดีฟแม๊ปเพิ่มเติมอีกสองรอบตามค่า smooth_factor
    for _ in range(2):
        overlay = cv2.addWeighted(overlay, smooth_factor, depth_resized[:, :, 0], 1 - smooth_factor, 0)
        overlay = cv2.addWeighted(overlay, 0.4, depth_resized[:, :, 0], 0.6, 0)
        overlay = cv2.addWeighted(overlay, 0.9, grayscale, 0.1, 0)
        overlay = cv2.addWeighted(overlay, 0.8, depth_resized[:, :, 0], 0.2, 0)
        overlay = cv2.addWeighted(overlay, 0.7, depth_resized[:, :, 0], 0.3, 0)
        overlay = cv2.addWeighted(overlay, 0.7, grayscale, 0.3, 0)
        overlay = cv2.addWeighted(depth_resized[:, :, 0], 0.9, grayscale, 0.1, 0)

    # บันทึกภาพเกรสเกลที่แปลงเป็น .png
    tmp_grayscale = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    grayscale_image.save(tmp_grayscale.name)

    # บันทึกภาพ overlay เป็น .png
    overlay_img = Image.fromarray(overlay)
    tmp_overlay = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    overlay_img.save(tmp_overlay.name)

    return colored_depth, depth, grayscale, overlay, tmp_gray_depth.name, tmp_raw_depth.name, tmp_grayscale.name, tmp_overlay.name, overlay

def on_enhance(image, enhance_factor):
    enhanced = enhance_image(image, enhance_factor)
    return enhanced

with gr.Blocks(css=css) as demo:
    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row():
        input_image = gr.Image(label="Input Image", type='numpy', elem_id='img-display-input')
        enhanced_image = gr.Image(label="Enhanced Image", type='numpy', elem_id='img-display-enhanced')
        with gr.Column():
            enhance_factor = gr.Slider(label="Enhance Factor", minimum=1, maximum=4, value=1, step=0.1)
            enhance_btn = gr.Button(value="Enhance Image")
            smooth_factor = gr.Slider(label="Smooth Factor", minimum=0, maximum=3, value=1, step=0.1)
            depth_image_slider = gr.Image(label="Depth Map with Slider View", type='numpy', elem_id='img-display-output')
            gray_depth_image = gr.Image(label="Grayscale Depth Map", type='numpy', elem_id='img-display-gray')
            download_gray_depth_file = gr.File(label="Download Grayscale Depth Map (.png)", elem_id="download_gray_depth_file")
        with gr.Column():
            grayscale_image = gr.Image(label="Grayscale Converted Image", type='numpy', elem_id='img-display-grayscale')
            overlay_image = gr.Image(label="Overlay Grayscale and Depth Map", type='numpy', elem_id='img-display-overlay')
            download_overlay_file = gr.File(label="Download Overlay Image (.png)", elem_id="download_overlay_file")

    submit = gr.Button(value="Compute Depth")
    raw_file = gr.File(label="16-bit raw output (can be considered as disparity)", elem_id="download")

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    # เรียกใช้ฟังก์ชัน on_enhance เมื่อกดปุ่ม Enhance Image
    enhance_btn.click(fn=on_enhance, inputs=[input_image, enhance_factor], outputs=enhanced_image)

    # เรียกใช้ฟังก์ชัน on_submit เมื่อกดปุ่ม Compute Depth
    submit.click(on_submit, inputs=[enhanced_image, smooth_factor], outputs=[depth_image_slider, gray_depth_image, grayscale_image, overlay_image, download_gray_depth_file, raw_file, gr.File(label="Converted Grayscale Image (.png)", elem_id="download"), download_overlay_file, overlay_image])

    example_files = glob.glob('assets/examples/*.webp')
    examples = [convert_webp_to_png(f) for f in example_files]
    gr.Examples(examples=examples, inputs=[input_image], outputs=[depth_image_slider, gray_depth_image, grayscale_image, overlay_image, download_gray_depth_file, raw_file, gr.File(label="Converted Grayscale Image (.png)", elem_id="download"), download_overlay_file, overlay_image], fn=on_submit)

if __name__ == '__main__':
    # ตั้งค่า Authtoken
    conf.get_default().auth_token = "2pwNlTxBThbcUfNKGxRAyj7osWr_3PcpSQcc52giKurGDh3ej"

    # เปิดการเชื่อมต่อ ngrok
    public_url = ngrok.connect(7860)
    print(f"Public URL: {public_url}")

    # รัน Gradio demo
    demo.queue().launch(share=True)

    # เปิดเว็บเบราว์เซอร์โดยอัตโนมัติด้วย URL สาธารณะ
    webbrowser.open_new_tab(public_url)

