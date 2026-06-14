import os
import sys
# 路径配置
ULTRALYTICS_ROOT = r'E:\project\fire_detection_test\ultralytics-8.3.19'
if ULTRALYTICS_ROOT not in sys.path:
    sys.path.insert(0, ULTRALYTICS_ROOT)
import json
import uuid
import cv2
import numpy as np
import requests
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# ========== 配置 ==========
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = r'./runs_fire_smoke/exp_yolo11_85plus_achieved/weights/best.pt'
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2:1.5b"
CONF_THRESHOLD = 0.3

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 全局模型加载（只加载一次）
model = YOLO(MODEL_PATH)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ========== 判读函数 ==========
def get_judgement(box, class_id, conf):
    area = (box[2] - box[0]) * (box[3] - box[1])
    if class_id == 0:  # smoke
        level = 'heavy' if area > 5000 else 'light'
        return f"{level}_smoke", conf, area
    elif class_id == 1:  # fire
        level = 'small' if area < 1000 else ('medium' if area < 3000 else 'large')
        return f"{level}_fire", conf, area

def call_llm_for_analysis(detections_text):
    prompt = f"""你是一个专业的火灾监控AI助手。请仔细分析以下基于YOLO模型检测到的火灾/烟雾目标信息，判断是否存在火灾或火灾风险。

检测到的目标列表：
{detections_text}

分析要求：
1. 明火检测：关注所有标注为"fire"的目标，特别是面积大、置信度高的火焰。
2. 烟雾检测：关注所有标注为"smoke"的目标，浓烟通常预示火势较大。
3. 整体风险：结合火焰和烟雾的数量、面积、等级，评估整体风险等级（high/medium/low/none）。
4. 置信度：综合模型置信度和目标数量，给出你对判断的信心（0.0-1.0）。

回答格式（严格JSON）：
{{"hasFire": true/false, "riskLevel": "high/medium/low/none", "description": "具体描述", "confidence": 0.0-1.0}}"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "max_tokens": 500}
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            llm_output = result.get("response", "").strip()
            if "```json" in llm_output:
                llm_output = llm_output.split("```json")[1].split("```")[0].strip()
            elif "```" in llm_output:
                llm_output = llm_output.split("```")[1].split("```")[0].strip()
            return json.loads(llm_output)
    except Exception as e:
        print(f"LLM调用失败: {e}")
    return None

# ========== 核心API：图片上传+检测+分析 ==========
@app.route('/api/analyze', methods=['POST'])
def analyze_fire():
    if 'image' not in request.files:
        return jsonify({"error": "未上传图片"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "文件名为空"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "只支持JPG/PNG格式"}), 400

    # 保存文件
    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(filepath)

    # YOLO检测
    img = cv2.imread(filepath)
    results = model(filepath, conf=CONF_THRESHOLD)

    # 后处理
    detections = []
    detections_text = ""
    fire_detected = False
    smoke_detected = False

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0])
            conf = float(box.conf[0])

            judgement, conf, area = get_judgement([x1, y1, x2, y2], class_id, conf)

            if class_id == 0:
                smoke_detected = True
                obj_type = "烟雾"
                color = (0, 255, 0)  # 绿色
            else:
                fire_detected = True
                obj_type = "火焰"
                color = (0, 0, 255)  # 红色

            label = f"{judgement} ({conf:.2f})"
            detections.append({
                "type": obj_type,
                "level": judgement,
                "confidence": conf,
                "area": int(area),
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(img, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            detections_text += f"- {obj_type}：等级 {judgement}，置信度 {conf:.2f}，面积约 {area:.0f} 像素\n"

    if not detections:
        detections_text = "未检测到任何火焰或烟雾目标。"
    else:
        detections_text = f"检测到火焰：{'是' if fire_detected else '否'}，烟雾：{'是' if smoke_detected else '否'}\n" + detections_text

    # 调用大模型
    llm_analysis = call_llm_for_analysis(detections_text)

    # 保存结果图
    result_filename = f"result_{unique_name}"
    result_path = os.path.join(UPLOAD_FOLDER, result_filename)
    cv2.imwrite(result_path, img)

    # 构造返回数据
    response_data = {
        "success": True,
        "filename": unique_name,
        "result_image": f"/uploads/{result_filename}",
        "detections": detections,
        "hasFire": llm_analysis.get("hasFire", False) if llm_analysis else False,
        "riskLevel": llm_analysis.get("riskLevel", "none") if llm_analysis else "none",
        "description": llm_analysis.get("description", "无分析结果") if llm_analysis else "无分析结果",
        "confidence": llm_analysis.get("confidence", 0.0) if llm_analysis else 0.0,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    return jsonify(response_data)

# ========== 历史记录API ==========
history_records = []  # 实际项目用数据库

@app.route('/api/history', methods=['GET'])
def get_history():
    return jsonify(history_records)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
