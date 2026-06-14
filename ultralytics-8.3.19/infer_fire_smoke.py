import os
import json
import requests

os.environ['WANDB_MODE'] = 'disabled'

from ultralytics import YOLO
import cv2
import numpy as np

# ========== 配置 ==========
MODEL_PATH = './runs_fire_smoke/exp_yolo11_85plus_achieved/weights/best.pt'  # ✅ 训练好的模型
IMG_PATH = './test1.jpg'

# 本地 Ollama 服务地址（默认端口11434）
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2:1.5b"  # 与你本地运行的模型名称一致

CONF_THRESHOLD = 0.1  # 置信度阈值

# 判读规则（和 my_data.yaml 里的 judgement 一致）
FIRE_LEVELS = {
    'small': (0, 1000),  # 面积 < 1000 → 小火
    'medium': (1000, 3000),  # 1000~3000 → 中火
    'large': (3000, 999999)  # > 3000 → 大火
}

SMOKE_LEVELS = {
    'light': (0, 5000),  # 面积 < 5000 → 轻烟
    'heavy': (5000, 999999)  # > 5000 → 浓烟
}


# ========== 判读函数 ==========
def get_judgement(box, class_id, conf):
    """
    box: [x1, y1, x2, y2]
    class_id: 0=smoke, 1=fire
    conf: 置信度
    """
    area = (box[2] - box[0]) * (box[3] - box[1])  # 计算框面积

    if class_id == 0:  # smoke
        level = 'heavy' if area > 5000 else 'light'
        return f"{level}_smoke", conf, area

    elif class_id == 1:  # fire
        if area < 1000:
            level = 'small'
        elif area < 3000:
            level = 'medium'
        else:
            level = 'large'
        return f"{level}_fire", conf, area


# ========== 调用本地大模型（Ollama）==========
def call_llm_for_analysis(detections_text):
    """
    发送检测结果文本到本地大模型，获取 JSON 分析结果
    """
    # 构建完整的提示词
    prompt = f"""
你是一个专业的火灾监控AI助手。请仔细分析以下基于YOLO模型检测到的火灾/烟雾目标信息，判断是否存在火灾或火灾风险。

检测到的目标列表：
{detections_text}

**分析要求：**
1. 明火检测：关注所有标注为"fire"的目标，特别是面积大、置信度高的火焰。
2. 烟雾检测：关注所有标注为"smoke"的目标，浓烟通常预示火势较大。
3. 整体风险：结合火焰和烟雾的数量、面积、等级，评估整体风险等级（high/medium/low/none）。
4. 置信度：综合模型置信度和目标数量，给出你对判断的信心（0.0-1.0）。

**回答格式（严格JSON，不要输出其他文字）：**
{{
    "hasFire": true/false,
    "riskLevel": "high/medium/low/none",
    "description": "具体描述观察到的火灾情况，比如检测到几处火焰、几处烟雾，面积情况如何，综合风险评估",
    "confidence": 0.0-1.0
}}
"""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,  # 降低随机性，让输出更确定
            "max_tokens": 500
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            llm_output = result.get("response", "").strip()
            # 尝试解析 JSON
            # 有时LLM会输出```json ... ```，需要清理
            if "```json" in llm_output:
                llm_output = llm_output.split("```json")[1].split("```")[0].strip()
            elif "```" in llm_output:
                llm_output = llm_output.split("```")[1].split("```")[0].strip()
            analysis = json.loads(llm_output)
            return analysis
        else:
            print(f"⚠️ LLM 调用失败，HTTP状态码：{response.status_code}")
            return None
    except Exception as e:
        print(f"⚠️ LLM 调用异常：{e}")
        return None


# ========== 推理 ==========
model = YOLO(MODEL_PATH)
results = model(IMG_PATH, conf=CONF_THRESHOLD)

# 读取原始图片（只需读一次）
img = cv2.imread(IMG_PATH)

# ========== 后处理（判读）+ 收集检测信息用于LLM ==========
detected = False
detections_text = ""  # 用于传给大模型的文字描述
fire_detected = False
smoke_detected = False

for result in results:
    for box in result.boxes:
        detected = True
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        class_id = int(box.cls[0])
        conf = float(box.conf[0])

        judgement, conf, area = get_judgement([x1, y1, x2, y2], class_id, conf)

        # 记录类型
        if class_id == 0:
            smoke_detected = True
            obj_type = "烟雾"
        else:
            fire_detected = True
            obj_type = "火焰"

        # 打印判读结果
        label = f"{judgement} ({conf:.2f})"
        print(f"  {label} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}], 面积={area:.0f}")

        # 画框 + 文字
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 添加到文本描述中
        detections_text += f"- {obj_type}：等级 {judgement}，置信度 {conf:.2f}，面积约 {area:.0f} 像素\n"

if not detected:
    print("⚠️ 未检测到任何火灾或烟雾目标，将保存原始图片。")
    cv2.putText(img, "No fire/smoke detected", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    detections_text = "未检测到任何火焰或烟雾目标。"
else:
    # 添加统计信息到文本描述
    detections_text = f"检测到火焰：{'是' if fire_detected else '否'}，烟雾：{'是' if smoke_detected else '否'}\n" + detections_text

# ========== 调用大语言模型进行高级分析 ==========
print("\n🤖 正在调用本地大模型分析火灾情况...")
llm_analysis = call_llm_for_analysis(detections_text)

if llm_analysis:
    print("\n========== 大模型分析结果 ==========")
    print(f"是否有火灾：{llm_analysis.get('hasFire', False)}")
    print(f"风险等级：{llm_analysis.get('riskLevel', 'unknown')}")
    print(f"置信度：{llm_analysis.get('confidence', 0)}")
    print(f"详细描述：{llm_analysis.get('description', '')}")

    # 将分析结果也绘制在图片上（可选）
    y_offset = 60
    cv2.putText(img, f"LLM: hasFire={llm_analysis.get('hasFire', False)}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    y_offset += 25
    cv2.putText(img, f"Risk: {llm_analysis.get('riskLevel', 'none')}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
else:
    print("⚠️ 大模型分析失败，可能 Ollama 服务未运行或模型未加载。")
    cv2.putText(img, "LLM analysis failed", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# 统一保存结果图片（使用绝对路径或脚本所在目录）
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, 'result_with_llm.jpg')
cv2.imwrite(save_path, img)
print(f"\n✅ 推理完成，结果图片保存到 {save_path}")