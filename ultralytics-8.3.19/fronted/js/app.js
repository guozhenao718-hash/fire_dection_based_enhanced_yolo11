const API_URL = 'http://localhost:5000/api';

let riskChart = null;

// 初始化时间
function updateTime() {
    const now = new Date();
    document.getElementById('currentTime').textContent = now.toLocaleString('zh-CN');
}
setInterval(updateTime, 1000);
updateTime();

// 上传逻辑
const uploadBox = document.getElementById('uploadBox');
const fileInput = document.getElementById('fileInput');

uploadBox.addEventListener('click', () => fileInput.click());

uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#ff6b6b';
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.style.borderColor = '#e94560';
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    if (e.dataTransfer.files[0]) handleImage(e.dataTransfer.files[0]);
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files[0]) handleImage(e.target.files[0]);
});

async function handleImage(file) {
    const formData = new FormData();
    formData.append('image', file);

    // 显示预览
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('previewImage').src = e.target.result;
        document.getElementById('previewCard').style.display = 'block';
    };
    reader.readAsDataURL(file);

    // 上传到后端
    const resultBox = document.getElementById('resultBox');
    resultBox.innerHTML = '<p class="loading">🤖 正在调用YOLO+大模型分析中...</p>';

    try {
        const response = await fetch(`${API_URL}/analyze`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            displayResult(data);
            updateChart(data);
            saveToHistory(data);
        } else {
            resultBox.innerHTML = `<p class="loading" style="color:#ff4757;">❌ ${data.error}</p>`;
        }
    } catch (err) {
        resultBox.innerHTML = `<p class="loading" style="color:#ff4757;">❌ 请求失败：${err.message}</p>`;
    }
}

function displayResult(data) {
    const resultBox = document.getElementById('resultBox');
    const riskClass = {
        'high': 'risk-high',
        'medium': 'risk-medium',
        'low': 'risk-low',
        'none': 'risk-low'
    }[data.riskLevel] || 'risk-low';

    resultBox.innerHTML = `
        <div class="result-item ${data.hasFire ? 'has-fire-true' : 'has-fire-false'}">
            <div class="label">🔥 是否存在火灾</div>
            <div class="value">${data.hasFire ? '⚠️ 检测到火灾！' : '✅ 未检测到火灾'}</div>
        </div>
        <div class="result-item">
            <div class="label">📊 风险等级</div>
            <div class="value ${riskClass}">${data.riskLevel.toUpperCase()}</div>
        </div>
        <div class="result-item">
            <div class="label">📝 详细描述</div>
            <div class="value" style="font-size:15px;font-weight:normal;line-height:1.6;">
                ${data.description}
            </div>
        </div>
        <div class="result-item">
            <div class="label">🎯 置信度</div>
            <div class="value">${(data.confidence * 100).toFixed(1)}%</div>
        </div>
        <div class="result-item">
            <div class="label">⏰ 检测时间</div>
            <div class="value" style="font-size:14px;">${data.timestamp}</div>
        </div>
    `;

    // 更新统计
    const fireCount = data.detections.filter(d => d.type === '火焰').length;
    const smokeCount = data.detections.filter(d => d.type === '烟雾').length;
    document.getElementById('statFire').textContent = fireCount;
    document.getElementById('statSmoke').textContent = smokeCount;
    document.getElementById('statRisk').textContent = data.riskLevel.toUpperCase();
    document.getElementById('statRisk').className = `stat-value ${riskClass}`;

    // 绘制检测框到canvas
    drawDetections(data.detections);
}

function drawDetections(detections) {
    const canvas = document.getElementById('overlayCanvas');
    const ctx = canvas.getContext('2d');
    const img = document.getElementById('previewImage');

    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;

    detections.forEach(det => {
        const [x1, y1, x2, y2] = det.bbox;
        const color = det.type === '火焰' ? '#ff0000' : '#00ff00';

        ctx.strokeStyle = color;
        ctx.lineWidth = 4;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        ctx.fillStyle = color;
        ctx.font = 'bold 16px Microsoft YaHei';
        ctx.fillText(`${det.type} ${det.level}`, x1, y1 - 10);
    });
}

function updateChart(data) {
    const chartDom = document.getElementById('riskChart');
    if (!riskChart) riskChart = echarts.init(chartDom);

    const fireCount = data.detections.filter(d => d.type === '火焰').length;
    const smokeCount = data.detections.filter(d => d.type === '烟雾').length;

    const option = {
        backgroundColor: 'transparent',
        tooltip: { trigger: 'item' },
        series: [{
            type: 'pie',
            radius: ['40%', '70%'],
            avoidLabelOverlap: false,
            itemStyle: {
                borderRadius: 10,
                borderColor: '#0f0c29',
                borderWidth: 2
            },
            label: { show: true, color: '#fff', fontSize: 14 },
            data: [
                { value: fireCount, name: '🔥 火焰', itemStyle: { color: '#ff4757' } },
                { value: smokeCount, name: '💨 烟雾', itemStyle: { color: '#2ed573' } },
                { value: Math.max(0, 10 - fireCount - smokeCount), name: '✅ 安全', itemStyle: { color: '#5352ed' } }
            ]
        }]
    };

    riskChart.setOption(option);
}

function saveToHistory(data) {
    let history = JSON.parse(localStorage.getItem('fireHistory') || '[]');
    history.unshift({
        id: data.filename,
        time: data.timestamp,
        hasFire: data.hasFire,
        riskLevel: data.riskLevel,
        description: data.description,
        image: `http://localhost:5000${data.result_image}`
    });
    localStorage.setItem('fireHistory', JSON.stringify(history.slice(0, 50)));
}
