function updateTime() {
    document.getElementById('currentTime').textContent = new Date().toLocaleString('zh-CN');
}
setInterval(updateTime, 1000);
updateTime();

function loadHistory() {
    const history = JSON.parse(localStorage.getItem('fireHistory') || '[]');
    const list = document.getElementById('historyList');

    if (history.length === 0) {
        list.innerHTML = '<p class="empty-msg">暂无历史记录</p>';
        return;
    }

    list.innerHTML = history.map(item => `
        <div class="history-item">
            <img src="${item.image}" alt="检测图">
            <div class="history-info">
                <h3>${item.hasFire ? '🔥 检测到火灾' : '✅ 未检测到火灾'}</h3>
                <p>${item.description}</p>
                <p>⏰ ${item.time} | 风险等级：${item.riskLevel.toUpperCase()}</p>
            </div>
            <span class="risk-badge risk-${item.riskLevel}">${item.riskLevel.toUpperCase()}</span>
        </div>
    `).join('');
}

loadHistory();
