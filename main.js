const { app, BrowserWindow } = require('electron');
const path = require('path');

// 引入启动 Flask 的文件
const flaskProcess = require('./start-flask.js');

// 设置 Flask 服务器的 URL
const FLASK_URL = 'http://127.0.0.1:5050';

function loadURLWithRetry(win, url, retryCount = 5) {
    win.loadURL(url).catch((err) => {
        if (retryCount > 0) {
            console.log(`Retrying to load URL... Attempts left: ${retryCount}`);
            setTimeout(() => loadURLWithRetry(win, url, retryCount - 1), 2000); // 2秒后重试
        } else {
            console.error("Failed to load URL:", err);
        }
    });
}

function createWindow() {
    const win = new BrowserWindow({
        width: 800,
        height: 600,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false,
        }
    });

    loadURLWithRetry(win, FLASK_URL);
}

// Electron 应用启动时创建窗口
app.whenReady().then(() => {
    createWindow();

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
});

// 关闭所有窗口时退出应用
app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        flaskProcess.kill();  // 停止 Flask 进程
        app.quit();
    }
});
// app.on('ready', createWindow);