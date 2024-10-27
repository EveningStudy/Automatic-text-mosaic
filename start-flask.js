const { spawn } = require('child_process');
const path = require('path');

// 使用虚拟环境中的 Python 解释器的绝对路径
const pythonPath = path.join('D:\\Python_Project\\Masaike\\.venv\\Scripts\\python.exe'); // Windows 示例

// 指定 Flask 应用的绝对路径
const flaskScriptPath = path.join('D:\\Python_Project\\Masaike\\app.py');

// 启动 Flask 应用
const flaskProcess = spawn(pythonPath, [flaskScriptPath]);

flaskProcess.stdout.on('data', (data) => {
    console.log(`Flask: ${data}`);
});

flaskProcess.stderr.on('data', (data) => {
    console.error(`Flask error: ${data}`);
});

// 关闭进程时的处理逻辑
flaskProcess.on('close', (code) => {
    console.log(`Flask process exited with code ${code}`);
});

module.exports = flaskProcess;