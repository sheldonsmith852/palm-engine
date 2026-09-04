# palm-engine

掌纹结构化提取引擎（OpenCV + MediaPipe）。

把一张手掌照片变成结构化数据：主线（生命/智慧/感情/事业/太阳）的清晰度与断口、进阶纹向（川字掌、双太阳线、感情线羽毛纹）、质量分，以及一张金色标注图。

本仓库是一个**纯算法引擎**，被上游 Node 服务（如 `bazi-fortune-api` 的 `/palm` 路由）通过子进程调用，自身不提供 HTTP 接口。

## 用法

```bash
pip install -r requirements.txt
python palm_read.py <图片路径> -o <输出目录>
```

执行后会在 `<输出目录>` 生成：

- `palm.json` —— 结构化分析结果
- `annotated.png` —— 金色层标注图（掌纹皱褶 + 主线/纹向标记）

Node 侧的典型调用方式：

```js
const { execFile } = require('child_process');
execFile(PYTHON, [ENGINE, imgPath, '-o', tmpDir], (err) => {
  // 读取 tmpDir/palm.json 与 tmpDir/annotated.png（PNG 转 base64 返回）
});
```

## 依赖

见 `requirements.txt`，**必须锁定 `mediapipe==0.10.14`**（该版本的 `mp.solutions.hands` 已包含手部 `.tflite` 模型，运行时无需联网下载）。

## 离线说明

MediaPipe 手部模型已打包进 wheel，在无外网环境（如国内云服务器）可完全离线运行。已在 `mediapipe==0.10.14` 上验证：模型本地加载约 0.06s，零联网。

## 结构

- `palm_read.py` —— 主入口与全部提取/检测逻辑（`analyze()` 为核心函数）
- `requirements.txt` —— 依赖清单

## 免责声明

手相/掌纹分析属民俗文化，本引擎仅供娱乐与研究参考，不构成任何医疗、健康或命运判断依据。
