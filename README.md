# 🦁 保护动物智能检测系统

基于 YOLOv26 的珍稀保护动物智能检测与识别平台

## 🌐 在线体验

点击链接即可体验：**[保护动物智能检测系统](https://resulting-apricot-jyhu3wul.pages.edgeone.ai)**

> ⚠️ 说明：当前为静态前端展示页面。如需体验完整检测功能（实时视频检测、图片上传检测等），请克隆项目到本地运行。

## 📖 项目简介

本项目基于 YOLOv8 深度学习算法，构建了一套珍稀保护动物智能检测系统。系统能够自动识别 8 类中国珍稀保护动物，并提供实时检测、图片检测、批量检测等多种交互方式。

### 🐾 支持的动物类别

| 中文名称 | 英文名称 | 保护级别 |
|---------|---------|---------|
| 亚洲象 | Asian elephant | 国家一级保护动物 |
| 白鹭 | egret | 国家三有保护动物 |
| 金丝猴 | Golden Monkey | 国家一级保护动物 |
| 红腹锦鸡 | golden pheasant | 国家二级保护动物 |
| 小麂 | muntjac | 国家三有保护动物 |
| 猫头鹰 | owl | 国家二级保护动物 |
| 小熊猫 | red panda | 国家二级保护动物 |
| 野骆驼 | wild camel | 国家一级保护动物 |

## 🛠️ 技术栈

| 技术 | 用途 |
|------|------|
| **Python** | 后端开发语言 |
| **YOLOv8** | 目标检测算法 |
| **PyTorch** | 深度学习框架 |
| **OpenCV** | 图像处理与视频流 |
| **Flask** | Web 后端服务 |
| **HTML/CSS/JavaScript** | 前端界面 |
| **EdgeOne Pages** | 静态页面部署 |

## 📁 项目结构



## ✨ 核心功能

- ✅ **实时视频检测**：连接摄像头，实时识别画面中的保护动物
- ✅ **单图检测**：上传图片，自动识别并标注动物位置
- ✅ **批量检测**：多张图片批量处理，导出检测结果
- ✅ **动物科普**：点击动物卡片，查看详细介绍和图片
- ✅ **统计信息**：实时显示检测数量、置信度等数据

## 🚀 本地运行

### 1. 克隆项目

```bash
git clone https://github.com/ayn-yyh/animal-detection.git
cd animal-detection
2. 安装依赖
bash
pip install -r requirements.txt
3. 运行项目
bash
python app.py
4. 访问页面
打开浏览器访问：http://localhost:5000

📊 模型评估
指标	数值
检测类别	8 类保护动物
数据集规模	每类 200 张左右
置信度阈值	0.25（可调）
IOU 阈值	0.45（可调）
👨‍💻 作者
GitHub: @ayn-yyh

📄 许可证
MIT License

text

---

## 📝 第三步：提交并保存

1. 滚动到页面底部
2. 在 "Commit new file" 区域：
   - 写提交信息：`add README`
   - 选择 **Commit directly to the main branch**
3. 点击 **Commit new file**

---

## ✅ 第四步：在 README 中插入在线体验链接

如果你想让在线体验链接更醒目，在 README 开头加上：

```markdown
## 🌐 在线体验

👉 [点击这里体验系统](https://resulting-apricot-jyhu3wul.pages.edgeone.ai)
📌 最终效果
你的 GitHub 仓库首页会显示：

✅ 项目标题和描述

✅ 在线体验链接（面试官点击即可访问）

✅ 技术栈说明

✅ 功能列表

✅ 本地运行步骤

✅ 项目结构
