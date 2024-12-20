# Go Trainer (围棋练习应用)

一个现代化的围棋练习Web应用，支持智能题目识别、互动练习和围棋知识库。

## 功能特点

### 围棋题目识别
- 支持拍照上传围棋练习题
- 使用计算机视觉自动识别棋盘和棋子位置
- 在Web界面上还原棋盘布局

### 互动练习
- 支持在iPad等触屏设备上练习下棋
- 记录每一步棋的位置
- 支持悔棋功能

### 围棋知识库
- 内置围棋规则判断
- 气的计算
- 死活判断
- 基本定式库

## 技术栈

### 后端
- FastAPI
- Python 3.9+
- OpenCV (棋盘识别)
- SQLAlchemy (数据库)

### 前端
- React 18
- Ant Design 5
- TypeScript
- Canvas (棋盘渲染)

## 开发环境设置

1. 安装后端依赖
```bash
cd backend
pip install -r requirements.txt
```

2. 安装前端依赖
```bash
cd frontend
npm install
```

3. 运行开发服务器
```bash
# 后端
cd backend
uvicorn main:app --reload

# 前端
cd frontend
npm run dev
```

## 项目结构
```
go-trainer/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── models/
│   │   └── services/
│   ├── requirements.txt
│   └── main.py
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── services/
│   ├── package.json
│   └── index.html
└── README.md
```
