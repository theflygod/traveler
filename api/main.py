"""
FastAPI主应用
提供Web服务和静态文件
"""

import os
import sys
from pathlib import Path

# 确保项目根目录在Python路径中
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from config.logging_config import setup_logging
from api.routes import router

setup_logging(log_level='INFO')

app = FastAPI(
    title="旅游知识库系统",
    description="智能旅游问答系统",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# 确保templates目录存在
templates_dir = project_root / 'templates'
templates_dir.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(templates_dir))


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """首页"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/import", response_class=HTMLResponse)
async def import_page(request: Request):
    """导入页面"""
    return templates.TemplateResponse("import.html", {"request": request})


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """聊天页面"""
    return templates.TemplateResponse("chat.html", {"request": request})


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host='0.0.0.0',
        port=8000,
        reload=True,
        reload_dirs=[str(project_root)]
    )
