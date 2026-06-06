# 19. FastAPI 工程实践

> **LLM 后端标配**：FastAPI 是 LLM 应用后端的首选框架，异步、高性能、自动文档

---

## 一、核心概念

### 1. ⭐⭐ Q: FastAPI 的核心优势是什么？和 Flask/Django 的区别？

**答**：

| 特性 | FastAPI | Flask | Django |
|------|---------|-------|--------|
| 异步 | ✅ 原生 async/await | ⚠️ 需扩展 | ⚠️ 需扩展 |
| 性能 | 极高（接近 Go/Node） | 中 | 中 |
| 类型提示 | 原生支持（Pydantic） | 无 | 部分 |
| 自动文档 | ✅ Swagger + ReDoc | ❌ | 需扩展 |
| 数据验证 | ✅ Pydantic | ❌ | 需扩展 |
| WebSocket | ✅ 原生 | 需扩展 | 需扩展 |
| 依赖注入 | ✅ 原生 | ❌ | 部分 |
| 学习曲线 | 低 | 低 | 高 |

**FastAPI 为什么快**：
1. 基于 Starlette（ASGI 框架）
2. 使用 Pydantic 做数据验证（Rust 实现）
3. 原生异步 I/O
4. 自动生成路由代码

---

### 2. ⭐⭐⭐ Q: FastAPI 的依赖注入系统是怎么工作的？

**答**：

```python
from fastapi import FastAPI, Depends, HTTPException
from typing import Optional

app = FastAPI()

# 依赖函数
async def get_db():
    db = DatabaseSession()
    try:
        yield db
    finally:
        await db.close()

async def get_current_user(token: str = Depends(oauth2_scheme)):
    user = decode_token(token)
    if not user:
        raise HTTPException(status_code=401)
    return user

async def get_current_active_user(
    current_user: User = Depends(get_current_user)
):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# 使用依赖
@app.get("/users/me")
async def read_users_me(
    current_user: User = Depends(get_current_active_user),
    db: Database = Depends(get_db)
):
    return current_user

# 依赖链：
# read_users_me → get_current_active_user → get_current_user → oauth2_scheme

# 类依赖
class CommonQueryParams:
    def __init__(
        self,
        q: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ):
        self.q = q
        self.skip = skip
        self.limit = limit

@app.get("/items/")
async def read_items(params: CommonQueryParams = Depends()):
    return {"q": params.q, "skip": params.skip, "limit": params.limit}
```

---

### 3. ⭐⭐⭐ Q: 如何在 FastAPI 中实现中间件？

**答**：

```python
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import uuid

app = FastAPI()

# 1. 内置中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 自定义中间件
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} "
        f"status={response.status_code} "
        f"duration={duration:.3f}s"
    )
    
    return response

# 3. 错误处理中间件
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "request_id": getattr(request.state, "request_id", None)
        }
    )

# 4. 限流中间件
from collections import defaultdict
import asyncio

class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
    
    async def check(self, client_ip: str) -> bool:
        now = time.time()
        window_start = now - self.window_seconds
        
        # 清理过期记录
        self.requests[client_ip] = [
            t for t in self.requests[client_ip] if t > window_start
        ]
        
        if len(self.requests[client_ip]) >= self.max_requests:
            return False
        
        self.requests[client_ip].append(now)
        return True

rate_limiter = RateLimiter(max_requests=100, window_seconds=60)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    
    if not await rate_limiter.check(client_ip):
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded"}
        )
    
    return await call_next(request)
```

---

### 4. ⭐⭐⭐ Q: FastAPI 如何处理 WebSocket？

**答**：

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List
import json

app = FastAPI()

# WebSocket 连接管理器
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket)
    
    try:
        while True:
            # 接收消息
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # 处理消息
            if message["type"] == "chat":
                response = await process_chat(message["content"])
                await manager.send_personal_message(
                    json.dumps({"type": "chat", "content": response}),
                    websocket
                )
            
            elif message["type"] == "broadcast":
                await manager.broadcast(
                    json.dumps({
                        "type": "broadcast",
                        "from": client_id,
                        "content": message["content"]
                    })
                )
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(
            json.dumps({
                "type": "system",
                "content": f"{client_id} disconnected"
            })
        )

# 流式 LLM 响应
@app.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_text()
            
            # 流式调用 LLM
            async for chunk in llm.stream(data):
                await websocket.send_text(chunk)
            
            await websocket.send_text("[DONE]")
    
    except WebSocketDisconnect:
        pass
```

---

### 5. ⭐⭐⭐ Q: FastAPI 如何实现后台任务？

**答**：

```python
from fastapi import FastAPI, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
import asyncio

app = FastAPI()

# 方式一：BackgroundTasks（简单任务）
def send_email(email: str, message: str):
    # 同步任务
    smtp_client.send(email, message)

def log_action(user_id: int, action: str):
    # 同步任务
    db.execute(f"INSERT INTO logs ...")

@app.post("/submit/")
async def submit(
    data: Data,
    background_tasks: BackgroundTasks
):
    background_tasks.add_task(send_email, data.email, "Submitted!")
    background_tasks.add_task(log_action, data.user_id, "submit")
    
    return {"message": "Processing in background"}

# 方式二：asyncio.create_task（异步任务）
async def process_large_file(file_path: str):
    async with aiofiles.open(file_path) as f:
        content = await f.read()
    # 处理...
    await save_to_db(result)

@app.post("/upload/")
async def upload(file: UploadFile):
    file_path = f"/tmp/{file.filename}"
    
    # 保存文件
    async with aiofiles.open(file_path, 'wb') as f:
        await f.write(await file.read())
    
    # 后台处理
    asyncio.create_task(process_large_file(file_path))
    
    return {"message": "File uploaded, processing in background"}

# 方式三：Celery（分布式任务队列）
from celery import Celery

celery_app = Celery(
    "tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1"
)

@celery_app.task
def heavy_computation(data: dict) -> dict:
    # 耗时计算
    result = do_computation(data)
    return result

@app.post("/compute/")
async def compute(data: dict):
    task = heavy_computation.delay(data)
    return {"task_id": task.id, "status": "queued"}

@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    task = heavy_computation.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status": task.status,
        "result": task.result if task.ready() else None
    }
```

---

### 6. ⭐⭐ Q: FastAPI 的安全认证怎么实现？

**答**：

```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta

app = FastAPI()

# 密码哈希
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# JWT 配置
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# 创建 JWT Token
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# 验证 Token
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = get_user(username)
    if user is None:
        raise credentials_exception
    
    return user

# 登录端点
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

# 受保护的端点
@app.get("/users/me")
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user
```

---

### 7. ⭐⭐⭐ Q: 如何组织大型 FastAPI 项目？

**答**：

```
project/
├── app/
│   ├── __init__.py
│   ├── main.py                # 应用入口
│   ├── config.py              # 配置管理
│   ├── database.py            # 数据库连接
│   │
│   ├── api/                   # API 路由
│   │   ├── __init__.py
│   │   ├── deps.py            # 公共依赖
│   │   ├── v1/                # API v1
│   │   │   ├── __init__.py
│   │   │   ├── router.py      # v1 路由汇总
│   │   │   ├── auth.py        # 认证端点
│   │   │   ├── users.py       # 用户端点
│   │   │   └── items.py       # 业务端点
│   │   └── v2/                # API v2
│   │
│   ├── core/                  # 核心模块
│   │   ├── __init__.py
│   │   ├── security.py        # 认证/授权
│   │   ├── exceptions.py      # 自定义异常
│   │   └── middleware.py      # 中间件
│   │
│   ├── models/                # 数据模型
│   │   ├── __init__.py
│   │   ├── domain.py          # 领域模型
│   │   └── schemas.py         # Pydantic schemas
│   │
│   ├── services/              # 业务逻辑
│   │   ├── __init__.py
│   │   ├── user_service.py
│   │   └── llm_service.py
│   │
│   ├── repositories/          # 数据访问层
│   │   ├── __init__.py
│   │   ├── user_repo.py
│   │   └── item_repo.py
│   │
│   └── utils/                 # 工具函数
│       ├── __init__.py
│       └── helpers.py
│
├── tests/                     # 测试
│   ├── conftest.py
│   ├── test_api/
│   ├── test_services/
│   └── test_repositories/
│
├── alembic/                   # 数据库迁移
│   └── versions/
│
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

**main.py**：
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.router import api_router
from app.core.middleware import setup_middleware
from app.config import settings

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # 中间件
    setup_middleware(app)
    
    # 路由
    app.include_router(api_router, prefix="/api/v1")
    
    # 健康检查
    @app.get("/health")
    async def health():
        return {"status": "ok"}
    
    return app

app = create_app()
```

**api/v1/router.py**：
```python
from fastapi import APIRouter
from app.api.v1 import auth, users, items

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(items.router, prefix="/items", tags=["items"])
```

---

### 8. ⭐⭐ Q: FastAPI 如何集成 SQLAlchemy？

**答**：

```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String
from fastapi import Depends

# 数据库配置
DATABASE_URL = "postgresql+asyncpg://user:password@localhost/dbname"

engine = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

# 模型
class UserDB(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)

# 依赖注入
async def get_db():
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

# Repository 模式
class UserRepository:
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_by_id(self, user_id: int) -> UserDB:
        result = await self.db.execute(
            select(UserDB).where(UserDB.id == user_id)
        )
        return result.scalar_one_or_none()
    
    async def create(self, user_data: dict) -> UserDB:
        user = UserDB(**user_data)
        self.db.add(user)
        await self.db.flush()
        return user

# API 端点
@app.post("/users/", response_model=UserResponse)
async def create_user(
    user: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    repo = UserRepository(db)
    db_user = await repo.create(user.model_dump())
    return db_user
```

---

### 9. ⭐⭐⭐ Q: FastAPI 如何实现流式响应（SSE）？

**答**：

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio
import json

app = FastAPI()

# Server-Sent Events (SSE)
async def event_generator():
    while True:
        # 发送数据
        data = {"time": datetime.now().isoformat()}
        yield f"data: {json.dumps(data)}\n\n"
        await asyncio.sleep(1)

@app.get("/events")
async def sse_endpoint():
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

# LLM 流式响应
async def llm_stream(prompt: str):
    async for chunk in llm.astream(prompt):
        yield f"data: {json.dumps({'content': chunk})}\n\n"
    yield "data: [DONE]\n\n"

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    return StreamingResponse(
        llm_stream(request.prompt),
        media_type="text/event-stream"
    )

# 文件流式下载
@app.get("/download/{file_id}")
async def download_file(file_id: str):
    async def file_generator():
        async with aiofiles.open(f"files/{file_id}", "rb") as f:
            while chunk := await f.read(8192):
                yield chunk
    
    return StreamingResponse(
        file_generator(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={file_id}"}
    )
```
