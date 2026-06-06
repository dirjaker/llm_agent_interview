# 01. Python 高级编程

> **LLM 工程师必备**：Python 是 AI 工程的主力语言，高级特性在框架源码和生产代码中随处可见

---

## 一、数据结构与内置类型

### 1. ⭐ Q: list 和 tuple 的区别？什么时候用哪个？

**答**：

| 特性 | list | tuple |
|------|------|-------|
| 可变性 | 可变 | 不可变 |
| 内存 | 更大（需要额外空间扩容） | 更小 |
| 速度 | 较慢 | 较快（CPython 有优化） |
| 可哈希 | ❌ | ✅（如果元素都可哈希） |
| 用作 dict key | ❌ | ✅ |

```python
# tuple 可以作为 dict key，list 不行
cache = {}
cache[(1, 2, 3)] = "value"  # ✅
# cache[[1, 2, 3]] = "value"  # ❌ TypeError

# tuple 解包
a, b, c = (1, 2, 3)

# namedtuple —— 带字段名的 tuple
from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])
p = Point(1, 2)
print(p.x, p.y)  # 1 2
```

**什么时候用 tuple**：
- 函数返回多个值
- 作为 dict key 或 set 元素
- 不需要修改的数据
- 想表达"结构体"语义

---

### 2. ⭐⭐ Q: dict 的底层实现是什么？为什么查找是 O(1)？

**答**：

```python
# dict 底层是哈希表（Hash Table）
# Python 3.6+ 使用紧凑字典（Compact Dict）

# 哈希过程
d = {"name": "Alice", "age": 25}

# 查找 "name" 的过程：
# 1. hash("name") → 某个整数
# 2. 整数 % 表大小 → 桶索引
# 3. 在桶中找到 key → 返回 value

# 为什么 O(1)？
# 理想情况：每个桶只有一个元素 → 直接访问
# 冲突时：使用开放寻址法 → 平均仍是 O(1)

# 哈希冲突演示
class BadKey:
    def __hash__(self):
        return 42  # 所有实例哈希值相同
    def __eq__(self, other):
        return isinstance(other, BadKey)

d = {}
for i in range(1000):
    d[BadKey()] = i  # 严重哈希冲突，查找退化为 O(n)
```

**Python 3.7+ dict 保持插入顺序**：
```python
d = {}
d["c"] = 3
d["a"] = 1
d["b"] = 2
print(list(d.keys()))  # ['c', 'a', 'b'] —— 保持插入顺序
```

---

### 3. ⭐⭐ Q: set 和 frozenset 的区别？应用场景？

**答**：

```python
# set —— 可变集合
s = {1, 2, 3}
s.add(4)
s.discard(1)

# frozenset —— 不可变集合（可哈希）
fs = frozenset([1, 2, 3])
# fs.add(4)  # ❌ AttributeError

# frozenset 可以作为 dict key 或 set 元素
d = {frozenset([1, 2]): "pair"}  # ✅

# 集合运算
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}
print(a & b)   # 交集 {3, 4}
print(a | b)   # 并集 {1, 2, 3, 4, 5, 6}
print(a - b)   # 差集 {1, 2}
print(a ^ b)   # 对称差集 {1, 2, 5, 6}

# 去重（保持顺序）
def deduplicate(lst):
    seen = set()
    return [x for x in lst if not (x in seen or seen.add(x))]

deduplicate([3, 1, 2, 1, 3, 2])  # [3, 1, 2]
```

---

### 4. ⭐⭐ Q: defaultdict、OrderedDict、Counter 的使用场景？

**答**：

```python
from collections import defaultdict, OrderedDict, Counter

# defaultdict —— 自动初始化默认值
word_count = defaultdict(int)  # 默认值 0
for word in words:
    word_count[word] += 1  # 不需要先检查 key 是否存在

graph = defaultdict(list)  # 默认值 []
graph["A"].append("B")
graph["A"].append("C")
# graph = {"A": ["B", "C"]}

# Counter —— 计数器
counter = Counter("abracadabra")
print(counter)  # Counter({'a': 5, 'b': 2, 'r': 2, 'c': 1, 'd': 1})
print(counter.most_common(2))  # [('a', 5), ('b', 2)]

# 集合运算
c1 = Counter(a=3, b=1)
c2 = Counter(a=1, b=2)
print(c1 + c2)  # Counter({'a': 4, 'b': 3})
print(c1 - c2)  # Counter({'a': 2})  # 只保留正数

# OrderedDict（Python 3.7+ 普通 dict 已保持顺序，较少使用）
od = OrderedDict()
od["first"] = 1
od["second"] = 2
od.move_to_end("first")  # 移到末尾
```

---

## 二、函数式编程

### 5. ⭐⭐ Q: 装饰器的原理？如何实现一个带参数的装饰器？

**答**：

```python
import functools
import time

# 基础装饰器
def timer(func):
    @functools.wraps(func)  # 保留原函数的元信息
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.3f}s")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)

# 带参数的装饰器
def retry(max_attempts=3, delay=1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    print(f"Attempt {attempt + 1} failed: {e}")
                    time.sleep(delay * (2 ** attempt))  # 指数退避
        return wrapper
    return decorator

@retry(max_attempts=3, delay=2)
def unstable_api_call():
    # 可能失败的 API 调用
    pass

# 类装饰器
class CacheDecorator:
    def __init__(self, maxsize=128):
        self.cache = {}
        self.maxsize = maxsize
    
    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args):
            if args in self.cache:
                return self.cache[args]
            result = func(*args)
            if len(self.cache) >= self.maxsize:
                # LRU: 移除最旧的
                self.cache.pop(next(iter(self.cache)))
            self.cache[args] = result
            return result
        return wrapper

@CacheDecorator(maxsize=100)
def expensive_computation(n):
    return sum(i * i for i in range(n))
```

---

### 6. ⭐⭐ Q: 生成器和迭代器的区别？yield 的原理？

**答**：

```python
# 迭代器 —— 实现 __iter__ 和 __next__
class CountDown:
    def __init__(self, start):
        self.current = start
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current <= 0:
            raise StopIteration
        self.current -= 1
        return self.current + 1

for i in CountDown(5):
    print(i)  # 5, 4, 3, 2, 1

# 生成器 —— 用 yield 简化迭代器
def count_down(start):
    while start > 0:
        yield start
        start -= 1

for i in count_down(5):
    print(i)  # 5, 4, 3, 2, 1

# yield 的原理 —— 暂停和恢复
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

fib = fibonacci()
print(next(fib))  # 0
print(next(fib))  # 1
print(next(fib))  # 1

# 生成器表达式 —— 惰性求值
squares = (x * x for x in range(1000000))  # 不会立即计算
total = sum(squares)  # 此时才逐个计算

# yield from —— 委托生成器
def flatten(nested):
    for item in nested:
        if isinstance(item, (list, tuple)):
            yield from flatten(item)
        else:
            yield item

list(flatten([1, [2, [3, 4], 5], 6]))  # [1, 2, 3, 4, 5, 6]
```

**生成器的优势**：
- **内存效率**：不需要一次性加载所有数据
- **惰性计算**：只在需要时计算
- **无限序列**：可以表示无限序列（如斐波那契）

---

### 7. ⭐⭐ Q: 上下文管理器（with 语句）的原理？

**答**：

```python
# 方式一：实现 __enter__ 和 __exit__
class DatabaseConnection:
    def __init__(self, db_url):
        self.db_url = db_url
        self.conn = None
    
    def __enter__(self):
        self.conn = connect(self.db_url)
        return self.conn
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.conn.rollback()
        else:
            self.conn.commit()
        self.conn.close()
        return False  # 不抑制异常

with DatabaseConnection("postgresql://...") as conn:
    conn.execute("SELECT 1")

# 方式二：contextmanager 装饰器
from contextlib import contextmanager

@contextmanager
    conn.execute("SELECT 1")

# 方式二：contextmanager 装饰器
from contextlib import contextmanager

@contextmanager
def timer_context(label):
    start = time.time()
    try:
        yield  # 这里的值会传给 as 变量
    finally:
        print(f"{label} took {time.time() - start:.3f}s")

with timer_context("my_operation"):
    time.sleep(1)

# 方式三：asynccontextmanager
from contextlib import asynccontextmanager

@asynccontextmanager
async def get_db_session():
    session = AsyncSession()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()

async with get_db_session() as session:
    await session.execute(...)
```

---

## 三、面向对象

### 8. ⭐⭐ Q: Python 的 MRO（方法解析顺序）是什么？

**答**：

```python
# MRO 决定多继承时方法的查找顺序
# Python 3 使用 C3 线性化算法

class A:
    def method(self):
        print("A")

class B(A):
    def method(self):
        print("B")

class C(A):
    def method(self):
        print("C")

class D(B, C):
    pass

d = D()
d.method()  # 输出: B

# 查看 MRO
print(D.__mro__)
# (<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>)

# super() 按 MRO 顺序调用
class B(A):
    def method(self):
        print("B")
        super().method()  # 调用 MRO 中的下一个（C）

class C(A):
    def method(self):
        print("C")
        super().method()  # 调用 MRO 中的下一个（A）

D().method()  # B → C → A
```

---

### 9. ⭐⭐⭐ Q: 元类（Metaclass）是什么？什么时候用？

**答**：

```python
# 元类是"创建类的类"
# type 是最基本的元类

# 普通创建类
class MyClass:
    x = 1

# 等价于（type 创建类）
MyClass = type('MyClass', (), {'x': 1})

# 自定义元类
class RegistryMeta(type):
    """自动注册子类的元类"""
    registry = {}
    
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        if bases:  # 不注册基类本身
            mcs.registry[name] = cls
        return cls

class Plugin(metaclass=RegistryMeta):
    pass

class AudioPlugin(Plugin):
    pass

class VideoPlugin(Plugin):
    pass

print(RegistryMeta.registry)
# {'AudioPlugin': <class 'AudioPlugin'>, 'VideoPlugin': <class 'VideoPlugin'>}

# Pydantic 就大量使用元类
from pydantic import BaseModel

class User(BaseModel):  # BaseModel 的元类会自动生成 __init__, __eq__ 等
    name: str
    age: int

user = User(name="Alice", age=25)
```

**什么时候用元类**：
- ORM 框架（SQLAlchemy, Django Models）
- 序列化框架（Pydantic, Marshmallow）
- API 框架（FastAPI 依赖注入）
- 插件系统（自动注册）
- 单例模式

---

### 10. ⭐⭐ Q: descriptor 是什么？property 的底层原理？

**答**：

```python
# descriptor 是实现了 __get__, __set__, __delete__ 的对象
# property, classmethod, staticmethod 都是 descriptor

class Validated:
    """带验证的 descriptor"""
    
    def __init__(self, validator, error_msg="Invalid value"):
        self.validator = validator
        self.error_msg = error_msg
    
    def __set_name__(self, owner, name):
        self.name = name
        self.private_name = f"_{name}"
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self.private_name, None)
    
    def __set__(self, obj, value):
        if not self.validator(value):
            raise ValueError(f"{self.name}: {self.error_msg}")
        setattr(obj, self.private_name, value)

class User:
    age = Validated(lambda x: 0 < x < 150, "年龄必须在 0-150 之间")
    email = Validated(lambda x: "@" in str(x), "邮箱格式不正确")
    
   501|    def __init__(self, age, email):
        self.age = age      # 触发 __set__
        self.email = email

user = User(25, "alice@example.com")
print(user.age)  # 25

# user = User(-1, "invalid")  # ❌ ValueError: age: 年龄必须在 0-150 之间

# property 就是内置的 descriptor
class Temperature:
    def __init__(self, celsius):
        self._celsius = celsius
    
    @property
    def fahrenheit(self):
        return self._celsius * 9/5 + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value):
        self._celsius = (value - 32) * 5/9

t = Temperature(100)
print(t.fahrenheit)  # 212.0
t.fahrenheit = 32
print(t._celsius)    # 0.0
```

---

## 四、并发编程

### 11. ⭐⭐⭐ Q: asyncio 的底层原理？事件循环是怎么工作的？

**答**：

```python
import asyncio

# asyncio 的核心：事件循环（Event Loop）
# 事件循环不断检查：有没有就绪的协程？有没有就绪的 I/O？

# 协程 —— 用 async/await 定义的函数
async def fetch_data(url):
    print(f"开始请求 {url}")
    await asyncio.sleep(1)  # 模拟 I/O，让出控制权
    print(f"完成请求 {url}")
    return {"url": url, "data": "..."}

# 事件循环的工作原理（简化）：
"""
while tasks_remain:
    1. 检查就绪队列
       - 有就绪协程 → 执行到下一个 await
       - 无就绪 → 等待
    
    2. 检查 I/O 就绪
       - select/poll/epoll 检查哪些 socket 可读/可写
       - 就绪的 I/O 对应的协程加入就绪队列
    
    3. 检查定时器
       - asyncio.sleep 到期 → 对应协程加入就绪队列
"""

# 并发执行多个协程
async def main():
    # 方式一：gather —— 并发执行，收集结果
    results = await asyncio.gather(
        fetch_data("url1"),
        fetch_data("url2"),
        fetch_data("url3"),
    )
    
    # 方式二：TaskGroup（Python 3.11+）
    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(fetch_data("url1"))
        task2 = tg.create_task(fetch_data("url2"))
    results = [task1.result(), task2.result()]

asyncio.run(main())
```

**GIL 对 asyncio 的影响**：
```python
# asyncio 是单线程的，不受 GIL 影响
# 因为 await 会让出控制权，不是真正的并行计算

# CPU 密集型任务应该用 ProcessPoolExecutor
async def cpu_bound_task(data):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        ProcessPoolExecutor(),  # 多进程
        heavy_computation,
        data
    )
    return result
```

---

### 12. ⭐⭐⭐ Q: GIL 是什么？为什么有 GIL？怎么绕过？

**答**：

```python
# GIL（Global Interpreter Lock）—— 全局解释器锁
# 同一时刻只有一个线程执行 Python 字节码

# 为什么有 GIL？
# 1. 简化 CPython 内存管理（引用计数不需要加锁）
# 2. 历史原因（C 扩展兼容性）
# 3. 单线程性能最优

# GIL 的影响：
# - CPU 密集型：多线程几乎没用（甚至更慢）
# - I/O 密集型：多线程有效（I/O 时会释放 GIL）

# 绕过 GIL 的方法：

# 方法一：多进程
from multiprocessing import Pool

def cpu_heavy(n):
    return sum(i * i for i in range(n))

with Pool(4) as p:
    results = p.map(cpu_heavy, [10**7] * 4)

# 方法二：C 扩展（NumPy 等释放 GIL）
import numpy as np
arr = np.random.rand(10000, 10000)
result = np.dot(arr, arr)  # C 代码执行，释放 GIL

# 方法三：使用 nogil Python（Python 3.13+ 实验性）
# PEP 703: Making the Global Interpreter Lock Optional

# 方法四：异步 I/O（asyncio）
# 不需要多线程，单线程即可高效处理 I/O
```

---

### 13. ⭐⭐ Q: 多线程、多进程、协程怎么选？

**答**：

```python
"""
选择决策树：

任务类型是什么？
│
├── I/O 密集型（网络请求、文件读写、数据库查询）
│   ├── 数量少（<100）→ 多线程
│   └── 数量多（>100）→ 协程（asyncio）
│
└── CPU 密集型（计算、图像处理、加密）
    └── 多进程
"""

# I/O 密集型 —— 协程
async def fetch_all(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_one(session, url) for url in urls]
        return await asyncio.gather(*tasks)

# I/O 密集型 —— 多线程
from concurrent.futures import ThreadPoolExecutor

def fetch_all_threaded(urls):
    with ThreadPoolExecutor(max_workers=10) as executor:
        return list(executor.map(fetch_one, urls))

# CPU 密集型 —— 多进程
from concurrent.futures import ProcessPoolExecutor

def compute_all(data_list):
    with ProcessPoolExecutor() as executor:
        return list(executor.map(cpu_heavy, data_list))

# 性能对比（处理 1000 个网络请求）：
# 多线程 (10 workers): ~10s
# 协程: ~2s
# 多进程: ~10s（开销大，不适合 I/O）
```

---

### 14. ⭐⭐ Q: Pydantic 的工作原理？为什么 FastAPI 用它？

**答**：

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, List

# Pydantic 核心：运行时类型验证 + 序列化
class User(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    age: int = Field(..., ge=0, le=150)
    email: str
    tags: List[str] = []
    
    @validator('email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email')
        return v.lower()

# 自动验证
user = User(name="Alice", age=25, email="ALICE@EXAMPLE.COM")
print(user.name)   # Alice
print(user.email)  # alice@example.com（小写）

# User(name="", age=-1, email="invalid")  # ❌ ValidationError

# 自动序列化
print(user.model_dump())
# {'name': 'Alice', 'age': 25, 'email': 'alice@example.com', 'tags': []}

print(user.model_dump_json())
# '{"name":"Alice","age":25,"email":"alice@example.com","tags":[]}'

# 从 JSON/Dict 创建
user = User.model_validate({"name": "Bob", "age": 30, "email": "bob@test.com"})

# 为什么 FastAPI 用 Pydantic？
# 1. 自动请求体验证
# 2. 自动 OpenAPI 文档生成
# 3. 自动序列化/反序列化
# 4. 类型提示支持 IDE 补全

# FastAPI 示例
from fastapi import FastAPI

app = FastAPI()

@app.post("/users/")
async def create_user(user: User):  # 自动验证请求体
    return {"message": f"Created {user.name}"}
```

---

### 15. ⭐⭐⭐ Q: Python 的垃圾回收机制是什么？

**答**：

```python
# Python 使用引用计数 + 分代回收

# 1. 引用计数
import sys

a = [1, 2, 3]
print(sys.getrefcount(a))  # 2（a + getrefcount 的参数）

b = a  # 引用计数 +1
print(sys.getrefcount(a))  # 3

del b  # 引用计数 -1
print(sys.getrefcount(a))  # 2

# 2. 分代回收（处理循环引用）
import gc

# 三代对象
# Gen 0: 新创建的对象（最频繁回收）
# Gen 1: 存活过一次 Gen 0 回收的对象
# Gen 2: 存活过多次回收的对象（最少回收）

gc.get_threshold()  # (700, 10, 10) —— 各代阈值

# 循环引用示例
class Node:
    def __init__(self):
        self.ref = None

a = Node()
b = Node()
a.ref = b
b.ref = a  # 循环引用！

del a
del b
# 引用计数不会降到 0（因为循环引用）
# 分代回收会处理这种情况

# 3. 弱引用（不增加引用计数）
import weakref

class ExpensiveObject:
    def __init__(self, value):
        self.value = value

obj = ExpensiveObject(42)
weak_ref = weakref.ref(obj)

print(weak_ref())  # <ExpensiveObject object>
del obj
print(weak_ref())  # None（对象已被回收）

# 弱引用在缓存中很有用
class WeakCache:
    def __init__(self):
        self._cache = weakref.WeakValueDictionary()
    
    def get(self, key):
        return self._cache.get(key)
    
    def set(self, key, value):
        self._cache[key] = value
```


---

## 十二、async 高级用法

### 25. ⭐⭐⭐ Q: 如何实现异步上下文管理器（__aenter__/__aexit__）？

**答**：

```python
# 异步上下文管理器 —— 实现 __aenter__ 和 __aexit__（均为协程）
import asyncio
import aiohttp

class AsyncHTTPClient:
    """异步 HTTP 客户端上下文管理器"""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(base_url=self.base_url)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
        return False  # 不抑制异常

    async def get(self, path: str):
        async with self.session.get(path) as resp:
            return await resp.json()

# 使用方式
async def main():
    async with AsyncHTTPClient("https://api.example.com") as client:
        data = await client.get("/users")
        print(data)

# 使用 @asynccontextmanager 简化
from contextlib import asynccontextmanager

@asynccontextmanager
async def managed_lock(lock: asyncio.Lock):
    """带超时的异步锁"""
    acquired = await asyncio.wait_for(lock.acquire(), timeout=5.0)
    try:
        yield lock
    finally:
        if acquired:
            lock.release()

async def critical_section():
    lock = asyncio.Lock()
    async with managed_lock(lock) as l:
        await asyncio.sleep(0.1)  # 临界区操作
```

---

### 26. ⭐⭐⭐ Q: asyncio.TaskGroup 的用法和优势？

**答**：

```python
# TaskGroup —— Python 3.11+ 引入，替代 asyncio.gather
# 核心优势：自动错误传播 + 结构化并发

import asyncio

async def fetch(url: str) -> str:
    await asyncio.sleep(0.1)  # 模拟网络请求
    return f"Response from {url}"

# 使用 TaskGroup（推荐）
async def fetch_all_taskgroup(urls: list[str]) -> list[str]:
    results = []
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(fetch(url)) for url in urls]
    # TaskGroup 退出时，所有任务已完成
    results = [t.result() for t in tasks]
    return results

# 与 asyncio.gather 的区别
async def fetch_all_gather(urls: list[str]) -> list[str]:
    # gather 默认不会在第一个异常时取消其他任务
    return await asyncio.gather(*[fetch(url) for url in urls])

# TaskGroup 的异常处理
async def risky_task(fail: bool):
    if fail:
        raise ValueError("Task failed!")
    await asyncio.sleep(0.1)
    return "ok"

async def demo_taskgroup_error():
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(risky_task(fail=False))
            tg.create_task(risky_task(fail=True))  # 会触发异常
            tg.create_task(risky_task(fail=False))
    except* ValueError as eg:
        # Python 3.11 ExceptionGroup 语法
        for exc in eg.exceptions:
            print(f"Caught: {exc}")

asyncio.run(demo_taskgroup_error())
```

---

### 27. ⭐⭐⭐ Q: 异步生成器（async generator）的使用场景？

**答**：

```python
# 异步生成器 —— async def + yield
# 适用于：流式数据、分页 API、数据库游标

import asyncio
from collections.abc import AsyncIterator

# 示例 1：流式读取大文件
async def async_read_lines(filepath: str) -> AsyncIterator[str]:
    with open(filepath, "r") as f:
        for line in f:
            await asyncio.sleep(0)  # 让出控制权
            yield line.strip()

# 示例 2：分页 API
async def fetch_all_pages(base_url: str) -> AsyncIterator[dict]:
    page = 1
    while True:
        # 模拟 API 调用
        data = {"items": [f"item_{page}_{i}" for i in range(10)], "has_more": page < 5}
        for item in data["items"]:
            yield item
        if not data["has_more"]:
            break
        page += 1

# 示例 3：带速率限制的异步生成器
async def rate_limited(items: list, delay: float = 0.1) -> AsyncIterator:
    for item in items:
        await asyncio.sleep(delay)
        yield item

# 使用 async for 消费
async def main():
    # 消费异步生成器
    async for page in fetch_all_pages("https://api.example.com"):
        print(page)

    # 异步列表推导
    results = [item async for item in rate_limited([1, 2, 3, 4, 5])]

    # asyncitertools 风格
    async def async_enumerate(aiterable, start=0):
        n = start
        async for item in aiterable:
            yield n, item
            n += 1

    async for idx, val in async_enumerate(rate_limited(["a", "b", "c"])):
        print(f"{idx}: {val}")

asyncio.run(main())
```

---

### 28. ⭐⭐⭐ Q: asyncio.Queue 实现生产者-消费者模式？

**答**：

```python
import asyncio
import random

async def producer(queue: asyncio.Queue, name: str, count: int):
    """生产者：生成任务放入队列"""
    for i in range(count):
        task = f"{name}_task_{i}"
        await asyncio.sleep(random.uniform(0.01, 0.1))  # 模拟生产耗时
        await queue.put(task)
        print(f"[Producer {name}] produced: {task}")
    # 发送结束信号
    await queue.put(None)

async def consumer(queue: asyncio.Queue, name: str):
    """消费者：从队列取出任务处理"""
    while True:
        task = await queue.get()
        if task is None:
            # 通知其他消费者也结束
            await queue.put(None)
            break
        await asyncio.sleep(random.uniform(0.05, 0.15))  # 模拟处理耗时
        print(f"[Consumer {name}] processed: {task}")
        queue.task_done()

async def main():
    queue = asyncio.Queue(maxsize=10)  # 有界队列，控制背压

    # 启动 2 个生产者 + 3 个消费者
    producers = [
        asyncio.create_task(producer(queue, f"P{i}", 4))
        for i in range(2)
    ]
    consumers = [
        asyncio.create_task(consumer(queue, f"C{i}"))
        for i in range(3)
    ]

    # 等待所有生产者完成
    await asyncio.gather(*producers)
    # 等待队列被完全消费
    await queue.join()
    # 消费者会通过 None 信号自行退出

asyncio.run(main())

# PriorityQueue —— 优先级队列
async def priority_demo():
    pq = asyncio.PriorityQueue()

    # 放入 (优先级, 数据) 元组，数字越小优先级越高
    await pq.put((3, "low priority"))
    await pq.put((1, "high priority"))
    await pq.put((2, "medium priority"))

    while not pq.empty():
        priority, task = await pq.get()
        print(f"[P{priority}] {task}")
    # 输出顺序: high → medium → low
```

---

## 十三、Pydantic V2

### 29. ⭐⭐⭐ Q: model_validator 和 field_validator 的区别？

**答**：

```python
from pydantic import BaseModel, field_validator, model_validator

class User(BaseModel):
    name: str
    password: str
    password_confirm: str

    # field_validator —— 验证单个字段
    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("name 不能为空")
        return v.strip()  # 返回处理后的值

    # mode="before" —— 在类型转换之前验证
    @field_validator("password", mode="before")
    @classmethod
    def password_min_length(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("密码长度至少 8 位")
        return v

    # model_validator —— 验证多个字段之间的关系
    @model_validator(mode="after")
    def passwords_match(self) -> "User":
        if self.password != self.password_confirm:
            raise ValueError("两次密码不一致")
        return self

# mode="before" 的 model_validator 接收原始 dict
class Config(BaseModel):
    host: str
    port: int

    @model_validator(mode="before")
    @classmethod
    def parse_connection_string(cls, data: dict) -> dict:
        # 支持 "host:port" 格式的输入
        if "connection_string" in data:
            host, port = data.pop("connection_string").split(":")
            data["host"] = host
            data["port"] = int(port)
        return data

# 使用
config = Config(connection_string="localhost:8080")
print(config)  # host='localhost' port=8080
```

---

### 30. ⭐⭐⭐ Q: computed_field 的用途？

**答**：

```python
from pydantic import BaseModel, computed_field
from datetime import datetime

class Order(BaseModel):
    items: list[dict]  # [{"name": "item", "price": 10.0, "qty": 2}]
    tax_rate: float = 0.08

    @computed_field
    @property
    def subtotal(self) -> float:
        """计算字段，自动包含在序列化输出中"""
        return sum(item["price"] * item["qty"] for item in self.items)

    @computed_field
    @property
    def tax(self) -> float:
        return self.subtotal * self.tax_rate

    @computed_field
    @property
    def total(self) -> float:
        return self.subtotal + self.tax

order = Order(
    items=[
        {"name": "Python Book", "price": 49.99, "qty": 1},
        {"name": "USB Cable", "price": 9.99, "qty": 3},
    ]
)

print(order.model_dump())
# {
#   'items': [...],
#   'tax_rate': 0.08,
#   'subtotal': 79.96,    ← computed_field 自动出现
#   'tax': 6.3968,
#   'total': 86.3568,
# }

# 与 @property 的区别：computed_field 会参与序列化
# 普通 @property 不会出现在 model_dump() 中
```

---

### 31. ⭐⭐⭐ Q: model_config 和 BaseSettings 的用法？

**答**：

```python
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings

# model_config —— 控制模型行为
class StrictModel(BaseModel):
    model_config = ConfigDict(
        strict=True,            # 严格模式，不做类型强制转换
        frozen=True,            # 不可变（类似 frozen dataclass）
        str_strip_whitespace=True,  # 自动去除字符串首尾空白
        validate_default=True,      # 也验证默认值
        extra="forbid",             # 禁止额外字段
    )
    name: str
    value: int = 0

# strict 模式下不会自动转换
try:
    StrictModel(name="test", value="123")  # ❌ 严格模式不接受字符串
except Exception as e:
    print(e)

StrictModel(name="test", value=123)  # ✅

# frozen 模式下不可修改
m = StrictModel(name="test")
try:
    m.name = "other"  # ❌ ValidationError
except Exception as e:
    print(e)

# BaseSettings —— 从环境变量读取配置
class AppSettings(BaseSettings):
    """自动从环境变量 / .env 文件读取"""
    model_config = ConfigDict(
        env_file=".env",
        env_prefix="APP_",     # 环境变量前缀
        case_sensitive=False,
    )

    database_url: str = "sqlite:///default.db"
    redis_host: str = "localhost"
    redis_port: int = 6379
    debug: bool = False
    secret_key: str = Field(..., min_length=16)  # 必填

# 读取 APP_DATABASE_URL, APP_REDIS_HOST 等环境变量
# settings = AppSettings()  # 如果缺少 SECRET_KEY 会报错

# 嵌套 Settings
class DatabaseSettings(BaseSettings):
    host: str = "localhost"
    port: int = 5432
    name: str = "mydb"

class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", env_prefix="APP_")
    database: DatabaseSettings = DatabaseSettings()
    debug: bool = False

# 环境变量 APP_DATABASE__HOST=192.168.1.1 → database.host = "192.168.1.1"
```

---

### 32. ⭐⭐⭐ Q: Pydantic V2 与 FastAPI 的最佳实践？

**答**：

```python
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field, EmailStr, computed_field
from datetime import datetime

app = FastAPI()

# 请求模型 —— 用于验证输入
class UserCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=50, examples=["Alice"])
    email: EmailStr
    age: int = Field(..., ge=0, le=150)

# 响应模型 —— 控制输出，隐藏敏感字段
class UserResponse(BaseModel):
    id: int
    name: str
    email: EmailStr
    created_at: datetime

    @computed_field
    @property
    def display_name(self) -> str:
        return f"{self.name} <{self.email}>"

# 更新模型 —— 所有字段可选
class UserUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=50)
    email: EmailStr | None = None
    age: int | None = Field(None, ge=0, le=150)

# 数据库模型
class UserDB(UserCreate):
    id: int
    created_at: datetime = Field(default_factory=datetime.now)

    model_config = {"from_attributes": True}  # 支持 ORM 对象转换

# FastAPI 路由
@app.post("/users", response_model=UserResponse, status_code=201)
async def create_user(user: UserCreate):
    """请求体自动验证，response_model 自动过滤输出字段"""
    db_user = UserDB(id=1, **user.model_dump())
    return db_user  # 自动转换为 UserResponse

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: int):
    user = UserDB(id=user_id, name="Alice", email="alice@example.com", age=25)
    return user

@app.patch("/users/{user_id}", response_model=UserResponse)
async def update_user(user_id: int, update: UserUpdate):
    # model_dump(exclude_unset=True) 只返回显式设置的字段
    update_data = update.model_dump(exclude_unset=True)
    if not update_data:
        raise HTTPException(400, "No fields to update")
    # ... 执行更新
    user = UserDB(id=user_id, name="Updated", email="u@example.com", age=30)
    return user

# 依赖注入中使用 Pydantic
class PaginationParams(BaseModel):
    page: int = Field(1, ge=1)
    size: int = Field(20, ge=1, le=100)

    @computed_field
    @property
    def offset(self) -> int:
        return (self.page - 1) * self.size

async def get_pagination(
    page: int = 1, size: int = 20
) -> PaginationParams:
    return PaginationParams(page=page, size=size)

@app.get("/items")
async def list_items(pagination: PaginationParams = Depends(get_pagination)):
    return {"page": pagination.page, "offset": pagination.offset}
```

---

## 十四、Type Hints 高级

### 33. ⭐⭐⭐ Q: Protocol 的结构化子类型（Structural Subtyping）？

**答**：

```python
from typing import Protocol, runtime_checkable

# Protocol —— 定义接口，不需要显式继承
@runtime_checkable  # 使其可以在运行时用 isinstance 检查
class Drawable(Protocol):
    def draw(self) -> str: ...
    @property
    def color(self) -> str: ...

# 不需要继承 Drawable，只要实现了 draw() 和 color 属性即可
class Circle:
    def __init__(self, radius: float, color: str):
        self.radius = radius
        self._color = color

    def draw(self) -> str:
        return f"Drawing circle with radius {self.radius}"

    @property
    def color(self) -> str:
        return self._color

class Square:
    def __init__(self, side: float):
        self.side = side

    def draw(self) -> str:
        return f"Drawing square with side {self.side}"

    @property
    def color(self) -> str:
        return "black"

# Circle 和 Square 都满足 Drawable Protocol（鸭子类型）
def render(shape: Drawable) -> None:
    print(f"{shape.color}: {shape.draw()}")

render(Circle(5, "red"))    # ✅
render(Square(3))           # ✅

# 运行时检查
print(isinstance(Circle(1, "blue"), Drawable))  # True
print(isinstance("hello", Drawable))            # False

# 与 ABC 的区别
from abc import ABC, abstractmethod

class AbstractShape(ABC):
    @abstractmethod
    def draw(self) -> str: ...

# class BadShape(AbstractShape):  # ❌ 必须继承才能检查
#     pass

# Protocol: 不需要继承，任何实现了同名方法的类都满足
# ABC: 必须显式继承，是名义子类型（Nominal Subtyping）
```

---

### 34. ⭐⭐⭐ Q: TypeVar 和 Generic 实现泛型？

**答**：

```python
from typing import TypeVar, Generic, Sequence

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")

# 泛型栈
class Stack(Generic[T]):
    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        return self._items.pop()

    def peek(self) -> T:
        return self._items[-1]

    def __len__(self) -> int:
        return len(self._items)

# 使用泛型
int_stack: Stack[int] = Stack()
int_stack.push(1)
int_stack.push(2)
# int_stack.push("hello")  # type checker 会报错

# 泛型字典
class TypedMap(Generic[K, V]):
    def __init__(self) -> None:
        self._data: dict[K, V] = {}

    def get(self, key: K) -> V | None:
        return self._data.get(key)

    def set(self, key: K, value: V) -> None:
        self._data[key] = value

# TypeVar 约束
Numeric = TypeVar("Numeric", int, float, complex)

def add(a: Numeric, b: Numeric) -> Numeric:
    return a + b  # 只接受 int, float, complex

# TypeVar bound
class Comparable(Protocol):
    def __lt__(self, other) -> bool: ...

C = TypeVar("C", bound=Comparable)

def find_min(items: Sequence[C]) -> C:
    return min(items)

# 自定义泛型容器
from typing import Iterator

class PaginatedResult(Generic[T]):
    def __init__(self, items: list[T], total: int, page: int, size: int):
        self.items = items
        self.total = total
        self.page = page
        self.size = size

    @property
    def has_next(self) -> bool:
        return self.page * self.size < self.total

    def __iter__(self) -> Iterator[T]:
        return iter(self.items)

# 使用
def get_users(page: int = 1) -> PaginatedResult["User"]:
    ...
```

---

### 35. ⭐⭐⭐ Q: Annotated 添加元数据？

**答**：

```python
from typing import Annotated
from pydantic import BaseModel, Field

# Annotated —— 为类型添加运行时元数据
# 语法: Annotated[BaseType, metadata1, metadata2, ...]

# 示例 1：与 Pydantic 结合
class User(BaseModel):
    # Field 的约束实际就是 Annotated 的应用
    name: Annotated[str, Field(min_length=1, max_length=50)]
    age: Annotated[int, Field(ge=0, le=150)]
    email: Annotated[str, Field(pattern=r"^[\w.-]+@[\w.-]+\.\w+$")]

# 示例 2：FastAPI 中的参数验证
from fastapi import FastAPI, Query, Path, Body

app = FastAPI()

UserId = Annotated[int, Path(ge=1, description="用户 ID")]
SearchQuery = Annotated[str, Query(min_length=1, max_length=100)]

@app.get("/users/{user_id}")
async def get_user(user_id: UserId, q: SearchQuery | None = None):
    return {"user_id": user_id, "q": q}

# 示例 3：自定义验证器元数据
from dataclasses import dataclass

class Positive:
    """标记正数"""
    pass

class NonEmpty:
    """标记非空字符串"""
    pass

PositiveInt = Annotated[int, Positive()]
NonEmptyStr = Annotated[str, NonEmpty()]

# 示例 4：用于文档/序列化的元数据
SerializedField = Annotated[str, {"json_key": "field_name", "sensitive": True}]

# 示例 5：类型别名组合
from datetime import datetime

Timestamp = Annotated[float, Field(description="Unix timestamp")]
RequestId = Annotated[str, Field(min_length=16, max_length=64)]
HttpUrl = Annotated[str, Field(pattern=r"https?://.*")]

class LogEntry(BaseModel):
    timestamp: Timestamp
    request_id: RequestId
    url: HttpUrl
    message: str
```

---

### 36. ⭐⭐⭐ Q: ParamSpec 实现装饰器类型标注？

**答**：

```python
from typing import TypeVar, ParamSpec
from functools import wraps
import time
import logging

P = ParamSpec("P")  # 捕获参数规格
R = TypeVar("R")    # 捕获返回类型

# ParamSpec 让装饰器保留原函数的签名
def timer(func: "Callable[P, R]") -> "Callable[P, R]":
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper

@timer
def process_data(data: list[int], threshold: float = 0.5) -> dict:
    return {"count": len(data), "mean": sum(data) / len(data)}

# 类型检查器知道 process_data 的签名仍然是 (list[int], float) -> dict
result = process_data([1, 2, 3], threshold=0.3)

# 示例 2：带参数的装饰器
def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
) -> "Callable[[Callable[P, R]], Callable[P, R]]":
    def decorator(func: "Callable[P, R]") -> "Callable[P, R]":
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            last_exc: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay * (2 ** attempt))
            raise last_exc  # type: ignore
        return wrapper
    return decorator

@retry(max_attempts=3, delay=0.5)
def unreliable_api(url: str, timeout: float = 10.0) -> dict:
    """类型检查器知道签名是 (str, float) -> dict"""
    import random
    if random.random() < 0.7:
        raise ConnectionError("Failed")
    return {"status": "ok"}

# 示例 3：日志装饰器
def log_call(
    logger: logging.Logger,
    level: int = logging.INFO,
) -> "Callable[[Callable[P, R]], Callable[P, R]]":
    def decorator(func: "Callable[P, R]") -> "Callable[P, R]":
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            logger.log(level, f"Calling {func.__name__}({args}, {kwargs})")
            result = func(*args, **kwargs)
            logger.log(level, f"{func.__name__} returned {result}")
            return result
        return wrapper
    return decorator

logger = logging.getLogger(__name__)

@log_call(logger, level=logging.DEBUG)
def calculate(x: int, y: int) -> int:
    return x + y

# 与 Concatenate 结合（高级用法）
from typing import Concatenate, Callable

def with_context(
    func: "Callable[Concatenate[str, P], R]"
) -> "Callable[P, R]":
    """自动注入第一个参数"""
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        context = "auto_generated_context"
        return func(context, *args, **kwargs)
    return wrapper

@with_context
def process(context: str, data: list[int]) -> int:
    print(f"Using context: {context}")
    return sum(data)

# 调用时不需要传 context
result = process([1, 2, 3])  # context 自动注入
```
