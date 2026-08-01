# ================================================================
# PlannerCopilot —— 完整教学版 Dockerfile
# 作用：把整个应用（代码+依赖+运行环境）打包成一个镜像
# 构建：docker build -t planner-copilot .
# 运行：docker run -p 8501:8501 --env-file .env planner-copilot
# ================================================================


# ── 第 1 层：基础镜像 ────────────────────────────────────────────
# 写什么：基于 python:3.12-slim 这个现成镜像往上盖。
# 为什么选 slim：完整版 python:3.12 约 1GB（带编译工具链等），slim 约
#   120MB。我们的依赖（psycopg2-binary、pymupdf 等）都是预编译的 wheel
#   包，不需要现场编译 C 扩展，slim 完全够用，拉取快、攻击面小。
# 不写会怎样：FROM 必须是第一条有效指令，没有它镜像"无根"，无法构建。
FROM python:3.12-slim


# ── 第 2 层：环境变量 ────────────────────────────────────────────
# 写什么：往镜像里永久写入三个环境变量（容器运行时一直有效）。
# 为什么逐个：
#   PYTHONUNBUFFERED=1 —— 让 Python 的 print/日志【不缓冲】直接输出。
#     不写的话日志会攒在缓冲区里，docker logs 看不到实时输出，排查
#     问题时会以为"程序卡死了"，其实它只是在憋日志。
#   PYTHONDONTWRITEBYTECODE=1 —— 不生成 .pyc 缓存文件。容器是即用即弃
#     的，.pyc 没有任何复用价值，只会让镜像变大。
#   UV_LINK_MODE=copy —— uv 装包默认用"硬链接"指向全局缓存，在 Docker
#     分层文件系统里硬链接可能跨层失效并刷警告；改成复制最稳。
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy


# ── 第 3 层：工作目录 ────────────────────────────────────────────
# 写什么：在镜像里创建 /app 目录，并把它设为"当前目录"。
# 为什么：之后所有 COPY、RUN、CMD 的相对路径都以 /app 为基准——相当于
#   给镜像内的操作 cd 到了一个固定工位。不写的话文件会散在根目录 /，
#   路径引用全乱。
WORKDIR /app


# ── 第 4 层：安装 uv ────────────────────────────────────────────
# 写什么：在镜像里装 uv（我们的包管理器）。
# 为什么用 uv 而不是直接 pip install -r requirements.txt：项目的依赖
#   清单写在 pyproject.toml 里（不是 requirements.txt），uv 读的就是它，
#   和你 Mac 本地开发用的是同一个工具、同一份清单——本地和容器行为一致。
# --no-cache-dir：pip 自己的下载缓存不保留，镜像更小。
RUN pip install --no-cache-dir uv


# ── 第 5 层：先只复制依赖清单（本文件最重要的一层）───────────────
# 写什么：只把 pyproject.toml 一个文件复制进镜像，源码先不复制。
# 为什么（Docker 层缓存机制，必须理解）：
#   Docker 构建是一层一层来的，每一层只要"输入没变"，下次构建就直接
#   用缓存跳过。第 6 层装依赖很慢（几分钟），它的输入就是 pyproject.toml。
#   把"复制清单"和"复制源码"分开后：你改了一行业务代码 → 第 5 层的输入
#   （pyproject.toml）没变 → 第 6 层直接命中缓存秒过 → 只有第 7 层
#   重新复制代码（几秒钟）。
#   如果合成一句 COPY . . ——任何代码改动都会让"装依赖"这层缓存失效，
#   每次 build 都重装几分钟依赖，日常开发会被拖死。
COPY pyproject.toml .


# ── 第 6 层：安装项目依赖 ───────────────────────────────────────
# 写什么：按 pyproject.toml 把依赖装进镜像里的 .venv。
# --no-dev：pytest 等开发工具不进生产镜像（测试在你本地跑，服务器上
#   不需要），镜像更小。
# 注意：Langfuse 在 optional-dependencies 里，默认不装——需要可观测性时
#   改成 uv sync --no-dev --extra observability 重新 build 即可。
RUN uv sync --no-dev


# ── 第 7 层：复制全部源码 ───────────────────────────────────────
# 写什么：把项目根目录的全部内容复制进 /app。
# 复制范围 = 全部文件 − .dockerignore 排除的（.env、.venv、.git、*.pdf
#   等）。所以 .dockerignore 不是可选项，是这道工序的"安检门"。
COPY . .


# ── 第 8 层：声明端口 ───────────────────────────────────────────
# 写什么：声明这个容器"打算"监听 8501 和 8000。
# 必须知道的真相：EXPOSE 只是【说明书】，告诉使用者"这个镜像用这些端口"，
#   它自己不做任何端口映射！真正让外面能访问的是 docker run -p 或
#   compose 里的 ports:。删掉这两行镜像照样能跑——但没人知道该映射什么。
EXPOSE 8501 8000


# ── 第 9 层：启动命令 ───────────────────────────────────────────
# 写什么：容器启动时执行 Streamlit 工作台（简历展示/演示的默认形态）。
# 为什么是 JSON 数组形式（exec 形式）而不是 CMD streamlit run app.py
#   （shell 形式）：exec 形式下 streamlit 直接成为容器的 1 号进程，能收到
#   docker stop 发来的停止信号、优雅退出；shell 形式会让命令裹在一层
#   sh 里，1 号进程是 sh，停止信号传不到你的程序，容器只能被强杀。
# 想跑 FastAPI 而不是工作台？不用改文件，运行时覆盖命令即可：
#   docker run -p 8000:8000 --env-file .env planner-copilot \
#     uv run uvicorn api:app --host 0.0.0.0 --port 8000
CMD ["uv", "run", "streamlit", "run", "app.py", \
     "--server.address", "0.0.0.0", "--server.port", "8501"]