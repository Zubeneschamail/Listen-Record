# Windows 发布

专属订单交付：新版主程序支持安装后首次启用内置客户知识包。每个软件版本完成通用构建后，`dist/Wenlu/base-release.json` 记录交付协议、版本和 EXE 哈希；使用 `packaging/customer_installer.py` 在独立订单目录编译专属安装器。完整的 Linux 门户 + Windows worker 启动与部署说明见相邻项目 `../wenlu-delivery/README.md`。客户资料不会写入通用主程序或通用发布。

知识包接入：构建脚本会准备固定版本 BGE ONNX 模型。`wenlu.spec` 仅收集 `knowledge_embedding.FILES` 列出的三个通用模型文件，不收集 `.wlkb`、客户资料或 demo。程序自检会实际执行一次向量推理；无需访问知识库生成工具目录。源码环境可先运行 `prepare_knowledge_model.py`，直接调用 PyInstaller 前需先运行 `packaging/bundle_knowledge_model.py`。

1. 使用 Python 3.12 创建 `.venv`，安装 `requirements-build.txt`，安装 Inno Setup 6。
2. 修改 `version.py` 的版本号，执行 `packaging/build.ps1`。脚本准备并校验 GPU 运行库、执行测试、打包、独立程序自检，并生成 `release/Wenlu-Setup-版本号-Offline.exe`。
3. 使用 `python packaging/sign_release.py release/Wenlu-Setup-版本号-Offline.exe --key 私钥路径` 生成 `update.json` 和 `update.sig`。
4. 在 GitHub 创建匹配的 `v版本号` Release，将安装包和这两个文件一起上传。确认测试完成后发布；发布前客户端不会收到该版本。

可选的中文准确识别完整版使用单独的构建环境。用 Python 3.12 建立该环境，先从 `https://download.pytorch.org/whl/cu128` 安装 `torch==2.11.0+cu128`，再安装 `requirements-qwen-build.txt`；不要将它装进主程序 `.venv`。执行 `packaging/build.ps1 -WithQwen -QwenPython 独立环境的python.exe`。构建会准备 `packaging/qwen-models.json` 固定的两个官方模型、编译独立 `WenluQwen.exe`、逐文件校验并在 GPU 实际预热两个模型，再执行主程序测试与打包。需要构建机具备可用的 BF16 NVIDIA GPU。已有模型可预先用 `packaging/prepare_qwen_models.py --offline-source 目录` 从两个模型子目录复制并验证；默认缓存为 `build/qwen-models`。

完整版输出到 `release/版本号-Accurate-Offline`，包含安装 EXE 与分卷 `.bin` 文件，必须整体交付并保留同一目录。安装器增加「中文准确识别」类型和可选组件，模型、独立 Python/PyTorch 运行时及许可证全部离线安装到应用私有的 `_internal/qwen-engine`；不包含评测录音、用户资料或模型缓存的其他文件。标准构建仍生成原来的单 EXE，主程序不收集 PyTorch；标准版升级保留已经安装的准确组件。使用完整版安装器明确取消该组件时才移除它。

应用更新签名与自动下载仍用于标准单文件安装包，不要将分卷完整版 EXE 单独写入自动更新清单。完整组件按独立离线包交付，全部文件分别记录 SHA256。

本机 GPU 验收可增加 `Wenlu.exe --self-test 报告路径 --verify-bundled-model --verify-gpu --verify-capture-model`。额外检查在不同录音调用线程之间复用 Whisper 模型，并确认其常驻识别线程已退出；必须同时检查实际进程返回码，不能只看提前写出的 JSON。

更新签名使用 Ed25519。私钥不能进入仓库、安装包或日志；必须备份。当前发布公钥为 `assets/update-public.key`。CI 使用 `WENLU_UPDATE_PRIVATE_KEY` secret（原始 32 字节私钥的 Base64 编码），工作流只生成制品，不自动公开发布。更换公钥需要规划旧客户端的信任迁移。

安装包本身的 Windows Authenticode 签名需要发行者证书；更新签名不能替代 Windows 的发行者签名。当前构建不包含该证书。

安装包内置 small 转录模型，CPU 可离线运行；同一个安装包也内置固定版本的官方 cuBLAS、cuDNN 和 NVRTC wheel 内容，选择 GPU 安装时会把已校验的 DLL 和许可文件安装到应用私有的 `_internal/gpu-runtime`。NVRTC 固定为 CUDA 12.4 的 12.4.127，包含编译器与 builtins DLL；源码 GPU 依赖与安装器保持一致。构建机首次打包会从 PyPI 下载并按固定 SHA-256 校验 wheel，可通过 `WENLU_PYPI_INDEX` 指定可信镜像；后续构建复用 `build/gpu-wheels` 缓存。最终安装过程不联网。更换组件版本必须同时核对 URL、SHA-256、解压体积和 CTranslate2 兼容性。

也可以给已经完成 CPU 标准安装的闻录单独补装 `Wenlu-GPU-Offline-cu12.4-cudnn9.1-win64.exe`。退出应用后运行补装包并选择实际安装目录；安装完成后在应用内执行 GPU 检测。补装包只写入 `_internal/gpu-runtime`，包含三个官方组件及许可证，不包含主程序、驱动或模型，也不修改系统 PATH。CPU 模式重装/升级主程序可能移除该目录，此时再补装即可。

独立 GPU 补装包构建：将固定 wheel 缓存到 `build/gpu-wheels`，执行 `.\.venv\Scripts\python.exe packaging/build_gpu_offline.py`。脚本依据 `gpu-components.iss` 中的 URL 和哈希校验三个 wheel，再提取 DLL、许可证、文件哈希清单，使用 Inno Setup 生成独立 EXE 及 `.sha256`。不将其标记为应用最新版本，也不替换应用更新清单；单独发布到 GPU 组件 Release。

升级前正常退出闻录。应用内更新会等转写完成并退出后启动安装器；手动安装时安装器检测运行实例并提示退出。升级/卸载不删除 `%LOCALAPPDATA%\Wenlu` 数据。安装失败可重新运行上一个安装包恢复程序，用户记录独立保留；暂不支持自动版本回滚。

发布验收：全新 Windows 用户环境（无 Python）、有/无 GPU、模型下载失败后重试、升级保留配置记录、双击只启动一个实例、录制中更新等待保存、卸载后用户数据保留。开发机上的打包自检不能代替干净虚拟机验收。
