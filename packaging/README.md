# Windows 发布

知识包接入：构建脚本会准备固定版本 BGE ONNX 模型。`wenlu.spec` 仅收集 `knowledge_embedding.FILES` 列出的三个通用模型文件，不收集 `.wlkb`、客户资料或 demo。程序自检会实际执行一次向量推理；无需访问知识库生成工具目录。源码环境可先运行 `prepare_knowledge_model.py`，直接调用 PyInstaller 前需先运行 `packaging/bundle_knowledge_model.py`。

1. 使用 Python 3.12 创建 `.venv`，安装 `requirements-build.txt`，安装 Inno Setup 6。
2. 修改 `version.py` 的版本号，执行 `packaging/build.ps1`。脚本执行测试、打包、独立程序自检并生成 `release/Wenlu-Setup-版本号.exe`。
3. 使用 `python packaging/sign_release.py release/Wenlu-Setup-版本号.exe --key 私钥路径` 生成 `update.json` 和 `update.sig`。
4. 在 GitHub 创建匹配的 `v版本号` Release，将安装包和这两个文件一起上传。确认测试完成后发布；发布前客户端不会收到该版本。

更新签名使用 Ed25519。私钥不能进入仓库、安装包或日志；必须备份。当前发布公钥为 `assets/update-public.key`。CI 使用 `WENLU_UPDATE_PRIVATE_KEY` secret（原始 32 字节私钥的 Base64 编码），工作流只生成制品，不自动公开发布。更换公钥需要规划旧客户端的信任迁移。

安装包本身的 Windows Authenticode 签名需要发行者证书；更新签名不能替代 Windows 的发行者签名。当前构建不包含该证书。

标准包内置 small 转录模型，CPU 可离线运行；不嵌入 NVIDIA 运行库。GPU 选项通过 `gpu-components.iss` 从 PyPI 下载固定版本官方 cuBLAS、cuDNN 和 NVRTC wheel，验证内置 SHA-256 后解压到应用私有的 `_internal/gpu-runtime`，同时保留包内许可文件。NVRTC 固定为 CUDA 12.4 的 12.4.127，包含编译器与 builtins DLL；源码 GPU 依赖与安装器保持一致。主程序构建不下载或暂存 GPU 库；网络受限时可另外发布下述 GPU 离线补装包。下载总量约 1.0 GiB；不选择 GPU 时不会请求这些文件。需要支持 `download`、`extractarchive` 和 `Hash` 的新版 Inno Setup 6。更换组件版本必须同时核对 URL、SHA-256、解压体积和 CTranslate2 兼容性。

GPU 下载受限时，可使用独立离线补装包 `Wenlu-GPU-Offline-cu12.4-cudnn9.1-win64.exe`。先按 CPU 标准安装闻录，再退出应用、运行补装包并选择实际安装目录；安装完成后在应用内执行 GPU 检测。补装包只写入 `_internal/gpu-runtime`，包含三个官方组件及许可证，不包含主程序、驱动或模型，也不修改系统 PATH。CPU 模式重装/升级主程序可能移除该目录，此时再补装即可。

离线包构建：将固定 wheel 缓存到 `build/gpu-wheels`，执行 `.\.venv\Scripts\python.exe packaging/build_gpu_offline.py`。脚本依据在线安装器的 URL/哈希校验三个 wheel，再提取 DLL、许可证、文件哈希清单，使用 Inno Setup 生成独立 EXE 及 `.sha256`。不将其标记为应用最新版本，也不替换应用更新清单；单独发布到 GPU 组件 Release。

升级前正常退出闻录。应用内更新会等转写完成并退出后启动安装器；手动安装时安装器检测运行实例并提示退出。升级/卸载不删除 `%LOCALAPPDATA%\Wenlu` 数据。安装失败可重新运行上一个安装包恢复程序，用户记录独立保留；暂不支持自动版本回滚。

发布验收：全新 Windows 用户环境（无 Python）、有/无 GPU、模型下载失败后重试、升级保留配置记录、双击只启动一个实例、录制中更新等待保存、卸载后用户数据保留。开发机上的打包自检不能代替干净虚拟机验收。
