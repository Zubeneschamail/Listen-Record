闻录 GPU 离线组件（Windows x64 / NVIDIA）

1. 先安装闻录。安装主程序时选择“标准安装（CPU 转录）”，跳过联网 GPU 下载。
2. 正常退出闻录（包括系统托盘），再运行本离线组件安装程序。
3. 选择包含 Wenlu.exe 和 _internal 文件夹的闻录安装目录。
   默认：%LOCALAPPDATA%\Programs\Wenlu
   自定义位置安装的用户请手动选择对应目录。
4. 完成后重新打开闻录，在设置中执行 GPU 检测。

所有依赖已内置，安装过程不需要联网、Python 或 pip。
包含官方 NVIDIA cuBLAS 12.4.5.8、cuDNN 9.1.0.70、NVRTC 12.4.127。
文件仅写入所选应用的 _internal\gpu-runtime，不修改系统 PATH。
组件许可证保存在 _internal\gpu-runtime\licenses。

需要可用的 NVIDIA 显卡及相容驱动；本包不包含显卡驱动或转录模型。
更新/重装主程序若选择 CPU 安装，可能移除 GPU 组件；此时重新运行本离线包。
这是依赖补装包，不包含闻录主程序，不会创建独立的卸载项。
