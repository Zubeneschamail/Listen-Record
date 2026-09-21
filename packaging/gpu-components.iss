; Fixed NVIDIA releases from PyPI. URL overrides are for integration tests only.
#ifndef GpuCublasUrl
#define GpuCublasUrl "https://files.pythonhosted.org/packages/e2/2a/4f27ca96232e8b5269074a72e03b4e0d43aa68c9b965058b1684d07c6ff8/nvidia_cublas_cu12-12.4.5.8-py3-none-win_amd64.whl"
#endif
#ifndef GpuCudnnUrl
#define GpuCudnnUrl "https://files.pythonhosted.org/packages/3f/d0/f90ee6956a628f9f04bf467932c0a25e5a7e706a684b896593c06c82f460/nvidia_cudnn_cu12-9.1.0.70-py3-none-win_amd64.whl"
#endif
#ifndef GpuNvrtcUrl
#define GpuNvrtcUrl "https://files.pythonhosted.org/packages/7c/30/8c844bfb770f045bcd8b2c83455c5afb45983e1a8abf0c4e5297b481b6a5/nvidia_cuda_nvrtc_cu12-12.4.127-py3-none-win_amd64.whl"
#endif
[Files]
Source: "{#GpuNvrtcUrl}"; DestDir: "{app}\_internal\gpu-runtime"; DestName: "nvidia-cuda-nvrtc-12.4.127.zip"; ExternalSize: 50204828; Hash: "a961b2f1d5f17b14867c619ceb99ef6fcec12e46612711bcec78eb05068a60ec"; Components: gpu; Flags: external download extractarchive ignoreversion recursesubdirs createallsubdirs
Source: "{#GpuCublasUrl}"; DestDir: "{app}\_internal\gpu-runtime"; DestName: "nvidia-cublas-12.4.5.8.zip"; ExternalSize: 574555389; Hash: "5a796786da89203a0657eda402bcdcec6180254a8ac22d72213abc42069522dc"; Components: gpu; Flags: external download extractarchive ignoreversion recursesubdirs createallsubdirs
Source: "{#GpuCudnnUrl}"; DestDir: "{app}\_internal\gpu-runtime"; DestName: "nvidia-cudnn-9.1.0.70.zip"; ExternalSize: 1041157403; Hash: "6278562929433d68365a07a4a1546c237ba2849852c0d4b2262a486e805b977a"; Components: gpu; Flags: external download extractarchive ignoreversion recursesubdirs createallsubdirs
