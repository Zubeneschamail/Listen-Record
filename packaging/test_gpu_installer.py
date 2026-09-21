"""Manual integration test: real Inno downloader/extractor against cached NVIDIA wheels.

Run after caching the pinned wheels in build/gpu-wheels. Uses only loopback HTTP,
isolated build directories, and a separate non-registered test installer.
"""
import functools
import http.server
import os
from pathlib import Path
import subprocess
import threading

ROOT = Path(__file__).resolve().parents[1]


def main():
    wheels = ROOT / 'build/gpu-wheels'
    names = ['nvidia_cublas_cu12-12.4.5.8-py3-none-win_amd64.whl',
             'nvidia_cudnn_cu12-9.1.0.70-py3-none-win_amd64.whl',
             'nvidia_cuda_nvrtc_cu12-12.4.127-py3-none-win_amd64.whl']
    assert all((wheels / name).is_file() for name in names)
    requests = []

    class Handler(http.server.SimpleHTTPRequestHandler):
        corrupt = False

        def do_GET(self):
            requests.append(self.path)
            if self.corrupt:
                self.send_response(200)
                self.send_header('Content-Length', '3')
                self.end_headers()
                self.wfile.write(b'bad')
            else:
                super().do_GET()

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Handler, directory=str(wheels)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    folder = ROOT / 'build/gpu-installer-check'
    folder.mkdir(exist_ok=True)
    script = folder / 'test.iss'
    script.write_text(f'''[Setup]
AppName=Wenlu GPU Integration
AppVersion=1.0
AppId=WenluGpuIntegrationOnly
DefaultDirName={{tmp}}\\WenluGpuIntegrationOnly
PrivilegesRequired=lowest
Uninstallable=no
CreateUninstallRegKey=no
UsePreviousAppDir=no
ArchiveExtraction=full
OutputDir={folder}
OutputBaseFilename=test
[Types]
Name: "custom"; Description: "Custom"; Flags: iscustom
[Components]
Name: "core"; Description: "Core"; Types: custom; Flags: fixed
Name: "gpu"; Description: "GPU"
#include "{ROOT / 'packaging/gpu-components.iss'}"
''', encoding='utf-8-sig')
    compiler = Path(os.environ['LOCALAPPDATA']) / 'Programs/Inno Setup 6/ISCC.exe'
    base = f'http://127.0.0.1:{server.server_port}/'
    try:
        subprocess.run([str(compiler), '/Q', '/DGpuCublasUrl=' + base + names[0],
                        '/DGpuCudnnUrl=' + base + names[1], '/DGpuNvrtcUrl=' + base + names[2], str(script)], check=True)
        for mode in ('cpu', 'gpu', 'corrupt'):
            requests.clear()
            Handler.corrupt = mode == 'corrupt'
            target = folder / mode
            result = subprocess.run([str(folder / 'test.exe'), '/VERYSILENT', '/SUPPRESSMSGBOXES',
                '/NORESTART', '/DIR=' + str(target), '/LOG=' + str(folder / (mode + '.log')),
                '/COMPONENTS=core' + (',gpu' if mode != 'cpu' else '')], timeout=180)
            if mode == 'cpu':
                assert result.returncode == 0 and not requests, (result.returncode, requests)
                assert not list(target.rglob('*.dll'))
            elif mode == 'gpu':
                assert result.returncode == 0, result.returncode
                assert all('/' + name in requests for name in names), requests
                assert (target / '_internal/gpu-runtime/nvidia/cublas/bin/cublas64_12.dll').is_file()
                assert (target / '_internal/gpu-runtime/nvidia/cudnn/bin/cudnn64_9.dll').is_file()
                assert (target / '_internal/gpu-runtime/nvidia/cuda_nvrtc/bin/nvrtc64_120_0.dll').is_file()
                assert (target / '_internal/gpu-runtime/nvidia/cuda_nvrtc/bin/nvrtc-builtins64_124.dll').is_file()
                assert list(target.rglob('License.txt')) or list(target.rglob('LICENSE*'))
            else:
                assert result.returncode != 0 and requests
                assert not list(target.rglob('*.dll'))
            print(f'PASS {mode}: exit={result.returncode}, downloads={len(requests)}', flush=True)
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    main()
