"""Signed release metadata and staged installer downloads."""
import base64
import hashlib
import json
from pathlib import Path
import re
import urllib.request
from urllib.parse import urlparse
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from app_paths import DATA, RESOURCES
from version import VERSION, REPOSITORY

BASE = f'https://github.com/{REPOSITORY}/releases/'


def version_tuple(value):
    if not re.fullmatch(r'\d+\.\d+\.\d+', value):
        raise ValueError('不支持的版本格式')
    return tuple(map(int, value.split('.')))


def fetch(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'Wenlu/' + VERSION})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read(1024 * 1024)


def verify_manifest(raw, signature, public_key=None):
    public_key = public_key or (RESOURCES / 'assets/update-public.key').read_bytes()
    Ed25519PublicKey.from_public_bytes(public_key).verify(base64.b64decode(signature, validate=True), raw)
    value = json.loads(raw)
    version_tuple(value['version'])
    if not value['url'].startswith(BASE + 'download/') or urlparse(value['url']).scheme != 'https':
        raise ValueError('更新地址不属于本项目发布页')
    if not re.fullmatch('[0-9a-f]{64}', value['sha256']):
        raise ValueError('无效的安装包校验值')
    if not isinstance(value['size'], int) or not 0 < value['size'] < 4 * 1024**3:
        raise ValueError('无效的安装包大小')
    return value


def check_update():
    raw = fetch(BASE + 'latest/download/update.json')
    signature = fetch(BASE + 'latest/download/update.sig')
    value = verify_manifest(raw, signature)
    return value if version_tuple(value['version']) > version_tuple(VERSION) else None


def download_update(value, progress=lambda n: None):
    directory = DATA / 'updates'
    directory.mkdir(exist_ok=True)
    target = directory / ('Wenlu-Setup-' + value['version'] + '.exe')
    partial = target.with_suffix('.part')
    digest = hashlib.sha256()
    count = 0
    request = urllib.request.Request(value['url'], headers={'User-Agent': 'Wenlu/' + VERSION})
    try:
        with urllib.request.urlopen(request, timeout=30) as response, partial.open('wb') as file:
            while block := response.read(1024 * 1024):
                count += len(block)
                if count > value['size']:
                    raise ValueError('安装包长度不正确')
                digest.update(block)
                file.write(block)
                progress(int(count * 100 / value['size']))
        if count != value['size'] or digest.hexdigest() != value['sha256']:
            raise ValueError('安装包完整性校验失败，请重新下载')
        partial.replace(target)
        return target
    finally:
        partial.unlink(missing_ok=True)


def launch_installer_when_closed(path, digest):
    """Separate hidden process waits for us, rechecks the installer, then starts setup."""
    import os
    import subprocess
    escaped = str(path).replace("'", "''")
    script = f"Wait-Process -Id {os.getpid()} -ErrorAction SilentlyContinue; "
    script += f"if ((Get-FileHash -LiteralPath '{escaped}' -Algorithm SHA256).Hash -ne '{digest}') {{exit 2}}; "
    script += f"Start-Process -FilePath '{escaped}' -ArgumentList '/SILENT','/NORESTART','/RESTARTWENLU=1'"
    encoded = base64.b64encode(script.encode('utf-16le')).decode('ascii')
    subprocess.Popen(['powershell.exe', '-NoProfile', '-WindowStyle', 'Hidden', '-EncodedCommand', encoded],
                     creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS)
