"""Compile an isolated personal installer from a reusable, validated program build."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build_installer(package, order_id, output, iscc, base=None, gpu=None):
    from knowledge_packages import inspect_package
    if not re.fullmatch('[a-f0-9]{32}', order_id):
        raise ValueError('订单 ID 必须为 32 位十六进制。')
    base = Path(base or ROOT / 'dist/Wenlu').resolve(strict=True)
    gpu = Path(gpu or ROOT / 'build/gpu-runtime').resolve(strict=True)
    release = json.loads((base / 'base-release.json').read_text(encoding='utf-8'))
    if (release.get('delivery_protocol') != 1 or
            not re.fullmatch(r'\d+\.\d+\.\d+', str(release.get('version', ''))) or
            digest(base / 'Wenlu.exe') != release.get('exe_sha256')):
        raise ValueError('主程序未包含交付功能或构建校验失败，请重新构建主程序。')
    if (base / '_internal/customer-delivery').exists() or list(base.rglob('*.wlkb')):
        raise ValueError('通用主程序目录含客户资料，拒绝构建。')
    package = Path(package).resolve(strict=True)
    manifest = inspect_package(package)
    if manifest['customer_id'] != order_id:
        raise ValueError('知识包客户身份与订单不一致。')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    delivery = output / 'payload'
    delivery.mkdir(exist_ok=True)
    copied = delivery / 'customer.wlkb'
    shutil.copyfile(package, copied)
    package_hash = digest(package)
    if digest(copied) != package_hash:
        raise ValueError('复制知识包失败。')
    (delivery / 'delivery.json').write_text(json.dumps({
        'protocol': 1, 'order_id': order_id, 'customer_id': manifest['customer_id'],
        'name': manifest['name'], 'package': 'customer.wlkb', 'sha256': package_hash
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    basename = 'Wenlu-' + release['version'] + '-' + order_id
    defines = {'AppVersion': release['version'], 'AppSourceDir': base,
               'GpuSourceDir': gpu, 'InstallerOutputDir': output,
               'InstallerBaseName': basename, 'PrivateDeliveryDir': delivery}
    if any(any(c in str(v) for c in '\r\n"') for v in defines.values()):
        raise ValueError('构建路径含不支持的字符。')
    subprocess.run([str(Path(iscc).resolve(strict=True)), '/Q',
                    *('/D' + k + '=' + str(v) for k, v in defines.items()),
                    str(ROOT / 'packaging/wenlu.iss')], check=True)
    installer = output / (basename + '.exe')
    if not installer.is_file() or installer.stat().st_size < 1024:
        raise ValueError('未生成安装包。')
    return {'path': str(installer), 'sha256': digest(installer),
            'size': installer.stat().st_size, 'version': release['version']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', required=True)
    parser.add_argument('--order-id', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--iscc', required=True)
    parser.add_argument('--base')
    parser.add_argument('--gpu')
    args = parser.parse_args()
    print(json.dumps(build_installer(args.package, args.order_id, args.output,
                                    args.iscc, args.base, args.gpu)))
