"""Import an installer delivery once, before loading the reference controls."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

from app_paths import DATA, RESOURCES


def bootstrap_delivery(resources=RESOURCES, data=DATA):
    from knowledge_packages import inspect_package, validate_selection
    from reference_files import load_settings, save_settings

    folder = Path(resources) / 'customer-delivery'
    manifest_path = folder / 'delivery.json'
    if not manifest_path.exists():
        return False
    if manifest_path.stat().st_size > 16384:
        raise ValueError('专属安装包的交付清单过大。')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest.get('protocol') != 1 or
            not re.fullmatch(r'[a-f0-9]{32}', str(manifest.get('order_id', ''))) or
            manifest.get('package') != 'customer.wlkb' or
            not re.fullmatch(r'[a-f0-9]{64}', str(manifest.get('sha256', '')))):
        raise ValueError('专属安装包的交付清单无效。')
    data = Path(data)
    data.mkdir(parents=True, exist_ok=True)
    marker = data / 'delivery-imported.json'
    identity = {'order_id': manifest['order_id'], 'sha256': manifest['sha256']}
    if marker.exists() and json.loads(marker.read_text(encoding='utf-8')) == identity:
        return False  # Respect subsequent removal or disabling by the user.
    source = folder / 'customer.wlkb'
    with source.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != manifest['sha256']:
            raise ValueError('内置客户知识包校验失败，请重新下载安装包。')
    package = inspect_package(source)
    if package['customer_id'] != manifest.get('customer_id'):
        raise ValueError('内置知识包与订单的客户身份不一致。')
    managed = data / 'knowledge-packages'
    managed.mkdir(exist_ok=True)
    target = managed / ('delivery-' + manifest['order_id'] + '-' + manifest['sha256'][:16] + '.wlkb')
    settings_path = data / 'reference-files.json'
    settings = load_settings(settings_path)
    # A new personal delivery replaces the previous installer delivery in the selection.
    # Preserve references added manually through the application.
    paths = [p for p in settings['paths'] if not (
        Path(p).resolve().parent == managed.resolve() and Path(p).name.startswith('delivery-'))]
    if len(paths) >= 32:
        raise ValueError('资料已达 32 项，请先移除一项再重新启动闻录。')
    validate_selection(paths + [str(source)])
    handle = tempfile.NamedTemporaryFile(dir=managed, suffix='.tmp', delete=False)
    temporary = Path(handle.name)
    handle.close()
    try:
        shutil.copyfile(source, temporary)
        with temporary.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != manifest['sha256']:
                raise ValueError('客户知识包复制校验失败。')
        temporary.replace(target)
        save_settings({'enabled': True, 'paths': paths + [str(target)]}, settings_path)
        pending_marker = marker.with_suffix('.tmp')
        pending_marker.write_text(json.dumps(identity), encoding='utf-8')
        pending_marker.replace(marker)
    finally:
        temporary.unlink(missing_ok=True)
    return True
