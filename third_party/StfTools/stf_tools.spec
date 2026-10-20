# PyInstaller 打包配置：生成单一可执行程序
# 使用: pyinstaller stf_tools.spec
# 输出: dist/stf_tools  (Linux/mac) 或 dist/stf_tools.exe (Windows)

# -*- mode: python ; coding: utf-8 -*-

import os

block_cipher = None

# 脚本所在目录
script_dir = os.path.dirname(os.path.abspath(SPEC))

# 需随包的数据文件（如描述符）
datas = []
if os.path.isfile(os.path.join(script_dir, 'proto_descriptors.bin')):
    datas.append(('proto_descriptors.bin', '.'))

# 入口与依赖脚本（import 时会自动打包）
# stf_tools.py, stf_to_json_full.py, stf_extract_images.py, stf_extract_h265_images.py,
# stf_split_fast.py, stf_reader.py, proto_from_descriptors.py

a = Analysis(
    ['stf_tools.py'],
    pathex=[script_dir],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'google.protobuf',
        'google.protobuf.descriptor_pb2',
        'google.protobuf.descriptor_pool',
        'google.protobuf.reflection',
        'google.protobuf.json_format',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='stf_tools',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
