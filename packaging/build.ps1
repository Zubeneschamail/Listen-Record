param([string]$ISCC = "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
      [switch]$WithQwen, [string]$QwenPython)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
& .\.venv\Scripts\python.exe packaging/bundle_model.py
if ($LASTEXITCODE -ne 0) { throw 'Bundled model preparation failed' }
& .\.venv\Scripts\python.exe packaging/bundle_knowledge_model.py
if ($LASTEXITCODE -ne 0) { throw 'Knowledge model preparation failed' }
& .\.venv\Scripts\python.exe packaging/bundle_gpu.py
if ($LASTEXITCODE -ne 0) { throw 'GPU runtime preparation failed' }
if ($WithQwen) {
    if (!$QwenPython -or !(Test-Path -LiteralPath $QwenPython)) { throw 'Specify -QwenPython for the separate pinned engine build environment' }
    & .\.venv\Scripts\python.exe packaging/bundle_qwen_engine.py --check-cache
    if ($LASTEXITCODE -ne 0) {
        & .\.venv\Scripts\python.exe packaging/prepare_qwen_models.py
        if ($LASTEXITCODE -ne 0) { throw 'Pinned Qwen model preparation failed' }
        & $QwenPython -m PyInstaller packaging/qwen-worker.spec --noconfirm --distpath build/qwen-dist --workpath build/qwen-worker
        if ($LASTEXITCODE -ne 0) { throw 'Qwen executable build failed' }
        & .\.venv\Scripts\python.exe packaging/bundle_qwen_engine.py
        if ($LASTEXITCODE -ne 0) { throw 'Qwen engine staging failed' }
    }
    & .\.venv\Scripts\python.exe packaging/verify_qwen_engine.py
    if ($LASTEXITCODE -ne 0) { throw 'Standalone Qwen engine verification failed' }
}
$previousDataDir = $env:WENLU_DATA_DIR
$env:WENLU_DATA_DIR = Join-Path (Get-Location) ('build/test-data-' + $PID)
try {
    & .\.venv\Scripts\python.exe -m unittest discover -s tests -q
    $testExitCode = $LASTEXITCODE
} finally {
    if ($null -eq $previousDataDir) {
        Remove-Item Env:\WENLU_DATA_DIR -ErrorAction SilentlyContinue
    } else {
        $env:WENLU_DATA_DIR = $previousDataDir
    }
}
if ($testExitCode -ne 0) { throw 'Tests failed' }
& .\.venv\Scripts\python.exe -m PyInstaller packaging/wenlu.spec --noconfirm
if ($LASTEXITCODE -ne 0) { throw 'Executable build failed' }
$report = Join-Path (Get-Location) 'build/smoke-test.json'
$previousSmokeDataDir = $env:WENLU_DATA_DIR
$env:WENLU_DATA_DIR = Join-Path (Get-Location) ('build/smoke-test-data-' + $PID)
try {
    $process = Start-Process -FilePath 'dist/Wenlu/Wenlu.exe' -ArgumentList @('--self-test', ('"' + $report + '"'), '--verify-bundled-model') -WindowStyle Hidden -PassThru -Wait
} finally {
    if ($null -eq $previousSmokeDataDir) {
        Remove-Item Env:\WENLU_DATA_DIR -ErrorAction SilentlyContinue
    } else {
        $env:WENLU_DATA_DIR = $previousSmokeDataDir
    }
}
if ($process.ExitCode -ne 0 -or !(Test-Path $report) -or !(Get-Content $report -Raw | ConvertFrom-Json).ok) { throw 'Packaged smoke test failed' }
& .\.venv\Scripts\python.exe packaging/record_base_release.py
if ($LASTEXITCODE -ne 0) { throw 'Base release metadata failed' }
$version = & .\.venv\Scripts\python.exe -c 'from version import VERSION; print(VERSION)'
if ($WithQwen) {
    $accurateOutput = Join-Path (Get-Location) ('release/' + $version + '-Accurate-Offline')
    & $ISCC ('/DAppVersion=' + $version) ('/DQwenSourceDir=' + (Join-Path (Get-Location) 'build/qwen-engine')) ('/DInstallerOutputDir=' + $accurateOutput) ('/DInstallerBaseName=Wenlu-Setup-' + $version + '-Accurate-Offline') packaging/wenlu.iss
} else {
    & $ISCC ('/DAppVersion=' + $version) packaging/wenlu.iss
}
if ($LASTEXITCODE -ne 0) { throw 'Installer build failed' }
