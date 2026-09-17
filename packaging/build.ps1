param([string]$ISCC = "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe")
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
& .\.venv\Scripts\python.exe packaging/bundle_model.py
if ($LASTEXITCODE -ne 0) { throw 'Bundled model preparation failed' }
& .\.venv\Scripts\python.exe -m unittest discover -s tests -q
if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
& .\.venv\Scripts\python.exe -m PyInstaller packaging/wenlu.spec --noconfirm
if ($LASTEXITCODE -ne 0) { throw 'Executable build failed' }
$report = Join-Path (Get-Location) 'build/smoke-test.json'
$process = Start-Process -FilePath 'dist/Wenlu/Wenlu.exe' -ArgumentList @('--self-test', ('"' + $report + '"'), '--verify-bundled-model') -WindowStyle Hidden -PassThru -Wait
if ($process.ExitCode -ne 0 -or !(Test-Path $report) -or !(Get-Content $report -Raw | ConvertFrom-Json).ok) { throw 'Packaged smoke test failed' }
$version = & .\.venv\Scripts\python.exe -c 'from version import VERSION; print(VERSION)'
& $ISCC ('/DAppVersion=' + $version) packaging/wenlu.iss
if ($LASTEXITCODE -ne 0) { throw 'Installer build failed' }
