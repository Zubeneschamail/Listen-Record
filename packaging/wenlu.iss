#ifndef AppVersion
#define AppVersion "0.3.0"
#endif
[Setup]
AppId={{9D469B55-B7E6-497F-89B4-FCD21C0929C1}
AppName=闻录
AppVersion={#AppVersion}
AppPublisher=Listen-Record
AppMutex=Local\Wenlu.InstallerGuard
DefaultDirName={localappdata}\Programs\Wenlu
DefaultGroupName=闻录
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\release
OutputBaseFilename=Wenlu-Setup-{#AppVersion}
SetupIconFile=..\assets\wenlu.ico
UninstallDisplayIcon={app}\Wenlu.exe
Compression=lzma2/fast
SolidCompression=yes
ArchiveExtraction=full
WizardStyle=modern
CloseApplications=no
RestartApplications=no
DisableProgramGroupPage=yes
[Types]
Name: "standard"; Description: "标准安装（CPU 转录）"
Name: "full"; Description: "GPU 加速安装（需联网下载组件）"
Name: "custom"; Description: "自定义安装"; Flags: iscustom
[Components]
Name: "core"; Description: "闻录与内置转录模型（必需）"; Types: standard full custom; Flags: fixed
Name: "gpu"; Description: "GPU 加速组件（NVIDIA 显卡，需联网下载约 1.0 GB）"; Types: full
[Files]
Source: "..\dist\Wenlu\*"; DestDir: "{app}"; Excludes: "_internal\nvidia\*,_internal\gpu-runtime\*"; Components: core; Flags: ignoreversion recursesubdirs createallsubdirs
#include "gpu-components.iss"
[InstallDelete]
Type: filesandordirs; Name: "{app}\_internal\nvidia"; Check: not WizardIsComponentSelected('gpu')
Type: filesandordirs; Name: "{app}\_internal\gpu-runtime"; Check: not WizardIsComponentSelected('gpu')
[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; Flags: checkedonce
[Icons]
Name: "{group}\闻录"; Filename: "{app}\Wenlu.exe"; IconFilename: "{app}\_internal\assets\wenlu.ico"; AppUserModelID: "Wenlu.Desktop"
Name: "{autodesktop}\闻录"; Filename: "{app}\Wenlu.exe"; IconFilename: "{app}\_internal\assets\wenlu.ico"; AppUserModelID: "Wenlu.Desktop"; Tasks: desktopicon
[Run]
Filename: "{app}\Wenlu.exe"; Description: "打开闻录"; Flags: nowait postinstall skipifsilent
Filename: "{app}\Wenlu.exe"; Flags: nowait; Check: RestartAfterUpdate
[Code]
function RestartAfterUpdate(): Boolean;
begin
  Result := ExpandConstant('{param:RESTARTWENLU|0}') = '1';
end;
function InitializeSetup(): Boolean;
begin
  Result := True;
end;
