#ifndef AppVersion
  #define AppVersion "0.4.20"
#endif
#ifndef AppSourceDir
#define AppSourceDir "..\dist\Wenlu"
#endif
#ifndef GpuSourceDir
#define GpuSourceDir "..\build\gpu-runtime"
#endif
#ifndef InstallerOutputDir
#define InstallerOutputDir "..\release"
#endif
#ifndef InstallerBaseName
#define InstallerBaseName "Wenlu-Setup-" + AppVersion + "-Offline"
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
OutputDir={#InstallerOutputDir}
OutputBaseFilename={#InstallerBaseName}
SetupIconFile=..\assets\wenlu.ico
UninstallDisplayIcon={app}\Wenlu.exe
Compression=lzma2/fast
SolidCompression=yes
#ifdef QwenSourceDir
DiskSpanning=yes
DiskSliceSize=2100000000
#endif
ArchiveExtraction=full
WizardStyle=modern
CloseApplications=no
RestartApplications=no
DisableProgramGroupPage=yes
[Types]
Name: "standard"; Description: "标准安装（CPU 转录）"
Name: "full"; Description: "GPU 加速安装（离线组件）"
#ifdef QwenSourceDir
Name: "accurate"; Description: "中文准确识别（NVIDIA BF16 显卡，离线模型与运行组件）"
#endif
Name: "custom"; Description: "自定义安装"; Flags: iscustom
[Components]
#ifdef QwenSourceDir
Name: "core"; Description: "闻录与内置转录模型（必需）"; Types: standard full accurate custom; Flags: fixed
Name: "gpu"; Description: "Whisper GPU 加速组件（NVIDIA 显卡）"; Types: full accurate
Name: "qwen"; Description: "中文准确识别（Qwen 模型与独立运行组件，建议 8 GB 显存）"; Types: accurate
#else
Name: "core"; Description: "闻录与内置转录模型（必需）"; Types: standard full custom; Flags: fixed
Name: "gpu"; Description: "GPU 加速组件（NVIDIA 显卡，已包含在安装包中）"; Types: full
#endif
[Files]
Source: "{#AppSourceDir}\*"; DestDir: "{app}"; Excludes: "_internal\nvidia\*,_internal\gpu-runtime\*,_internal\qwen-engine\*,_internal\customer-delivery\*"; Components: core; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#GpuSourceDir}\*"; DestDir: "{app}\_internal\gpu-runtime"; Components: gpu; Flags: ignoreversion recursesubdirs createallsubdirs
#ifdef QwenSourceDir
Source: "{#QwenSourceDir}\*"; DestDir: "{app}\_internal\qwen-engine"; Components: qwen; Flags: ignoreversion recursesubdirs createallsubdirs
#endif
#ifdef PrivateDeliveryDir
Source: "{#PrivateDeliveryDir}\delivery.json"; DestDir: "{app}\_internal\customer-delivery"; Components: core; Flags: ignoreversion
Source: "{#PrivateDeliveryDir}\customer.wlkb"; DestDir: "{app}\_internal\customer-delivery"; Components: core; Flags: ignoreversion
#endif
[InstallDelete]
Type: filesandordirs; Name: "{app}\_internal\customer-delivery"
Type: filesandordirs; Name: "{app}\_internal\nvidia"; Check: not WizardIsComponentSelected('gpu')
Type: filesandordirs; Name: "{app}\_internal\gpu-runtime"; Check: not WizardIsComponentSelected('gpu')
#ifdef QwenSourceDir
Type: filesandordirs; Name: "{app}\_internal\qwen-engine"; Check: not WizardIsComponentSelected('qwen')
#endif
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
