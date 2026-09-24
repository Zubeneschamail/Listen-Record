#ifndef RuntimeFiles
  #error Build with packaging/build_gpu_offline.py
#endif
[Setup]
AppId=WenluGpuOfflineRuntime
AppName=闻录 GPU 离线组件
AppVersion=12.4.9.1
AppPublisher=Listen-Record
DefaultDirName={localappdata}\Programs\Wenlu
UsePreviousAppDir=no
DisableDirPage=no
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
Uninstallable=no
CreateUninstallRegKey=no
CloseApplications=no
RestartApplications=no
OutputDir=..\release
OutputBaseFilename=Wenlu-GPU-Offline-cu12.4-cudnn9.1-win64
SetupIconFile=..\assets\wenlu.ico
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern
InfoBeforeFile=gpu-offline-readme.txt
[Files]
#include RuntimeFiles
[Code]
function OpenApplicationFile(FileName: String; Access, ShareMode: Cardinal;
  Security: Integer; Disposition, Attributes: Cardinal; Template: THandle): THandle;
  external 'CreateFileW@kernel32.dll stdcall';
function CloseApplicationFile(Handle: THandle): Boolean;
  external 'CloseHandle@kernel32.dll stdcall';

function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  Handle: THandle;
begin
  Result := '';
  if not FileExists(ExpandConstant('{app}\Wenlu.exe')) or
     not DirExists(ExpandConstant('{app}\_internal')) then
  begin
    Result := '请选择已安装闻录的目录，该目录应包含 Wenlu.exe 和 _internal 文件夹。请先完成闻录标准安装，再安装 GPU 离线组件。';
    Exit;
  end;
  // Check only this installation. A running copy elsewhere must not block it.
  // OPEN_EXISTING never creates or truncates the application file.
  Handle := OpenApplicationFile(ExpandConstant('{app}\Wenlu.exe'), $40000000,
    0, 0, 3, $80, 0);
  if Handle = THandle(-1) then
    Result := '所选闻录正在运行或安装目录不可写。请正常退出该目录中的闻录（包括系统托盘），确认目录写入权限后重试。'
  else
    CloseApplicationFile(Handle);
end;
