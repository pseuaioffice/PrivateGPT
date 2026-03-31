; MyAIAssistant Windows Installer Script
; Build steps:
;   1. Run: python build.py
;   2. Compile this file with Inno Setup 6 (ISCC)
;   3. Installer output will be written to packaging\installer_output\

#define MyAppName      "MyAIAssistant"
#define MyAppVersion   "1.3.0"
#define MyAppPublisher "MyAIAssistant"
#define MyAppURL       "https://example.com/myaiassistant"
#define MyAppExeName   "MyAIAssistant.exe"

[Setup]
AppId={{4915CEE2-98F2-4038-87E7-D149636633E2}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}/releases
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=installer_output
OutputBaseFilename=MyAIAssistant_Setup_v{#MyAppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin
MinVersion=10.0
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayName={#MyAppName}
UninstallDisplayIcon={app}\{#MyAppExeName}
CloseApplications=yes
CloseApplicationsFilter=*.exe
RestartApplications=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"; Flags: checkedonce

[Dirs]
Name: "{app}\backend\documents"; Permissions: users-full
Name: "{app}\backend\chroma_db"; Permissions: users-full
Name: "{app}\backend\logs"; Permissions: users-full

[Files]
Source: "dist\backend\*"; DestDir: "{app}\backend"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "*.pyc,__pycache__,.git*,documents\*,chroma_db\*,logs\*"
Source: "dist\frontend\*"; DestDir: "{app}\frontend"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "*.pyc,__pycache__,.git*"
Source: "dist\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist\MyAIAssistant_Launcher.bat"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName} now"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "taskkill"; Parameters: "/F /IM backend.exe"; Flags: runhidden; RunOnceId: "KillBackend"
Filename: "taskkill"; Parameters: "/F /IM frontend.exe"; Flags: runhidden; RunOnceId: "KillFrontend"
Filename: "taskkill"; Parameters: "/F /IM {#MyAppExeName}"; Flags: runhidden; RunOnceId: "KillLauncher"

[UninstallDelete]
Type: filesandordirs; Name: "{app}\backend\chroma_db"
Type: filesandordirs; Name: "{app}\backend\logs"
Type: filesandordirs; Name: "{app}\backend\__pycache__"
Type: filesandordirs; Name: "{app}\frontend\__pycache__"
Type: filesandordirs; Name: "{app}\backend\_internal\__pycache__"

[Messages]
FinishedHeadingLabel=MyAIAssistant Installed Successfully
FinishedLabelNoIcons=MyAIAssistant has been installed on your computer.%nIt is configured for local-first AI and can reuse your local Ollama installation.
ClickFinish=Click Finish to launch MyAIAssistant.

[Code]
var
  RuntimePage: TWizardPage;
  lblBackendPort, lblFrontendPort, lblOllamaUrl, lblChatModel, lblEmbeddingModel: TLabel;
  edtBackendPort, edtFrontendPort, edtOllamaUrl, edtChatModel, edtEmbeddingModel: TEdit;

  DBPage: TWizardPage;
  chkEnableDB: TCheckBox;
  lblHost, lblPort, lblName, lblUser, lblPass: TLabel;
  edtHost, edtPort, edtName, edtUser, edtPass: TEdit;

procedure UpdateDBFieldVisibility();
var
  Enabled: Boolean;
begin
  Enabled := chkEnableDB.Checked;
  lblHost.Enabled := Enabled; edtHost.Enabled := Enabled;
  lblPort.Enabled := Enabled; edtPort.Enabled := Enabled;
  lblName.Enabled := Enabled; edtName.Enabled := Enabled;
  lblUser.Enabled := Enabled; edtUser.Enabled := Enabled;
  lblPass.Enabled := Enabled; edtPass.Enabled := Enabled;
end;

procedure ChkEnableDBClick(Sender: TObject);
begin
  UpdateDBFieldVisibility();
end;

procedure CreateRuntimePage();
var
  Y: Integer;
begin
  RuntimePage := CreateCustomPage(
    wpSelectDir,
    'Runtime Configuration',
    'Configure the launcher ports and local Ollama settings for this machine.'
  );

  Y := 8;

  lblBackendPort := TLabel.Create(RuntimePage);
  lblBackendPort.Parent := RuntimePage.Surface;
  lblBackendPort.Left := 0;
  lblBackendPort.Top := Y;
  lblBackendPort.Caption := 'Backend port:';
  Y := Y + 18;

  edtBackendPort := TEdit.Create(RuntimePage);
  edtBackendPort.Parent := RuntimePage.Surface;
  edtBackendPort.Left := 0;
  edtBackendPort.Top := Y;
  edtBackendPort.Width := 100;
  edtBackendPort.Text := '8000';
  Y := Y + 30;

  lblFrontendPort := TLabel.Create(RuntimePage);
  lblFrontendPort.Parent := RuntimePage.Surface;
  lblFrontendPort.Left := 0;
  lblFrontendPort.Top := Y;
  lblFrontendPort.Caption := 'Frontend port:';
  Y := Y + 18;

  edtFrontendPort := TEdit.Create(RuntimePage);
  edtFrontendPort.Parent := RuntimePage.Surface;
  edtFrontendPort.Left := 0;
  edtFrontendPort.Top := Y;
  edtFrontendPort.Width := 100;
  edtFrontendPort.Text := '5000';
  Y := Y + 30;

  lblOllamaUrl := TLabel.Create(RuntimePage);
  lblOllamaUrl.Parent := RuntimePage.Surface;
  lblOllamaUrl.Left := 0;
  lblOllamaUrl.Top := Y;
  lblOllamaUrl.Caption := 'Ollama base URL:';
  Y := Y + 18;

  edtOllamaUrl := TEdit.Create(RuntimePage);
  edtOllamaUrl.Parent := RuntimePage.Surface;
  edtOllamaUrl.Left := 0;
  edtOllamaUrl.Top := Y;
  edtOllamaUrl.Width := 320;
  edtOllamaUrl.Text := 'http://127.0.0.1:11434';
  Y := Y + 30;

  lblChatModel := TLabel.Create(RuntimePage);
  lblChatModel.Parent := RuntimePage.Surface;
  lblChatModel.Left := 0;
  lblChatModel.Top := Y;
  lblChatModel.Caption := 'Local chat model:';
  Y := Y + 18;

  edtChatModel := TEdit.Create(RuntimePage);
  edtChatModel.Parent := RuntimePage.Surface;
  edtChatModel.Left := 0;
  edtChatModel.Top := Y;
  edtChatModel.Width := 220;
  edtChatModel.Text := 'qwen2.5:0.5b';
  Y := Y + 30;

  lblEmbeddingModel := TLabel.Create(RuntimePage);
  lblEmbeddingModel.Parent := RuntimePage.Surface;
  lblEmbeddingModel.Left := 0;
  lblEmbeddingModel.Top := Y;
  lblEmbeddingModel.Caption := 'Local embedding model:';
  Y := Y + 18;

  edtEmbeddingModel := TEdit.Create(RuntimePage);
  edtEmbeddingModel.Parent := RuntimePage.Surface;
  edtEmbeddingModel.Left := 0;
  edtEmbeddingModel.Top := Y;
  edtEmbeddingModel.Width := 220;
  edtEmbeddingModel.Text := 'nomic-embed-text';
end;

procedure CreateDBPage();
var
  Y: Integer;
begin
  DBPage := CreateCustomPage(
    RuntimePage.ID,
    'PostgreSQL Database (Optional)',
    'Configure chat history storage. Leave this disabled if PostgreSQL is not installed.'
  );

  Y := 8;

  chkEnableDB := TCheckBox.Create(DBPage);
  chkEnableDB.Parent := DBPage.Surface;
  chkEnableDB.Left := 0;
  chkEnableDB.Top := Y;
  chkEnableDB.Width := DBPage.SurfaceWidth;
  chkEnableDB.Caption := 'Enable PostgreSQL chat history';
  chkEnableDB.Checked := False;
  chkEnableDB.OnClick := @ChkEnableDBClick;
  Y := Y + 30;

  lblHost := TLabel.Create(DBPage);
  lblHost.Parent := DBPage.Surface;
  lblHost.Left := 0;
  lblHost.Top := Y;
  lblHost.Caption := 'Host:';
  Y := Y + 18;

  edtHost := TEdit.Create(DBPage);
  edtHost.Parent := DBPage.Surface;
  edtHost.Left := 0;
  edtHost.Top := Y;
  edtHost.Width := 220;
  edtHost.Text := 'localhost';
  Y := Y + 30;

  lblPort := TLabel.Create(DBPage);
  lblPort.Parent := DBPage.Surface;
  lblPort.Left := 0;
  lblPort.Top := Y;
  lblPort.Caption := 'Port:';
  Y := Y + 18;

  edtPort := TEdit.Create(DBPage);
  edtPort.Parent := DBPage.Surface;
  edtPort.Left := 0;
  edtPort.Top := Y;
  edtPort.Width := 100;
  edtPort.Text := '5432';
  Y := Y + 30;

  lblName := TLabel.Create(DBPage);
  lblName.Parent := DBPage.Surface;
  lblName.Left := 0;
  lblName.Top := Y;
  lblName.Caption := 'Database name:';
  Y := Y + 18;

  edtName := TEdit.Create(DBPage);
  edtName.Parent := DBPage.Surface;
  edtName.Left := 0;
  edtName.Top := Y;
  edtName.Width := 220;
  edtName.Text := 'ragchat';
  Y := Y + 30;

  lblUser := TLabel.Create(DBPage);
  lblUser.Parent := DBPage.Surface;
  lblUser.Left := 0;
  lblUser.Top := Y;
  lblUser.Caption := 'Username:';
  Y := Y + 18;

  edtUser := TEdit.Create(DBPage);
  edtUser.Parent := DBPage.Surface;
  edtUser.Left := 0;
  edtUser.Top := Y;
  edtUser.Width := 220;
  edtUser.Text := 'postgres';
  Y := Y + 30;

  lblPass := TLabel.Create(DBPage);
  lblPass.Parent := DBPage.Surface;
  lblPass.Left := 0;
  lblPass.Top := Y;
  lblPass.Caption := 'Password:';
  Y := Y + 18;

  edtPass := TEdit.Create(DBPage);
  edtPass.Parent := DBPage.Surface;
  edtPass.Left := 0;
  edtPass.Top := Y;
  edtPass.Width := 220;
  edtPass.PasswordChar := '*';

  UpdateDBFieldVisibility();
end;

function BuildDatabaseURL(): String;
begin
  if not chkEnableDB.Checked then
  begin
    Result := '';
    Exit;
  end;

  Result :=
    'postgresql://' + Trim(edtUser.Text) + ':' + edtPass.Text + '@' +
    Trim(edtHost.Text) + ':' + Trim(edtPort.Text) + '/' + Trim(edtName.Text);
end;

function IsValidPort(const Value: String): Boolean;
var
  PortNumber: Integer;
begin
  PortNumber := StrToIntDef(Trim(Value), 0);
  Result := (PortNumber >= 1) and (PortNumber <= 65535);
end;

procedure SetEnvValue(const Key, Value: String);
var
  EnvPath, Prefix: String;
  Lines: TStringList;
  I: Integer;
  Updated: Boolean;
begin
  EnvPath := ExpandConstant('{app}\backend\.env');
  if not FileExists(EnvPath) then
    Exit;

  Prefix := Key + '=';
  Updated := False;
  Lines := TStringList.Create();
  try
    Lines.LoadFromFile(EnvPath);
    for I := 0 to Lines.Count - 1 do
    begin
      if Pos(Prefix, Lines[I]) = 1 then
      begin
        Lines[I] := Prefix + Value;
        Updated := True;
      end;
    end;

    if not Updated then
      Lines.Add(Prefix + Value);

    Lines.SaveToFile(EnvPath);
  finally
    Lines.Free();
  end;
end;

procedure WriteRuntimeConfig();
begin
  SetEnvValue('MODEL_PROVIDER', 'ollama');
  SetEnvValue('BACKEND_HOST', '127.0.0.1');
  SetEnvValue('BACKEND_PORT', Trim(edtBackendPort.Text));
  SetEnvValue('FRONTEND_HOST', '127.0.0.1');
  SetEnvValue('FRONTEND_PORT', Trim(edtFrontendPort.Text));
  SetEnvValue('OLLAMA_BASE_URL', Trim(edtOllamaUrl.Text));
  SetEnvValue('CHAT_MODEL_LOCAL', Trim(edtChatModel.Text));
  SetEnvValue('EMBEDDING_MODEL_LOCAL', Trim(edtEmbeddingModel.Text));
  SetEnvValue('DATABASE_URL', BuildDatabaseURL());
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;

  if CurPageID = RuntimePage.ID then
  begin
    if not IsValidPort(edtBackendPort.Text) then
    begin
      MsgBox('Please enter a valid backend port between 1 and 65535.', mbError, MB_OK);
      Result := False;
      Exit;
    end;

    if not IsValidPort(edtFrontendPort.Text) then
    begin
      MsgBox('Please enter a valid frontend port between 1 and 65535.', mbError, MB_OK);
      Result := False;
      Exit;
    end;

    if Trim(edtBackendPort.Text) = Trim(edtFrontendPort.Text) then
    begin
      MsgBox('Backend and frontend ports must be different.', mbError, MB_OK);
      Result := False;
      Exit;
    end;

    if Trim(edtOllamaUrl.Text) = '' then
    begin
      MsgBox('Please enter the Ollama base URL.', mbError, MB_OK);
      Result := False;
      Exit;
    end;

    if Trim(edtChatModel.Text) = '' then
    begin
      MsgBox('Please enter the local chat model name.', mbError, MB_OK);
      Result := False;
      Exit;
    end;

    if Trim(edtEmbeddingModel.Text) = '' then
    begin
      MsgBox('Please enter the local embedding model name.', mbError, MB_OK);
      Result := False;
      Exit;
    end;
  end;

  if CurPageID = DBPage.ID then
  begin
    if chkEnableDB.Checked then
    begin
      if Trim(edtHost.Text) = '' then
      begin
        MsgBox('Please enter the PostgreSQL host.', mbError, MB_OK);
        Result := False;
        Exit;
      end;

      if not IsValidPort(edtPort.Text) then
      begin
        MsgBox('Please enter a valid PostgreSQL port between 1 and 65535.', mbError, MB_OK);
        Result := False;
        Exit;
      end;

      if Trim(edtName.Text) = '' then
      begin
        MsgBox('Please enter the PostgreSQL database name.', mbError, MB_OK);
        Result := False;
        Exit;
      end;

      if Trim(edtUser.Text) = '' then
      begin
        MsgBox('Please enter the PostgreSQL username.', mbError, MB_OK);
        Result := False;
        Exit;
      end;
    end;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
    WriteRuntimeConfig();
end;

function OllamaIsInstalled(): Boolean;
var
  LocalAppData: String;
begin
  Result := False;
  try
    if FileExists(ExpandConstant('{pf}\Ollama\ollama.exe')) then
    begin
      Result := True;
      Exit;
    end;

    LocalAppData := GetEnv('LOCALAPPDATA');
    if (LocalAppData <> '') and FileExists(LocalAppData + '\Programs\Ollama\ollama.exe') then
    begin
      Result := True;
      Exit;
    end;

    if RegKeyExists(HKEY_LOCAL_MACHINE, 'SOFTWARE\Ollama') then
    begin
      Result := True;
      Exit;
    end;

    if RegKeyExists(HKEY_CURRENT_USER, 'SOFTWARE\Ollama') then
    begin
      Result := True;
      Exit;
    end;
  except
    Result := False;
  end;
end;

function InitializeSetup(): Boolean;
begin
  Result := True;
  if not OllamaIsInstalled() then
    MsgBox(
      'Ollama does not appear to be installed.' + #13#10 + #13#10 +
      'MyAIAssistant defaults to local Ollama models.' + #13#10 +
      'You can continue installing now and install Ollama later from https://ollama.com.',
      mbInformation,
      MB_OK
    );
end;

procedure InitializeWizard();
begin
  CreateRuntimePage();
  CreateDBPage();
end;
