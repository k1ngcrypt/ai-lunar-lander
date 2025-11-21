REM basilisk/build_basilisk.bat
REM Build Basilisk using Visual Studio Developer Command Prompt for VS insiders
@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\Common7\Tools\VsDevCmd.bat"
cd /d "%~dp0"
python conanfile.py --buildType Release --clean --generator Ninja
