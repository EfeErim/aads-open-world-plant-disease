@echo off
setlocal EnableExtensions EnableDelayedExpansion

for %%I in ("%~dp0..") do set "REPO_ROOT=%%~fI"
set "PYTHON_EXE="

if exist "%REPO_ROOT%\.venv\Scripts\python.exe" set "PYTHON_EXE=%REPO_ROOT%\.venv\Scripts\python.exe"

if not defined PYTHON_EXE if defined VIRTUAL_ENV if exist "%VIRTUAL_ENV%\Scripts\python.exe" set "PYTHON_EXE=%VIRTUAL_ENV%\Scripts\python.exe"

if not defined PYTHON_EXE (
  for /f "delims=" %%I in ('py -3.11 -c "import sys; print(sys.executable)" 2^>nul') do (
    if not defined PYTHON_EXE if exist "%%~fI" set "PYTHON_EXE=%%~fI"
  )
)

if not defined PYTHON_EXE if defined LocalAppData (
  for /f "delims=" %%I in ('dir /b /ad /o-n "%LocalAppData%\Programs\Python\Python*" 2^>nul') do (
    if not defined PYTHON_EXE if exist "%LocalAppData%\Programs\Python\%%I\python.exe" set "PYTHON_EXE=%LocalAppData%\Programs\Python\%%I\python.exe"
  )
)

if not defined PYTHON_EXE (
  for /f "usebackq delims=" %%I in (`where.exe python 2^>nul`) do (
    set "CANDIDATE=%%~fI"
    echo !CANDIDATE! | findstr /i /c:"WindowsApps\python.exe" >nul
    if errorlevel 1 if not defined PYTHON_EXE set "PYTHON_EXE=!CANDIDATE!"
  )
)

if defined PYTHON_EXE (
  "%PYTHON_EXE%" %*
  exit /b !ERRORLEVEL!
)

>&2 echo [AADS] No usable Python interpreter was found.
>&2 echo [AADS] Expected the repo venv at "%REPO_ROOT%\.venv\Scripts\python.exe"
>&2 echo [AADS] or a real install under "%LocalAppData%\Programs\Python\Python*".
>&2 echo [AADS] If "python" opens the Microsoft Store, disable the App Execution Alias.
exit /b 1
