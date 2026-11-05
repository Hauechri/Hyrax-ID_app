@echo off
setlocal

rem Starts Hyrax-ID using the project's virtual environment.
rem Put this file either
rem   (a) in the folder that contains both ".venv" and "Identification-App", or
rem   (b) inside "Identification-App" itself (with ".venv" one level up).

set "HERE=%~dp0"

if exist "%HERE%.venv\Scripts\activate.bat" (
    set "VENV=%HERE%.venv"
    set "APP_ROOT=%HERE%Identification-App"
) else if exist "%HERE%..\.venv\Scripts\activate.bat" (
    set "VENV=%HERE%..\.venv"
    set "APP_ROOT=%HERE%"
) else (
    echo Could not find the .venv folder next to or above this file.
    echo Looked in: "%HERE%.venv" and "%HERE%..\.venv"
    pause
    exit /b 1
)

if not exist "%APP_ROOT%\app\main.py" (
    echo Could not find "%APP_ROOT%\app\main.py"
    pause
    exit /b 1
)

call "%VENV%\Scripts\activate.bat"
cd /d "%APP_ROOT%"
python app\main.py

rem Keep the window open if the app crashed, so the traceback can be read.
if errorlevel 1 (
    echo.
    echo Hyrax-ID exited with an error ^(code %errorlevel%^).
    pause
)
endlocal
