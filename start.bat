
if "%~1" EQU "" set run=1

reg query HKEY_LOCAL_MACHINE\SOFTWARE\Python\ContinuumAnalytics\Anaconda37-64
if %ERRORLEVEL% EQU 1 goto :command

set isAnaconda=1
for /f  "tokens=3*" %%a in ('REG QUERY HKEY_LOCAL_MACHINE\SOFTWARE\Python\ContinuumAnalytics\Anaconda37-64\InstallPath /ve') do set "root=%%a"
call %root%\Scripts\activate.bat %root%

python "Z:\uni\CP3403\CP3403-Assessment-1\controller.py"

pause
