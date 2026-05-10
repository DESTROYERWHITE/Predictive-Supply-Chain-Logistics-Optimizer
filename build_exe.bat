@echo off
setlocal
cd /d "%~dp0"
python -m PyInstaller --noconfirm --onefile --name "SupplyChainDashboard" launcher.py
copy /Y "dist\SupplyChainDashboard.exe" "SupplyChainDashboard.exe" >nul
echo.
echo Built executable: SupplyChainDashboard.exe
echo A second copy is also available at dist\SupplyChainDashboard.exe
