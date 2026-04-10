@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_server.ps1" %*
