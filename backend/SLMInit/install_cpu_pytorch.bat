@echo off
echo Installing CPU-optimized PyTorch...
echo.

echo Uninstalling existing PyTorch...
pip uninstall torch torchvision torchaudio -y

echo.
echo Installing CPU-only PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo.
echo Installing other dependencies...
pip install -r requirements_cpu.txt

echo.
echo Installation complete! Testing installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print('Device: CPU')"

pause
