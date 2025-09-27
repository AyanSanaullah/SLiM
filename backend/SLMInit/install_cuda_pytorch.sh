#!/bin/bash
echo "Installing CUDA-enabled PyTorch for RTX 4060..."
echo

echo "Step 1: Uninstalling existing PyTorch..."
pip uninstall torch torchvision torchaudio -y

echo
echo "Step 2: Installing CUDA-enabled PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo
echo "Step 3: Installing other dependencies..."
pip install -r requirements.txt

echo
echo "Installation complete! Testing CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}'); print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
