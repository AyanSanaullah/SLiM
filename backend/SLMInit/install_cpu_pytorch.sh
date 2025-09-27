#!/bin/bash
echo "Installing CPU-optimized PyTorch..."
echo

# Detect if we're on Apple Silicon
if [[ $(uname -m) == "arm64" ]]; then
    echo "Detected Apple Silicon (M1/M2/M3)"
    echo "Installing Apple Silicon optimized PyTorch..."
    pip uninstall torch torchvision torchaudio -y
    pip install torch torchvision torchaudio
else
    echo "Detected Intel/AMD processor"
    echo "Installing CPU-only PyTorch..."
    pip uninstall torch torchvision torchaudio -y
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo
echo "Installing other dependencies..."
pip install -r requirements_cpu.txt

echo
echo "Installation complete! Testing installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'Device: {\"MPS\" if hasattr(torch.backends, \"mps\") and torch.backends.mps.is_available() else \"CPU\"}')"
