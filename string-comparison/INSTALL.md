# String Comparison Project - Installation

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd string-comparison
```

### 2. Create a virtual environment (recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the project
```bash
python backend.py
```

The server will be available at: `http://localhost:8000`

## Usage

1. Open `http://localhost:8000` in your browser
2. Enter two sentences for comparison
3. View the detailed semantic analysis

## API Endpoints

- `GET /` - Web interface
- `POST /compare` - Compare two sentences
- `GET /health` - Server status

## Troubleshooting

### PyTorch/torchvision Compatibility
If you encounter PyTorch-related errors, make sure you have compatible versions installed:
```bash
pip install torch==2.2.0 torchvision==0.17.0
```

### NLTK Error
NLTK will automatically download the required data on first run.

### Memory Issues
The project uses machine learning models that can consume significant RAM. If you have problems, consider using a computer with at least 4GB of available RAM.
