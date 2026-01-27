#!/bin/bash
# =============================================================================
# Patent Guard v2.0 - Environment Setup Script (Intel iGPU / RunPod)
# =============================================================================
# This script sets up the development environment for Patent Guard v2.0
# Supports Intel XPU (IPEX-LLM) and NVIDIA CUDA (bitsandbytes) backends
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh [intel|runpod]
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CONDA_ENV_NAME="patent-guard"
PYTHON_VERSION="3.11"

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}\n"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# =============================================================================
# Environment Detection
# =============================================================================

detect_environment() {
    if [[ -n "${RUNPOD_POD_ID}" ]]; then
        echo "runpod"
    else
        echo "intel"
    fi
}

# =============================================================================
# Intel XPU Setup (IPEX-LLM)
# =============================================================================

setup_intel_xpu() {
    print_header "Setting up Intel XPU Environment (IPEX-LLM)"
    
    echo "📦 Installing Intel oneAPI Base Toolkit dependencies..."
    # Note: User should have Intel oneAPI installed
    # https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html
    
    echo "📦 Installing PyTorch for Intel XPU..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    
    echo "📦 Installing IPEX-LLM for XPU..."
    pip install --pre --upgrade ipex-llm[xpu] \
        --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
    
    print_success "Intel XPU environment configured!"
}

# =============================================================================
# RunPod/NVIDIA Setup (bitsandbytes)
# =============================================================================

setup_nvidia_cuda() {
    print_header "Setting up NVIDIA CUDA Environment (bitsandbytes)"
    
    echo "📦 Installing PyTorch with CUDA 12.1..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    echo "📦 Installing bitsandbytes for 4-bit quantization..."
    pip install bitsandbytes>=0.42.0
    
    print_success "NVIDIA CUDA environment configured!"
}

# =============================================================================
# Main Setup
# =============================================================================

main() {
    print_header "Patent Guard v2.0 - Environment Setup"
    
    # Determine environment
    ENV_TYPE=${1:-$(detect_environment)}
    echo -e "🖥️  Environment: ${YELLOW}${ENV_TYPE}${NC}"
    
    # Check for conda
    if ! command -v conda &> /dev/null; then
        print_error "Conda not found. Please install Miniconda or Anaconda first."
        exit 1
    fi
    print_success "Conda detected"
    
    # Setup conda environment
    setup_conda_env
    
    # Install backend-specific packages
    if [[ "$ENV_TYPE" == "runpod" ]] || [[ "$ENV_TYPE" == "nvidia" ]]; then
        setup_nvidia_cuda
    else
        setup_intel_xpu
    fi
    
    # Install common dependencies
    install_dependencies
    
    # Verify installation
    verify_installation "$ENV_TYPE"
    
    print_header "Setup Complete!"
    echo -e "To activate the environment, run:"
    echo -e "  ${GREEN}conda activate ${CONDA_ENV_NAME}${NC}"
    echo -e "\nTo run the embedding test:"
    echo -e "  ${GREEN}python main.py${NC}"
}

setup_conda_env() {
    print_header "Setting up Conda Environment"
    
    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        print_warning "Environment '${CONDA_ENV_NAME}' already exists. Activating..."
    else
        echo "Creating new conda environment: ${CONDA_ENV_NAME}"
        conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION} -y
    fi
    
    # Activate environment
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate ${CONDA_ENV_NAME}
    
    print_success "Conda environment ready: ${CONDA_ENV_NAME}"
}

install_dependencies() {
    print_header "Installing Common Dependencies"
    
    pip install --upgrade pip
    pip install -r requirements.txt
    
    # Save locked versions
    pip freeze > requirements.lock.txt
    print_success "Locked versions saved to requirements.lock.txt"
}

verify_installation() {
    local env_type=$1
    print_header "Verifying Installation"
    
    echo "🔍 Checking critical packages..."
    
    python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
    
    if [[ "$env_type" == "intel" ]]; then
        python -c "import intel_extension_for_pytorch; print(f'  IPEX: Available')" || print_warning "IPEX not available"
        python -c "import torch; print(f'  XPU Available: {torch.xpu.is_available() if hasattr(torch, \"xpu\") else False}')"
    else
        python -c "import torch; print(f'  CUDA Available: {torch.cuda.is_available()}')"
        python -c "import bitsandbytes; print(f'  BitsAndBytes: {bitsandbytes.__version__}')" || print_warning "bitsandbytes not available"
    fi
    
    python -c "import transformers; print(f'  Transformers: {transformers.__version__}')"
    python -c "import accelerate; print(f'  Accelerate: {accelerate.__version__}')"
    
    print_success "Verification complete!"
}

# =============================================================================
# Run Main
# =============================================================================

main "$@"
