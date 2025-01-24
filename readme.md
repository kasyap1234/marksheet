# Create conda environment
conda create -n marksheet python=3.10 -y
conda activate marksheet

# Install core dependencies
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install \
  opencv-python \
  RealESRGAN \
  transformers \
  "flash-attn>=2.5.6" \
  "huggingface_hub>=0.22.2" \
  "accelerate>=0.27.0" \
  "bitsandbytes>=0.43.0"

# Download models
huggingface-cli download stepfun-ai/GOT-OCR2_0 --local-dir GOT_weights
huggingface-cli download microsoft/Phi-3-mini-4k-instruct