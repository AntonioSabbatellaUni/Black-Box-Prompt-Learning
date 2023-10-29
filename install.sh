conda create -n bdpl python=3.9 -y
conda activate bdpl
pip install torch torchvisio torchaudio -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers
pip install -U huggingface_hub
pip install accelerate
pip install datasets
pip install wandb
pip install scikit-learn
pip install openai