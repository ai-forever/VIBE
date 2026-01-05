# VIBE - Visual Instruction Based Editor

[[`Demo`](https://huggingface.co/spaces/iitolstykh/VIBE-Image-Edit-DEMO)] [[`🤗 HuggingFace Model`](https://huggingface.co/iitolstykh/VIBE-Image-Edit)]

VIBE is a powerful open-source framework for text-guided image editing.
It leverages the efficient [Sana1.5-1.6B](github.com/NVlabs/Sana) diffusion model and [Qwen3-VL-2B-Instruct](github.com/QwenLM/Qwen3-VL) to provide **exceptionally fast** and high-quality, instruction-based image manipulation.

## Features

- **Text-Guided Editing**: Edit images using natural language instructions (e.g., "Add a cat on the sofa").
- **Compact Size**: 1.6B params diffusion and 2B condition encoder.
- **High-Speed Inference**: Powered by Sana1.5's efficient linear attention mechanism, enabling rapid image editing.
- **Multimodal Understanding**: Uses Qwen3VL for strong visual understanding and Sana1.5 for high-fidelity image generation.
- **Flexible Pipeline**: Built on top of `diffusers` and `transformers`, making it easy to extend and customize.

## Installation

### Prerequisites

- Linux
- Python 3.11
- Conda (recommended)

### Setup Environment

1. **Create and activate a Conda environment**:

    ```bash
    conda create -y -q --prefix ./vibe_env python=3.11
    conda activate ./vibe_env
    ```

2. **Install CUDA Toolkit (for NVIDIA GPUs)**:

    ```bash
    conda install -y -c nvidia/label/cuda-12.3.0 --override-channels cuda-compiler
    conda install -y -c nvidia/label/cuda-12.3.0 --override-channels cuda-toolkit
    ```

3. **Install Dependencies**:

    ```bash
    pip install -r requirements/requirements.txt
    ```

4. **Install Package**:

    ```bash
    pip install .
    ```
   
### Docker Setup

Alternatively, you can use Docker to run the project without installing dependencies locally.

1. **Prerequisites**:
    - [Docker](docs.docker.com/get-docker/)
    - [NVIDIA Container Toolkit](docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

2. **Build and Run**:

    ```bash
    # Build and start the container
    docker compose up -d --build

    # Enter the container
    docker compose exec vibe bash
    ```

    Inside the container, you can run the inference scripts as usual. The current directory is mounted to `/app`, so changes are reflected immediately.

## Usage

### Single Image processing

You can use the provided shell script to edit single images directly from the terminal.

```bash
chmod +x scripts/inference_template.sh
./scripts/inference_template.sh
```

Alternatively, you can call the Python script directly:

```bash
PYTHONPATH="${PYTHONPATH:-}:$(pwd)" \
python scripts/inference.py edit-single-image \
    --image-path "examples/dog.jpg" \
    --instruction "Make the dog look like a painting" \
    --checkpoint-path "/path/to/pipeline/checkpoint" \
    --output-path "outputs/" \
    --num-images-per-prompt 1 \
    --image-guidance-scale 1.2 \
    --guidance-scale 4.5 \
    --num-inference-steps 20 \
    --device "cuda:0"
```

### Multiple Images processing

For processing multiple images, you can use the `edit-multiple-images` command. This allows you to provide a list of images and prompts via a JSON file.

1. **Create a mapping file (e.g., `mapping.json`)**:

    ```json
    [
        {
            "image_path": "examples/dog.jpg",
            "editing_prompt": "Make the dog look like a painting"
        },
        {
            "image_path": "examples/dog.jpg",
            "editing_prompt": "turn the dog into a cat"
        }
    ]
    ```

2. **Run the batch command**:

```bash
PYTHONPATH="${PYTHONPATH:-}:$(pwd)" \
python scripts/inference.py edit-multiple-images \
    --mapping-path examples/mapping.json \
    --checkpoint-path "/path/to/pipeline/checkpoint" \
    --output-path "outputs/multi_img_results" \
    --num-images-per-prompt 1 \
    --image-guidance-scale 1.2 \
    --guidance-scale 4.5 \
    --num-inference-steps 20 \
    --device "cuda:0"
```

**Arguments:**

- `--image-path`: Path to the input image file (single image mode).
- `--instruction`: The text instruction or instructions for editing (single image mode).
- `--mapping-path`: Path to the JSON mapping file (batch mode).
- `--checkpoint-path`: Path to the local pipeline checkpoint (directory containing weights).
- `--output-path`: Directory where the result will be saved.
- `--num-images-per-prompt`: Number of variations to generate (default: 1).
- `--image-guidance-scale`: Controls the influence of the input image (default: 1.2).
- `--guidance-scale`: Controls the strength of the text prompt guidance (default: 4.5).
- `--num-inference-steps`: Number of denoising steps (default: 20).
- `--device`: The device to use.

## Project Structure

- **`vibe/`**: Core source code.
  - `editor.py`: Main `ImageEditor` class for high-level interaction.
  - `generative_pipeline/`: Sana1.5-based diffusion pipeline logic.
  - `transformer/`: Custom transformer models and editing head.
- **`scripts/`**: Utility scripts.
  - `inference.py`: CLI entry point for image editing.

## Acknowledgements

This project builds upon the work of:

- [Qwen3-VL](github.com/QwenLM/Qwen3-VL)
- [Sana: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformer](github.com/NVlabs/Sana)

## Citation

```bibtex
@misc{vibe2025,
  author = {Grigorii Alekseenko, Aleksandr Gordeev, Irina Tostykh, Bulat Suleimanov, Vladimir Dokholyan, Georgii Fedorov, Sergey Yakubson, Aleksandra Tsybina, Mikhail Chernyshov, Maksim Kuprashevich},
  title = {VIBE: Visual Instruction Based Editor},
  year = {2025},
}
```
