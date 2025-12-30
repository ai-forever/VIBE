"""Image editor class."""

import random

import numpy as np
import torch
from loguru import logger
from PIL import Image

from src.generative_pipeline import VIBESanaEditingPipeline
from src.utils import retry_decorator
from src.utils.img_utils import get_multiscale_transform, postprocess_padded_image, revert_resize

MAX_SEED = np.iinfo(np.int32).max


def randomize_seed_fn(seed: int, *, randomize_seed: bool = False) -> int:
    """Randomize the seed.

    Args:
        seed (int): Seed.
        randomize_seed (bool): Whether to randomize the seed.

    Returns:
        int: Randomized seed.
    """
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)  # noqa: S311
    return seed


class ImageEditor:
    """Image editor class."""

    def __init__(
        self,
        checkpoint_path: str,
        image_guidance_scale: float = 1.5,
        guidance_scale: float = 4.5,
        num_inference_steps: int = 20,
        device: str = "cuda:0",
    ) -> None:
        """Initialize the image editor.

        Args:
            checkpoint_path (str): The path to the local model checkpoint.
            image_guidance_scale (float): The image guidance scale.
            guidance_scale (float): The guidance scale.
            num_inference_steps (int): The number of inference steps.
            device (str): The device to use.
        """
        self.weight_dtype = torch.bfloat16
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.image_guidance_scale = image_guidance_scale
        self.guidance_scale = guidance_scale
        self.get_generation_pipe(checkpoint_path)

    @retry_decorator(logger=logger, delay=1)
    def get_generation_pipe(self, checkpoint_path: str) -> None:
        """Get the generation pipe.

        Args:
            checkpoint_path (str): The path to the pipeline checkpoint.
        """
        self.pipe = VIBESanaEditingPipeline.from_pretrained(checkpoint_path, torch_dtype=self.weight_dtype).to(self.device)

    def prepare_image_for_diffusion(self, image: Image.Image) -> tuple[Image.Image, tuple[int, ...] | None]:
        """Prepare the image for diffusion.

        Args:
            image (Image.Image): The input image.

        Returns:
            tuple[Image.Image, tuple[int, ...] | None]: The prepared image and the crop coordinates.
        """
        inp_image = image.copy()
        ori_h, ori_w = inp_image.height, inp_image.width

        # multiscale when we need to calculate the closest aspect ratio and resize & crop image
        closest_size = self.pipe.get_closest_size(ori_h, ori_w)

        transform, crop_coords = get_multiscale_transform(
            closest_size=closest_size,
            orig_size=(ori_h, ori_w),
            resize_factor=self.pipe.vae_scale_factor,
            out_img_type="pil",
        )
        inp_image = transform(inp_image)
        return inp_image, crop_coords

    @torch.inference_mode()
    def generate_edited_image(  # noqa: PLR0913
        self,
        instruction: str | list[str],
        *,
        conditioning_image: Image.Image | None = None,
        randomize_seed: bool = False,
        do_revert_resize: bool = True,
        seed: int = 0,
        num_images_per_prompt: int = 1,
        t2i_height: int | None = None,
        t2i_width: int | None = None,
    ) -> list[Image.Image]:
        """Generate an image based on the provided image and text prompt.

        Args:
            instruction (str | list[str]): Generation instruction or list of instructions.
            conditioning_image (Image.Image | None): Input image to edit or None if t2i generation is enabled.
            randomize_seed (bool): Whether to randomize the seed.
            do_revert_resize (bool): resize generated image to initial size.
            seed (int): Seed for random number generation. Defaults to 0.
            num_images_per_prompt (int): number of images to generate for each prompt
            t2i_height (int | None): The height of the t2i image.
            t2i_width (int | None): The width of the t2i image.

        Returns:
            list[Image.Image]: The edited images.
        """
        # Randomize seed
        seed = randomize_seed_fn(seed, randomize_seed=randomize_seed)

        # Prepare input image for generation
        if conditioning_image is not None:
            orig_height, orig_width = conditioning_image.height, conditioning_image.width
            inp_image, crop_coords = self.prepare_image_for_diffusion(conditioning_image)
            inp_height, inp_width = inp_image.size[1], inp_image.size[0]
        else:
            if t2i_height is None or t2i_width is None:
                logger.warning("Height and width for t2i generation are not provided, using default value 1024x1024")
                t2i_height = t2i_width = 1024
            inp_height, inp_width = self.pipe.get_closest_size(t2i_height, t2i_width)
            if (inp_height, inp_width) != (t2i_height, t2i_width):
                logger.warning(
                    f"The desired size of t2i generation {t2i_height}x{t2i_width} "
                    f"were adjusted to {inp_height}x{inp_width} to follow default binning scheme."
                )

        # Generate edited image
        generated_images = self.pipe(  # type: ignore
            conditioning_image=conditioning_image,
            prompt=instruction,
            height=inp_height,
            width=inp_width,
            num_inference_steps=self.num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            guidance_scale=self.guidance_scale,
            image_guidance_scale=self.image_guidance_scale,
            output_type="pil",
            generator=torch.Generator(device=self.device).manual_seed(seed),
        ).images

        # Return generated images if t2i generation is enabled
        if conditioning_image is None:
            return generated_images  # type: ignore[return-value]

        # remove paddings in case of letterbox usage
        generated_images = [postprocess_padded_image(im, crop_coords) for im in generated_images]  # type: ignore

        if do_revert_resize:
            return [revert_resize(im, (orig_height, orig_width)) for im in generated_images]
        return generated_images
