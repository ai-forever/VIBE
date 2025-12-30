"""VIBE Sana Editing pipeline."""

from typing import Any

import torch
from diffusers.image_processor import PixArtImageProcessor
from diffusers.models import AutoencoderDC
from diffusers.pipelines.sana.pipeline_output import SanaPipelineOutput
from diffusers.pipelines.sana.pipeline_sana import (
    ASPECT_RATIO_512_BIN,
    ASPECT_RATIO_1024_BIN,
    ASPECT_RATIO_2048_BIN,
    SanaPipeline,
    retrieve_timesteps,
)
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers.utils import is_torch_xla_available
from loguru import logger
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from src.transformer import VIBESanaEditingModel

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm  # type: ignore # pylint: disable

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


class VIBESanaEditingPipeline(SanaPipeline):
    """Image Editing Pipeline for Sana."""

    edit_query_template = "what will this image be like if {instruction}"
    t2i_query_template = "generate the image by description: {instruction}"
    min_pixels = 352 * 352  # min size of image in Qwen3VLProcessor
    max_pixels = 672 * 672  # max size of image in Qwen3VLProcessor
    bin_512_space = 512 * 512
    bin_1024_space = 1024 * 1024
    bin_2048_space = 2048 * 2048

    def __init__(
        self,
        tokenizer: Qwen3VLProcessor,
        text_encoder: Qwen3VLForConditionalGeneration,
        vae: AutoencoderDC,
        transformer: VIBESanaEditingModel,
        scheduler: DPMSolverMultistepScheduler,
    ) -> None:
        """Initialize the SanaIP2P pipeline.

        Args:
            tokenizer (Qwen3VLProcessor): The tokenizer for the text encoder.
            text_encoder (Qwen3VLForConditionalGeneration): The text encoder for the editing prompt.
            vae (AutoencoderDC): The VAE model.
            transformer (VIBESanaEditingModel): The editing transformer.
            scheduler (DPMSolverMultistepScheduler): The sana scheduler.
        """
        super(SanaPipeline, self).__init__()

        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.vae_scale_factor = (
            2 ** (len(self.vae.config.encoder_block_out_channels) - 1)
            if hasattr(self, "vae") and self.vae is not None
            else 32
        )
        self.image_processor = PixArtImageProcessor(vae_scale_factor=self.vae_scale_factor)

    @property
    def do_classifier_free_guidance(self) -> bool:
        """Property to check if classifier free guidance is enabled.

        Returns:
            bool: True if classifier free guidance is enabled, False otherwise.
        """
        return self._guidance_scale > 1

    @property
    def do_image_guidance(self) -> bool:
        """Property to check if image guidance is enabled.

        Returns:
            bool: True if image guidance is enabled, False otherwise.
        """
        return self._image_guidance_scale >= 1

    def get_closest_size(self, height: int, width: int) -> tuple[int, int]:
        """Get the closest size to the original size.

        Args:
            height (int): The height of the image.
            width (int): The width of the image.

        Returns:
            tuple[int, int]: The closest size to the original size.
        """
        sample_area = height * width
        if sample_area <= self.bin_512_space * 2 + 256**2:
            aspect_ratio_bin = ASPECT_RATIO_512_BIN
        elif sample_area <= self.bin_1024_space * 2 + 512**2:
            aspect_ratio_bin = ASPECT_RATIO_1024_BIN
        else:
            aspect_ratio_bin = ASPECT_RATIO_2048_BIN
        height, width = self.image_processor.classify_height_width_bin(height, width, ratios=aspect_ratio_bin)
        return height, width

    def prepare_inputs_batch(
        self,
        instruction: str | list[str],
        image: Image.Image | None = None,
    ) -> dict[str, torch.Tensor]:
        """Prepare inputs for the batch.

        Args:
            instruction (str | list[str]): Edit prompts.
            image (Image.Image | None): The input image or None if t2i generation is enabled.

        Returns:
            dict[str, torch.Tensor]: The inputs for the batch.
                - input_ids: The input IDs.
                - attention_mask: The attention mask.
                - image_grid_thw: The image grid size.
                - pixel_values: The pixel values.
        """
        # Prepare instructions
        instructions = [instruction] if isinstance(instruction, str) else instruction
        samples = []
        for instr in instructions:
            query = (
                self.edit_query_template.format(instruction=instr.strip())
                if image is not None
                else self.t2i_query_template.format(instruction=instr.strip())
            )
            user_content = [{"type": "image", "image": image}] if image is not None else []
            user_content.append({"type": "text", "text": query})
            message = [{"role": "user", "content": user_content}]
            text = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
            sample = {"text": text}
            if image is not None:
                sample["image"] = image
            samples.append(sample)

        # Invoke the processor to get the input ids and attention mask
        texts = [sample["text"] for sample in samples]
        images = [sample["image"] for sample in samples] if image is not None else None
        kwargs = {"min_pixels": self.min_pixels, "max_pixels": self.max_pixels}
        inputs = self.tokenizer(text=texts, images=images, padding=True, return_tensors="pt", **kwargs).data
        return {key: value.to(self._execution_device) for key, value in inputs.items()}

    def prepare_inputs_for_meta_queries(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        image_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Add meta queries into input embeddings before any padding tokens.

        The meta queries (stored in `self.transformer.meta_queries`) are inserted after all valid tokens
        (as indicated by attention_mask) but before padding tokens.
        The attention mask is updated with ones for the meta queries.

        Args:
            inputs_embeds (torch.Tensor): Input embeddings of shape (B, L, d_model).
            attention_mask (torch.Tensor): Attention mask of shape (B, L), with 1 for valid tokens and 0 for padding.
            image_mask (Optional[torch.Tensor]): Image mask of shape (B, L).

        Returns:
            tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                - Updated input embeddings of shape (B, L + N_queries, d_model)
                - Updated attention mask of shape (B, L + N_queries)
                - Updated image mask of shape (B, L + N_queries)
        """
        batch_size, _, _ = inputs_embeds.size()
        num_meta_queries = self.transformer.meta_queries.size(0)  # Number of meta query tokens

        # Lists to collect updated tensors for each sample.
        updated_embeds_list = []
        updated_attention_list = []
        updated_image_mask_list = []

        for batch_idx in range(batch_size):
            # Compute the number of valid tokens (non-padding) using the attention mask.
            valid_length = int(attention_mask[batch_idx].sum().item())

            # Insert meta queries after valid tokens and before any padding tokens.
            valid_embeds = inputs_embeds[batch_idx, :valid_length, :]  # (valid_length, d_model)
            pad_embeds = inputs_embeds[batch_idx, valid_length:, :]  # (pad_length, d_model)
            updated_embeds = torch.cat([valid_embeds, self.transformer.meta_queries, pad_embeds], dim=0)
            updated_embeds_list.append(updated_embeds)

            # Update the attention mask:
            valid_mask = attention_mask[batch_idx, :valid_length]  # (valid_length,) all ones
            pad_mask = attention_mask[batch_idx, valid_length:]  # (pad_length,) all zeros
            meta_mask = torch.ones(num_meta_queries, device=attention_mask.device, dtype=attention_mask.dtype)
            updated_attention = torch.cat([valid_mask, meta_mask, pad_mask], dim=0)
            updated_attention_list.append(updated_attention)

            if image_mask is not None:
                valid_image_mask = image_mask[batch_idx, :valid_length]  # (valid_length,) all ones
                pad_image_mask = image_mask[batch_idx, valid_length:]  # (pad_length,) all zeros
                meta_image_mask = torch.zeros(num_meta_queries, device=image_mask.device, dtype=image_mask.dtype)
                updated_image_mask = torch.cat([valid_image_mask, meta_image_mask, pad_image_mask], dim=0)
                updated_image_mask_list.append(updated_image_mask)

        # Stack lists to form batch tensors.
        updated_embeds = torch.stack(updated_embeds_list, dim=0)  # (B, L + N_queries, d_model)
        updated_attention_mask = torch.stack(updated_attention_list, dim=0)  # (B, L + N_queries)

        if updated_image_mask_list:  # noqa: SIM108
            updated_image_mask = torch.stack(updated_image_mask_list, dim=0)  # (B, L + N_queries)
        else:
            updated_image_mask = None  # type: ignore

        return updated_embeds, updated_attention_mask, updated_image_mask

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.LongTensor:
        """Get the rope index.

        Args:
            input_ids (torch.LongTensor): The input IDs of the model.
            image_grid_thw (torch.LongTensor | None): The grid size of the image.
            attention_mask (torch.Tensor | None): The attention mask of the model.

        Returns:
            torch.LongTensor: The position IDs.
        """
        attention_mask_tensor = (
            attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]  # type: ignore
        )
        if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:  # noqa: PLR2004
            attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
            # Only apply conversion for floating point tensors (inverted masks)
            if attention_mask_tensor.dtype.is_floating_point:
                attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                attention_mask_tensor = (1.0 - attention_mask_tensor).int()

        # Calculate RoPE index once per generation.
        position_ids, rope_deltas = self.text_encoder.model.get_rope_index(
            input_ids,
            image_grid_thw,
            None,
            attention_mask=attention_mask_tensor,
        )
        self.text_encoder.model.rope_deltas = rope_deltas
        return position_ids

    def prepare_initial_input_embeddings(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        image_grid_thw: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Prepare initial input embeddings for the model.

        Args:
            input_ids (torch.LongTensor): The input IDs of the model.
            attention_mask (torch.Tensor): The attention mask of the model.
            image_grid_thw (torch.LongTensor | None): The grid size of the image.
            pixel_values (torch.Tensor | None): The pixel values of the image.

        Returns:
            dict[str, torch.Tensor]: The model inputs:
                - inputs_embeds: The initial input embeddings.
                - deepstack_visual_embeds: The deepstack visual embeds or None if no image is provided.
                - position_ids: The position ids.
                - attention_mask: The attention mask.
                - visual_pos_masks: The visual pos masks or None if no image is provided.
        """
        inputs_embeds = self.text_encoder.get_input_embeddings()(input_ids)
        attention_mask = attention_mask.to(inputs_embeds.device)
        attention_mask = attention_mask.bool()

        if pixel_values is not None:
            image_embeds, deepstack_visual_embeds = self.text_encoder.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0)
            image_mask, _ = self.text_encoder.model.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
        else:
            visual_pos_masks = None
            deepstack_visual_embeds = None

        # Add meta queries to the input embeddings
        inputs_embeds, attention_mask, visual_pos_masks = self.prepare_inputs_for_meta_queries(  # type: ignore
            inputs_embeds,
            attention_mask,
            image_mask=visual_pos_masks,
        )

        # add placeholder for meta queries, it will be used to create position ids for meta queries only
        img_input_ids = torch.full(
            (input_ids.shape[0], self.transformer.meta_queries.size(0)),
            -1,
            device=input_ids.device,
            dtype=torch.long,
        )
        input_ids = torch.cat([input_ids, img_input_ids], dim=1)  # type: ignore

        # computing rope index
        position_ids = self.get_rope_index(input_ids, image_grid_thw, attention_mask)

        return {
            "inputs_embeds": inputs_embeds,
            "deepstack_visual_embeds": deepstack_visual_embeds,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "visual_pos_masks": visual_pos_masks,
        }

    def _get_editing_embeddings(
        self,
        edit_prompt: list[str] | str,
        conditioning_image: Image.Image | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the editing embeddings for the given prompt and image.

        Args:
            edit_prompt (list[str] | str): The editing prompt.
            conditioning_image (Image.Image | None): The conditioning image or None if t2i generation is enabled.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple of prompt embeds and negative prompt embeds.
        """
        # Prepare inputs for the text encoder call.
        prepared_inputs = self.prepare_inputs_batch(instruction=edit_prompt, image=conditioning_image)
        model_inputs = self.prepare_initial_input_embeddings(**prepared_inputs)  # type: ignore[arg-type]

        # Forward the text encoder.
        llm_model_result = self.text_encoder.language_model(input_ids=None, **model_inputs)
        hidden_states = llm_model_result.last_hidden_state
        attention_mask = model_inputs["attention_mask"]

        # Get the hidden states for the meta tokens
        meta_hidden_states = self.transformer.get_hidden_states_for_meta_tokens(hidden_states, attention_mask)

        # Forward the edit heads
        batch_size = meta_hidden_states.shape[0]
        return self.transformer.forward_edit_heads(batch_size, meta_hidden_states)

    def encode_prompt(  # type: ignore[override]  # noqa: PLR0913
        self,
        batch_size: int,
        prompt: str | list[str] | None = None,
        conditioning_image: Image.Image | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        num_images_per_prompt: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the prompt into text encoder hidden states.

        Args:
            batch_size (int): The batch size.
            conditioning_image: Image.Image: The conditioning image to pass in prompt encoder.
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                Number of images that should be generated per prompt. torch device to place the resulting embeddings on

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple of prompt embeds and negative prompt embeds.
        """
        if prompt_embeds is None:
            prompt_embeds, negative_prompt_embeds = self._get_editing_embeddings(
                edit_prompt=prompt,  # type: ignore
                conditioning_image=conditioning_image,
            )

        seq_len = prompt_embeds.size(1)
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if self.do_classifier_free_guidance:
            if negative_prompt_embeds is None:
                msg = "Negative prompt embeds are required when classifier free guidance is enabled."
                raise ValueError(msg)
            seq_len = negative_prompt_embeds.size(1)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            if self.do_image_guidance:
                prompt_embeds = torch.cat(  # type: ignore
                    [prompt_embeds, negative_prompt_embeds, negative_prompt_embeds],
                    dim=0,
                )
            else:
                prompt_embeds = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0)

        return prompt_embeds, negative_prompt_embeds  # type: ignore

    def prepare_guidance_inputs(
        self,
        image_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare the guidance inputs for the denoising process.

        Args:
            image_latents (torch.Tensor): The image latents to be used for the denoising process.
            prompt_embeds (torch.Tensor): The prompt embeds from edit head.
            negative_prompt_embeds (torch.Tensor): The negative prompt embeds from edit head.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple of image latents and prompt embeds for the denoising process.
        """
        if self.do_classifier_free_guidance:
            uncond_image_latents = torch.zeros_like(image_latents)
            image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)

            prompt_embeds = torch.cat(
                [
                    prompt_embeds,
                    negative_prompt_embeds,
                    negative_prompt_embeds,
                ],
                dim=0,
            )
        return image_latents, prompt_embeds

    def prepare_image_latents(
        self,
        image: torch.Tensor,
        batch_size: int,
        num_images_per_prompt: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Prepare the coonditioning image latents for the denoising process.

        Args:
            image (torch.Tensor): The image to process.
            batch_size (int): The batch size.
            num_images_per_prompt (int): The number of images to generate per prompt.
            dtype (torch.dtype): The dtype of the image.
            device (torch.device): The device to use.
        """
        image = image.to(device=device, dtype=dtype)
        batch_size = batch_size * num_images_per_prompt

        image_latents = self.vae.encode(image).latent
        image_latents = image_latents * self.vae.config.scaling_factor

        image_latents = torch.cat([image_latents] * batch_size, dim=0)

        if self.do_classifier_free_guidance:
            if self.do_image_guidance:
                uncond_image_latents = torch.zeros_like(image_latents)
                image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)
            else:
                image_latents = torch.cat([image_latents, image_latents], dim=0)
        return image_latents

    def _run_denoising_loop(  # noqa: PLR0913
        self,
        latents: torch.Tensor,
        latent_channels: int,
        input_image_latents: torch.Tensor | None,
        timesteps: torch.Tensor,
        num_inference_steps: int,
        num_warmup_steps: int,
        prompt_embeds: torch.Tensor,
        guidance_scale: float,
        image_guidance_scale: float,
        extra_step_kwargs: dict,
        *,
        is_t2i: bool = False,
    ) -> torch.Tensor:
        """Run the denoising loop over the given timesteps.

        Args:
            latents (torch.Tensor): Latents to denoise.
            latent_channels (int): The number of latent channels.
            input_image_latents (torch.Tensor | None): The input image latents or None if t2i generation is enabled.
            timesteps (torch.Tensor): The timesteps for the scheduler.
            num_inference_steps (int): The number of total inference steps.
            num_warmup_steps (int): The number of warmup steps for the scheduler.
            prompt_embeds (torch.Tensor): The prompt embeds from edit head.
            guidance_scale (float): The guidance scale for the text prompt.
            image_guidance_scale (float): The guidance scale for the image prompt.
            extra_step_kwargs (dict): The extra step kwargs for the scheduler.
            is_t2i (bool): Whether the t2i generation is enabled.

        Returns:
            torch.Tensor: The final latents after the denoising loop.
        """
        transformer_dtype = self.transformer.dtype
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                if self.do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents] * (3 if self.do_image_guidance else 2))
                else:
                    latent_model_input = latents

                # concat noised and input image latents.
                if input_image_latents is None:  # In case of t2i generation, we don't have input image latents
                    scaled_latent_model_input = torch.cat([latent_model_input, latent_model_input], dim=1)
                else:
                    scaled_latent_model_input = torch.cat([latent_model_input, input_image_latents], dim=1)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])
                timestep = timestep * self.transformer.config.timestep_scale

                # predict noise model_output
                noise_pred = self.transformer(
                    hidden_states=scaled_latent_model_input.to(dtype=transformer_dtype),
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    return_dict=False,
                    attention_kwargs=self.attention_kwargs,
                    t2i_samples=[is_t2i] * scaled_latent_model_input.shape[0],
                )[0]
                noise_pred = noise_pred.float()

                # perform guidance
                if self.do_classifier_free_guidance:
                    if self.do_image_guidance:
                        noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                        noise_pred = (
                            noise_pred_uncond
                            + guidance_scale * (noise_pred_text - noise_pred_image)
                            + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                        )
                    else:
                        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # learned sigma
                if self.transformer.config.out_channels // 2 == latent_channels:
                    noise_pred = noise_pred.chunk(2, dim=1)[0]

                # compute previous image: x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        return latents

    @torch.no_grad()
    def __call__(  # noqa: PLR0913,PLR0912,PLR0915
        self,
        conditioning_image: Image.Image | None = None,
        prompt: str | list[str] | None = None,
        prompt_embeds: torch.FloatTensor | None = None,
        negative_prompt_embeds: torch.FloatTensor | None = None,
        latents: torch.FloatTensor | None = None,
        height: int | None = None,
        width: int | None = None,
        eta: float = 0.0,
        num_inference_steps: int = 20,
        timesteps: list[int] | None = None,
        sigmas: list[float] | None = None,
        guidance_scale: float = 4.5,
        image_guidance_scale: float = 1.5,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        output_type: str = "pil",
        attention_kwargs: dict[str, Any] | None = None,
        *,
        return_dict: bool = True,
        use_resolution_binning: bool = True,
        **_: Any,
    ) -> SanaPipelineOutput | tuple:
        """Function invoked when calling the pipeline for generation.

        Args:
            conditioning_image (Image.Image | None): The input conditioning image or None if t2i generation is enabled.
            prompt (str | list[str] | None): The editing prompt.
            prompt_embeds (torch.FloatTensor): The prompt embeds from edit head.
            negative_prompt_embeds (torch.FloatTensor): The negative prompt embeds from edit head.
            latents (Optional[torch.FloatTensor]): The latents to use for the denoising process.
            height (Optional[int]): The height of the conditioning_image.
            width (Optional[int]): The width of the conditioning_image.
            num_inference_steps (int): The number of inference steps.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: huggingface.co/papers/2010.02502. Only
                applies to [`schedulers.DDIMScheduler`], will be ignored for others.
            guidance_scale (float): The guidance scale for the text prompt.
            image_guidance_scale (float): The guidance scale for the conditioning_image.
            num_images_per_prompt (int): The number of images to generate per prompt.
            generator (Optional[Union[torch.Generator, List[torch.Generator]]]): The generator.
            output_type (str): The output type.
            attention_kwargs (dict[str, Any]): The attention kwargs.
            return_dict (bool): Whether to return a return_dict.
            use_resolution_binning (bool): Whether to use resolution binning.
            _: Additional keyword arguments.

        Returns:
            SanaPipelineOutput: The output of the pipeline.
        """
        # check if we need to fallback to t2i generation
        is_t2i = False
        if conditioning_image is None:
            is_t2i = True
            logger.warning("Fallback to t2i generation because conditioning_image is not provided.")

        # 0. Set pipeline attributes.
        device = self._execution_device
        dtype = self.dtype  # type: ignore
        self._guidance_scale = guidance_scale
        self._image_guidance_scale = 0 if is_t2i else image_guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        if (height is None or width is None) and conditioning_image is None:
            msg = "Either height and width or conditioning_image must be provided."
            raise ValueError(msg)

        # 1. Check inputs. Raise error if not correct
        if use_resolution_binning:
            if height is None or width is None:
                height, width = conditioning_image.height, conditioning_image.width  # type: ignore[union-attr]
            orig_height, orig_width = height, width
            height, width = self.get_closest_size(height, width)

        self.check_inputs(
            prompt,
            height,
            width,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # Identify batch size.
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]  # type: ignore

        # 1. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(  # type: ignore[assignment]
            batch_size=batch_size,
            conditioning_image=conditioning_image,
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_images_per_prompt=num_images_per_prompt,
        )

        # 2. Prepare conditioning image latents
        if is_t2i:
            image_latents = None
        else:
            processed_image = self.image_processor.preprocess(conditioning_image, height=height, width=width)
            processed_image = processed_image.to(device).to(dtype)
            image_latents = self.prepare_image_latents(
                processed_image,
                batch_size,
                num_images_per_prompt,
                self.dtype,
                device,
            )

        # 3. Prepare latents
        latent_channels = self.vae.config.latent_channels
        latents = self.prepare_latents(  # type: ignore
            batch_size * num_images_per_prompt,
            latent_channels,
            height,
            width,
            dtype,
            device,
            generator,
            latents,
        )

        num_channels_latents_check = (
            latent_channels * 2 if self.transformer.input_condition_type == "channel_cat" else latent_channels
        )
        if num_channels_latents_check != self.transformer.config.in_channels:
            msg = f"The config of `pipeline.transformer` expects {self.transformer.config.in_channels} channels,"
            f"but received {num_channels_latents_check}."
            raise ValueError(msg)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)  # type: ignore[arg-type]
        self._num_timesteps = len(timesteps)  # type: ignore[arg-type]

        # 5. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6. Denoising loop
        latents = self._run_denoising_loop(  # type: ignore
            latents=latents,  # type: ignore[arg-type]
            latent_channels=latent_channels,
            input_image_latents=image_latents,
            timesteps=timesteps,  # type: ignore[arg-type]
            num_inference_steps=num_inference_steps,
            num_warmup_steps=num_warmup_steps,
            prompt_embeds=prompt_embeds,  # type: ignore[arg-type]
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
            extra_step_kwargs=extra_step_kwargs,
            is_t2i=is_t2i,
        )

        if output_type == "latent":
            image = latents  # type: ignore[assignment]
        else:
            latents = latents.to(self.dtype)  # type: ignore
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

            if use_resolution_binning:
                image = self.image_processor.resize_and_crop_tensor(image, orig_width, orig_height)

        if output_type != "latent":
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return SanaPipelineOutput(images=image)  # type: ignore[arg-type]
