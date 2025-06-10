import base64
import os
import warnings

# Note: temporarily suppressing warnings for vLLM
# Qwen2 models currently generate a bunch of warnings:  https://github.com/vllm-project/vllm/issues/13143#issuecomment-2656958661
# Once this issue is resolved, remove the suppressants
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

from io import BytesIO
from typing import List, Optional, Tuple, Union
from unittest.mock import patch

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from decord import VideoReader, cpu
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)
from transformers.utils import logging
from vllm import LLM, SamplingParams

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav_base64

logging.set_verbosity_error()
import warnings

warnings.filterwarnings("ignore")


try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"


@register_model("vllm")
class VLLM_Wrapper(lmms):
    """
    Wrapper to use VLLM to run inference
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        max_model_len=4096,
        tensor_parallel_size=1,
        max_num_seqs=5,
        min_pixels: int = 28 * 28,
        max_pixels: int = 360 * 28 * 28,
        max_num_frames: int = 32,
        fps: float = 1,
        force_sample: bool = False,
        disable_mm_preprocessor_cache: Optional[bool] = False,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.fps = fps
        self.force_sample = force_sample
        self.max_num_frames = max_num_frames
        self.batch_size_per_gpu = batch_size

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

        device_str = f"cuda:{accelerator.local_process_index}"
        self._device = torch.device(device_str)
        self.accelerator = accelerator

        self._model = LLM(
            model=pretrained,
            max_model_len=max_model_len,
            device=device_str,
            tensor_parallel_size=tensor_parallel_size,
            max_num_seqs=max_num_seqs,
            distributed_executor_backend="external_launcher" if self._world_size > 1 else None,
            mm_processor_kwargs={
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "fps": fps,
            },
            disable_mm_preprocessor_cache=disable_mm_preprocessor_cache,
        )

        accelerator.wait_for_everyone()

        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for VLLM_Wrapper")

    def flatten(self, input):
        new_list = []
        for i in input:
            if i is not None and hasattr(i, '__iter__'):
                for j in i:
                    new_list.append(j)
            elif i is not None:
                new_list.append(i)
        return new_list


    def load_video(self, video_path, max_frames_num, fps, force_sample=False):
        if max_frames_num == 0:
            return np.zeros((1, 336, 336, 3))
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        video_time = total_frame_num / vr.get_avg_fps()
        fps = round(vr.get_avg_fps() / fps)
        frame_idx = [i for i in range(0, len(vr), fps)]
        frame_time = [i / fps for i in frame_idx]
        if len(frame_idx) > max_frames_num or force_sample:
            sample_fps = max_frames_num
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i / vr.get_avg_fps() for i in frame_idx]
        frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
        spare_frames = vr.get_batch(frame_idx).asnumpy()

        return spare_frames, frame_time, video_time

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, tasks, splits = zip(*chunk)
            visuals = [dvf(self.task_dict[task][split][ids]) for dvf, ids, task, split in zip(doc_to_visual, doc_id, tasks, splits)]
            visuals = self.flatten(visuals)
            
            
            gen_kwargs = all_gen_kwargs[0]
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 4096
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0

            # Prepare messages
            messages = []
            all_processed_visuals = []
            for i, context in enumerate(contexts):
                # context += "\nPlease think step by step."
                # if "<image>" in context:
                #     context = context.replace("<image>", "")

                message = [{"role": "system", "content": "You are a helpful assistant."}]
                processed_visuals = []
                
                if len(visuals) > 0:
                    visual = visuals[i] if i < len(visuals) else None
                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov", ".MP4")):  # Video file
                        message.append({"role": "user", "content": [{"type": "video", "video": visual, "max_pixels": 360 * 420}, {"type": "text", "text": context}]})
                        video_frames, _, _ = self.load_video(visual, max_frames_num=self.max_num_frames, fps=self.fps, force_sample=self.force_sample)
                        processed_visuals.append({"video": video_frames})
                    elif isinstance(visual, Image.Image):  # Single image
                        message.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": context}]})
                        processed_visuals.append({"image": visual.convert("RGB")})
                    elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):  # Multiple images
                        raise NotImplementedError("not implemented for now")
                        # image_content = []
                        # for v in visual:
                        #     image_content.append({"type": "image"})
                        # message.append({"role": "user", "content": image_content + [{"type": "text", "text": context}]})
                    else:
                        message.append({"role": "user", "content": [{"type": "text", "text": context}]})
                else:
                    message.append({"role": "user", "content": [{"type": "text", "text": context}]})
                    utils.eval_logger.warning("added_message")
                messages.append(message)
                all_processed_visuals.append(processed_visuals)
            utils.eval_logger.warning("generated contexts")
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = [{"prompt": prompt} for pi, prompt in enumerate(text)] #[{"prompt": prompt, "multi_modal_data": all_processed_visuals[pi][0]} for pi, prompt in enumerate(text)]

            # Run generation
            sampling_params = SamplingParams(temperature=gen_kwargs["temperature"], max_tokens=gen_kwargs["max_new_tokens"], stop_token_ids=None)
            utils.eval_logger.warning("generating")
            outputs = self.model.generate(inputs, sampling_params=sampling_params, use_tqdm=False)

            # Collect results
            for o in outputs:
                generated_text = o.outputs[0].text
                res.append(generated_text)
                pbar.update(1)

        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res
        
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "GPT4V not support"

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
