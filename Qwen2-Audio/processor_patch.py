"""
Processor patches registry for model-specific fixes.
Allows MCL wrapper to remain generic while supporting model-specific processor modifications.
"""
from transformers.models.qwen2_audio.processing_qwen2_audio import Qwen2AudioProcessor
from transformers.feature_extraction_utils import BatchFeature
import numpy as np
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def patch_qwen2audio_processor():
    """
    Patches the Qwen2AudioProcessor to use the local transformers version
    that properly handles chunk_length parameter.
    """    
    # Store the original __call__ method
    original_call = Qwen2AudioProcessor.__call__
    
    def patched_call(
        self,
        text=None,
        audios=None,
        padding=False,
        sampling_rate=None,
        **kwargs
    ):

        """
        Main method to prepare for the model one or several sequences(s) and audio(s). This method forwards the `text`
        and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the audio(s), this method forwards the `audios` and `kwrags` arguments to
        WhisperFeatureExtractor's [`~WhisperFeatureExtractor.__call__`] if `audios` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            audios (`np.ndarray`, `List[np.ndarray]`):
                The audio or batch of audios to be prepared. Each audio can be a NumPy array.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:
                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                lengths).
            sampling_rate (`int`, defaults to 16000):
                The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        """
        if "max_length_text" in kwargs:
            max_length_text = kwargs.pop("max_length_text")
        else:
            max_length_text = None
        if "max_length_audio" in kwargs:
            max_length_audio = kwargs.pop("max_length_audio")
        else:
            max_length_audio = None

        if text is None:
            raise ValueError("You need to specify either a `text` input to process.")
        elif isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        # ensure we have as much audios as audio tokens
        num_audio_tokens = sum(sample.count(self.audio_token) for sample in text)
        if audios is None:
            num_audios = 0
        else:
            num_audios = 1 if type(audios) == np.ndarray else len(audios)
        if num_audio_tokens != num_audios:
            raise ValueError(
                f"Found {num_audio_tokens} {self.audio_token} token{'s' if num_audio_tokens > 1 else ''} in provided text but received {num_audios} audio{'s' if num_audios > 1 else ''}"
            )

        if audios is not None:
            ### TODO: set to orig
            audio_inputs = self.feature_extractor(
                audios, sampling_rate=sampling_rate, return_attention_mask=True, padding="max_length", **kwargs
            )
            if "chunk_length" in kwargs:
                kwargs.pop("chunk_length")

            ### TODO: MODIFIED
            # audio_inputs = self.feature_extractor(
                # audios, sampling_rate=sampling_rate, return_attention_mask=True, padding="max_length", max_length=max_length_audio if max_length_audio is not None else None, **kwargs
            # )
            # audio_inputs = self.feature_extractor(
                # audios, sampling_rate=sampling_rate, return_attention_mask=True, **kwargs
            # )
            audio_inputs["feature_attention_mask"] = audio_inputs.pop(
                "attention_mask"
            )  # rename attention_mask to prevent conflicts later on

            expanded_text = []
            audio_lengths = audio_inputs["feature_attention_mask"].sum(-1).tolist()

            for sample in text:
                replace_str = []
                while self.audio_token in sample:
                    audio_length = audio_lengths.pop(0)
                    input_length = (audio_length - 1) // 2 + 1
                    num_audio_tokens = (input_length - 2) // 2 + 1

                    expanded_audio_token = self.audio_token * num_audio_tokens

                    audio_token_start_idx = sample.find(self.audio_token)
                    audio_token_end_idx = audio_token_start_idx + len(self.audio_token)

                    has_bos = (
                        sample[audio_token_start_idx - len(self.audio_bos_token) : audio_token_start_idx]
                        == self.audio_bos_token
                    )
                    has_eos = (
                        sample[audio_token_end_idx : audio_token_end_idx + len(self.audio_eos_token)]
                        == self.audio_eos_token
                    )

                    # Check if this audio token is surrounded by bos/eos tokens
                    if not has_bos and not has_eos:
                        expanded_audio_token = self.audio_bos_token + expanded_audio_token + self.audio_eos_token

                    replace_str.append(expanded_audio_token)
                    sample = sample.replace(self.audio_token, "<placeholder>", 1)

                while "<placeholder>" in sample:
                    sample = sample.replace("<placeholder>", replace_str.pop(0), 1)
                expanded_text.append(sample)
            text = expanded_text

        # TODO: MODIFIED: set to orig
        inputs = self.tokenizer(text, padding=padding, **kwargs)

        # inputs = self.tokenizer(text, padding=padding, max_length=max_length_text if max_length_text is not None else None, padding_side="right", **kwargs)

        if audios is not None:
            inputs.update(audio_inputs)

        return BatchFeature(data={**inputs})

    # Mark as patched to avoid double-patching
    patched_call._mcl_patched = True
    
    # Apply the patch
    Qwen2AudioProcessor.__call__ = patched_call
    
    logger.info("Successfully patched Qwen2AudioProcessor")
    