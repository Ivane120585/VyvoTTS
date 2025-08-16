from typing import List
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from snac import SNAC


class VyvoTTSInference:
    """High-performance TTS inference engine using vLLM backend."""

    # Token ID constants
    TOKENIZER_LENGTH = 64400
    
    # Basic tokens
    START_OF_TEXT = 1
    END_OF_TEXT = 7
    
    # Speech tokens
    START_OF_SPEECH = TOKENIZER_LENGTH + 1  # 64401
    END_OF_SPEECH = TOKENIZER_LENGTH + 2    # 64402
    
    # Human/AI conversation tokens
    START_OF_HUMAN = TOKENIZER_LENGTH + 3   # 64403
    END_OF_HUMAN = TOKENIZER_LENGTH + 4     # 64404
    START_OF_AI = TOKENIZER_LENGTH + 5      # 64405
    END_OF_AI = TOKENIZER_LENGTH + 6        # 64406
    
    # Special tokens
    PAD_TOKEN = TOKENIZER_LENGTH + 7        # 64407
    AUDIO_TOKENS_START = TOKENIZER_LENGTH + 10  # 64410
    
    # For compatibility with existing code
    AUDIO_MARKER_TOKEN = AUDIO_TOKENS_START
    STOP_TOKEN_ID = END_OF_SPEECH
    AUDIO_OFFSET = AUDIO_TOKENS_START
    CODES_PER_GROUP = 7

    def __init__(
        self,
        model_name: str = "Vyvo/VyvoTTS-LFM2-Neuvillette",
        snac_model_name: str = "hubertsiuzdak/snac_24khz"
    ):
        """Initialize the TTS inference engine.

        Args:
            model_name: HuggingFace model identifier for the TTS model
            snac_model_name: HuggingFace model identifier for SNAC audio decoder
        """
        self.model_name = model_name
        self.engine = LLM(model=model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.snac_model = SNAC.from_pretrained(snac_model_name)

    def _extract_audio_tokens(self, generated_ids: torch.Tensor) -> torch.Tensor:
        """Extract audio tokens from generated sequence."""
        token_indices = (generated_ids == self.AUDIO_MARKER_TOKEN).nonzero(as_tuple=True)

        if len(token_indices[1]) > 0:
            last_occurrence_idx = token_indices[1][-1].item()
            return generated_ids[:, last_occurrence_idx + 1:]
        return generated_ids

    def _clean_tokens(self, tokens: torch.Tensor) -> List[torch.Tensor]:
        """Remove stop tokens and prepare for processing."""
        return [row[row != self.STOP_TOKEN_ID] for row in tokens]

    def _group_and_offset_codes(self, processed_rows: List[torch.Tensor]) -> List[List[int]]:
        """Group tokens into groups of 7 and apply offset correction."""
        code_lists = []

        for row in processed_rows:
            row_length = row.size(0)
            new_length = (row_length // self.CODES_PER_GROUP) * self.CODES_PER_GROUP
            trimmed_row = row[:new_length]
            code_lists.append([t.item() - self.AUDIO_OFFSET for t in trimmed_row])

        return code_lists

    def _redistribute_codes(self, code_list: List[int]) -> torch.Tensor:
        """Redistribute codes into SNAC layers and decode to audio."""
        num_groups = len(code_list) // self.CODES_PER_GROUP

        layer_1, layer_2, layer_3 = [], [], []

        for i in range(num_groups):
            base_idx = self.CODES_PER_GROUP * i

            layer_1.append(code_list[base_idx])
            layer_2.extend([
                code_list[base_idx + 1] - 4096,
                code_list[base_idx + 4] - (4 * 4096)
            ])
            layer_3.extend([
                code_list[base_idx + 2] - (2 * 4096),
                code_list[base_idx + 3] - (3 * 4096),
                code_list[base_idx + 5] - (5 * 4096),
                code_list[base_idx + 6] - (6 * 4096)
            ])

        codes = [
            torch.tensor(layer_1).unsqueeze(0),
            torch.tensor(layer_2).unsqueeze(0),
            torch.tensor(layer_3).unsqueeze(0)
        ]

        return self.snac_model.decode(codes)

    def parse_tokens_to_audio(self, generated_ids: torch.Tensor) -> List[torch.Tensor]:
        """Convert generated token IDs to audio waveforms.

        Args:
            generated_ids: Raw token IDs from model generation

        Returns:
            List of decoded audio tensors
        """
        # Extract audio portion of tokens
        cropped_tokens = self._extract_audio_tokens(generated_ids)

        # Clean and prepare tokens
        processed_rows = self._clean_tokens(cropped_tokens)

        # Group and apply offsets
        code_lists = self._group_and_offset_codes(processed_rows)

        # Decode to audio
        return [self._redistribute_codes(code_list) for code_list in code_lists]

def text_to_speech(prompt, voice=None):
    """
    Given a text prompt and optional voice, generates audio tokens using
    the Orpheus TTS model and decodes them into audio samples via SNAC.
    """

    # Construct the prompt with optional voice prefix
    if voice:
        adapted_prompt = f"{voice}: {prompt}"
        prompt_tokens = tokenizer(adapted_prompt, return_tensors="pt")
    else:
        prompt_tokens = tokenizer(prompt, return_tensors="pt")

    # Insert special tokens
    start_token = torch.tensor([[START_TOKEN_ID]], dtype=torch.int64)
    end_tokens = torch.tensor([END_TOKEN_IDS], dtype=torch.int64)
    all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)

    # Decode to string for LLM
    final_prompt = tokenizer.decode(all_input_ids[0])

    # Sampling parameters
    params = SamplingParams(
        temperature=0.6,
        top_p=0.8,
        max_tokens=1200,
        stop_token_ids=[STOP_TOKEN_ID],
        repetition_penalty=1.3
    )

    # Generate token IDs from the model
    outputs = engine.generate([final_prompt], [params])
    token_ids = outputs[0].outputs[0].token_ids
    generated_ids = torch.tensor([token_ids], dtype=torch.long)

    # Convert generated tokens into audio
    audio_samples = parse_tokens_to_audio(generated_ids, snac_model)
    return audio_samples


    # Example usage
audio_output = text_to_speech("Hello world", voice="zoe")
print("Decoded audio (tensors):", audio_output)
