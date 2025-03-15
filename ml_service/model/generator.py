import sys
from pathlib import Path
import torch
import torchaudio
import argparse
import json
import shutil
from datetime import datetime
import uuid
from einops import rearrange
from torchaudio.functional import resample
from typing import Optional

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from open_musiclm.config import (
    create_clap_quantized_from_config,
    create_coarse_transformer_from_config,
    create_fine_transformer_from_config,
    create_semcoarsetosem_transformer_from_config,
    create_encodec_from_config,
    create_hubert_kmeans_from_config,
    my_load_model_config, load_model_config
)
from open_musiclm.open_musiclm import (
    SemcoarsetosemStage, CoarseStage, FineStage,
    get_or_compute_clap_token_ids,
    get_or_compute_acoustic_token_ids,
    get_or_compute_semantic_token_ids
)
from open_musiclm.utils import int16_to_float32, float32_to_int16, zero_mean_unit_var_norm

class MusicGenerator:
    def __init__(self):
        self.base_dir = project_root
        self.weights_dir = self.base_dir / "model_weights"
        
        self.model_config = load_model_config(
            self.base_dir / "open_musiclm/configs/model/musiclm_large_small_context.json"
        )
        self.my_model_config = my_load_model_config(
            self.base_dir / "open_musiclm/configs/model/my_musiclm_for_semcoarsetosem.json"
        )

        self.checkpoint_paths = {
            "kmeans": self.weights_dir / "kmeans_10s_no_fusion.joblib",
            "clap": self.weights_dir / "clap.rvq.950_no_fusion.pt",
            "semcoarsetosem": self.weights_dir / "real_semcoarsetosem.transformer.5170.pt",
            "coarse": self.weights_dir / "coarse.transformer.18000.pt",
            "fine": self.weights_dir / "fine.transformer.18000.pt",  # Add fine transformer checkpoint path
        }

        self.device = 'cpu'
        self._initialize_models()

    def _initialize_models(self):
        """Initialize all required models and stages"""
        self.wav2vec = create_hubert_kmeans_from_config(
            self.my_model_config, self.checkpoint_paths["kmeans"], self.device
        )
        
        self.encodec_wrapper = create_encodec_from_config(
            self.my_model_config, self.device
        )

        self.clap = create_clap_quantized_from_config(
            self.my_model_config, self.checkpoint_paths["clap"], self.device
        )

        self.semcoarsetosem_transformer = create_semcoarsetosem_transformer_from_config(
            self.my_model_config, self.checkpoint_paths["semcoarsetosem"], self.device
        )
        self.coarse_transformer = create_coarse_transformer_from_config(
            self.model_config, self.checkpoint_paths["coarse"], self.device
        )
        
        # Initialize fine transformer if the weights file exists
        self.fine_transformer = None
        self.fine_stage = None
        if self.checkpoint_paths["fine"].exists():
            try:
                self.fine_transformer = create_fine_transformer_from_config(
                    self.model_config, self.checkpoint_paths["fine"], self.device
                )
                self.fine_stage = FineStage(
                    fine_transformer=self.fine_transformer,
                    neural_codec=self.encodec_wrapper,
                    clap=self.clap
                )
            except Exception as e:
                print(f"Warning: Failed to initialize fine stage: {str(e)}")

        self.semcoarsetosem_stage = SemcoarsetosemStage(
            semcoarsetosem_transformer=self.semcoarsetosem_transformer,
            neural_codec=self.encodec_wrapper,
            wav2vec=self.wav2vec,
        )
        self.coarse_stage = CoarseStage(
            coarse_transformer=self.coarse_transformer,
            neural_codec=self.encodec_wrapper,
            wav2vec=self.wav2vec,
            clap=self.clap
        )

    def process_audio(
        self,
        audio_path: Path,
        output_dir: Path,
        semantic_steps: int,
        duration: int,
        time_steps_factor: int,
        temperature: float,
        prompt: str = "Add a bass line, harmony, and drums, and remove the piano melody to leave space for the soloist.",
        use_fine_stage: bool = False,
        request_id: Optional[str] = None,
        save_for_eval: bool = False,
    ) -> Path:
        """Process audio file and generate accompaniment
        
        Args:
            audio_path: Path to input audio file
            output_dir: Directory to save output audio
            semantic_steps: Number of semantic token generation steps
            duration: Target duration in seconds
            time_steps_factor: Temporal resolution multiplier
            temperature: Generation temperature (higher = more creative/random)
            prompt: Text prompt to guide generation
            use_fine_stage: Whether to use the fine stage for higher quality audio
            request_id: Unique ID for this request
            save_for_eval: Whether to save generation for evaluation
            
        Returns:
            Path to the generated audio file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        data, sample_hz = torchaudio.load(audio_path)
        if data.shape[0] > 1:
            data = torch.mean(data, dim=0).unsqueeze(0)

        target_length = int(10 * sample_hz)
        normalized_data = zero_mean_unit_var_norm(data)
        data = data[:, :target_length]
        normalized_data = normalized_data[:, :target_length]

        audio_for_encodec = resample(data, sample_hz, self.encodec_wrapper.sample_rate)
        audio_for_wav2vec = resample(normalized_data, sample_hz, self.wav2vec.target_sample_hz)

        audio_for_encodec = int16_to_float32(float32_to_int16(audio_for_encodec)).to(self.device)
        audio_for_wav2vec = int16_to_float32(float32_to_int16(audio_for_wav2vec)).to(self.device)

        vocals_semantic_token_ids = get_or_compute_semantic_token_ids(
            None, audio_for_wav2vec, self.wav2vec
        )
        vocals_coarse_token_ids, _ = get_or_compute_acoustic_token_ids(
            None, None, audio_for_encodec, self.encodec_wrapper, 
            self.model_config.global_cfg.num_coarse_quantizers
        )
            
        clap_token_ids = get_or_compute_clap_token_ids(None, self.clap, None, [prompt])

        generated_inst_semantic_ids = self.semcoarsetosem_stage.generate(
            vocals_semantic_token_ids=vocals_semantic_token_ids,
            vocals_coarse_token_ids=vocals_coarse_token_ids,
            max_time_steps=semantic_steps,
            temperature=temperature,
        )

        generated_coarse_tokens = self.coarse_stage.generate(
            clap_token_ids=clap_token_ids,
            semantic_token_ids=generated_inst_semantic_ids.squeeze(2),
            max_time_steps=duration * time_steps_factor,
            reconstruct_wave=False,
            include_eos_in_output=False,
            append_eos_to_conditioning_tokens=True,
            temperature=temperature,
        )
        
        # Apply fine stage if requested and available
        if use_fine_stage and self.fine_stage is not None:
            try:
                generated_fine_tokens = self.fine_stage.generate(
                    clap_token_ids=clap_token_ids,
                    coarse_token_ids=generated_coarse_tokens,
                    max_time_steps=duration * time_steps_factor,
                    include_eos_in_output=False,
                    append_eos_to_conditioning_tokens=True,
                    temperature=temperature * 0.5,  # Typically we use lower temperature for fine stage
                )
                
                # Combine coarse and fine tokens for encoding
                all_tokens = torch.cat((generated_coarse_tokens, generated_fine_tokens), dim=-1)
                generated_wave = self.encodec_wrapper.decode_from_codebook_indices(all_tokens)
            except Exception as e:
                print(f"Fine stage generation failed: {str(e)}. Falling back to coarse only.")
                generated_wave = self.encodec_wrapper.decode_from_codebook_indices(generated_coarse_tokens)
        else:
            # Just use coarse tokens for encoding
            generated_wave = self.encodec_wrapper.decode_from_codebook_indices(generated_coarse_tokens)

        generated_wave = rearrange(generated_wave, 'b n -> b 1 n').detach().cpu()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{audio_path.stem}_{timestamp}_generated.wav"
        output_path = output_dir / output_name

        torchaudio.save(output_path, generated_wave[0], self.encodec_wrapper.sample_rate)

        if save_for_eval:
            save_dir = self.base_dir / "saved_generations"
            save_dir.mkdir(exist_ok=True)
            
            request_id = request_id or str(uuid.uuid4())
            shutil.copy2(audio_path, save_dir / f"{request_id}_input.wav")
            shutil.copy2(output_path, save_dir / f"{request_id}_output.wav")
            
            metadata = {
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
                "input_file": audio_path.name,
                "output_file": output_path.name,
                "parameters": {
                    "semantic_steps": semantic_steps,
                    "duration": duration,
                    "time_steps_factor": time_steps_factor,
                    "temperature": temperature,
                    "prompt": prompt,
                    "use_fine_stage": use_fine_stage
                }
            }
            
            with open(save_dir / f"{request_id}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

        return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--semantic-steps", type=int, default=2)
    parser.add_argument("--duration", type=int, default=3)
    parser.add_argument("--time-steps-factor", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--prompt", type=str, default="Diverse kinds of instrument and richness")
    parser.add_argument("--use-fine-stage", action="store_true", help="Use fine stage for higher quality audio")
    parser.add_argument("--save-for-eval", action="store_true")
    
    args = parser.parse_args()
    
    generator = MusicGenerator()
    generator.process_audio(
        audio_path=Path(args.input),
        output_dir=Path(args.output_dir),
        semantic_steps=args.semantic_steps,
        duration=args.duration,
        time_steps_factor=args.time_steps_factor,
        temperature=args.temperature,
        prompt=args.prompt,
        use_fine_stage=args.use_fine_stage,
        save_for_eval=args.save_for_eval
    )