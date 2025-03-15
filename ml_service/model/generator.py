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
    create_semcoarsetosem_transformer_from_config,
    create_encodec_from_config,
    create_hubert_kmeans_from_config,
    my_load_model_config, load_model_config
)

try:
    # Try to import the original function if it exists
    from open_musiclm.config import create_fine_transformer_from_config
except ImportError:
    # If it doesn't exist, add it here
    from open_musiclm.open_musiclm import create_fine_transformer
    from open_musiclm.utils import exists, default, beartype_jit
    import torch

    @beartype_jit
    def create_fine_transformer_from_config(config, checkpoint_path=None, device=None):
        """
        Create a fine stage transformer from config, optionally load weights from checkpoint.
        Adapted specifically for the model config format used in this project.
        """
        # Use the exact fine_cfg parameters from the model configs
        try:
            if hasattr(config, "fine_cfg"):
                print("Found fine_cfg in config, using those parameters")
                dim = getattr(config.fine_cfg, "dim", 1024)
                depth = getattr(config.fine_cfg, "depth", 24)
                heads = getattr(config.fine_cfg, "heads", 16)
                attn_dropout = getattr(config.fine_cfg, "attn_dropout", 0.0)
                ff_dropout = getattr(config.fine_cfg, "ff_dropout", 0.1)
                grad_shrink_alpha = getattr(config.fine_cfg, "grad_shrink_alpha", 0.1)
                non_causal_prefix_size = getattr(config.fine_cfg, "non_causal_prefix_size", 0)
                relative_position_bias_type = getattr(config.fine_cfg, "relative_position_bias_type", "continuous")
                use_memory_efficient_attention = getattr(config.fine_cfg, "use_memory_efficient_attention", False)
            else:
                # Use parameters matching the weights file (24000.pt likely means 24 layers)
                print("No fine_cfg found, using defaults for large model")
                dim = 1024
                depth = 24
                heads = 16
                attn_dropout = 0.0
                ff_dropout = 0.1
                grad_shrink_alpha = 0.1
                non_causal_prefix_size = 0
                relative_position_bias_type = "continuous"
                use_memory_efficient_attention = False
            
            # Find codebook sizes and quantizer counts from config
            try:
                # Try accessing fields from different possible config locations
                clap_codebook_size = getattr(config.clap_rvq_cfg, "codebook_size", 1024)
                acoustic_codebook_size = getattr(config.encodec_cfg, "codebook_size", 1024)
                num_clap_quantizers = getattr(config.clap_rvq_cfg, "rq_num_quantizers", 12)
                num_coarse_quantizers = getattr(config.global_cfg, "num_coarse_quantizers", 4)
                num_fine_quantizers = getattr(config.global_cfg, "num_fine_quantizers", 8)
            except Exception as e:
                print(f"Error accessing codebook sizes: {e}, using defaults")
                clap_codebook_size = 1024
                acoustic_codebook_size = 1024
                num_clap_quantizers = 12
                num_coarse_quantizers = 4
                num_fine_quantizers = 8
            
            device = default(device, 'cuda' if torch.cuda.is_available() else 'cpu')
            
            print(f"Creating fine transformer with: dim={dim}, depth={depth}, heads={heads}")
            print(f"Codebook sizes: clap={clap_codebook_size}, acoustic={acoustic_codebook_size}")
            print(f"Quantizers: clap={num_clap_quantizers}, coarse={num_coarse_quantizers}, fine={num_fine_quantizers}")
            
            # Create transformer with full parameters
            transformer = create_fine_transformer(
                dim=dim,
                depth=depth,
                heads=heads,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                clap_codebook_size=clap_codebook_size,
                acoustic_codebook_size=acoustic_codebook_size,
                num_clap_quantizers=num_clap_quantizers,
                num_coarse_quantizers=num_coarse_quantizers,
                num_fine_quantizers=num_fine_quantizers,
                grad_shrink_alpha=grad_shrink_alpha,
                non_causal_prefix_size=non_causal_prefix_size,
                relative_position_bias_type=relative_position_bias_type,
                use_memory_efficient_attention=use_memory_efficient_attention
            ).to(device)
            
            # Try to load the weights
            if exists(checkpoint_path):
                print(f"Loading fine transformer weights from {checkpoint_path}")
                try:
                    # Try loading weights with strict=True first
                    state_dict = torch.load(checkpoint_path, map_location=device)
                    transformer.load_state_dict(state_dict)
                    print("Successfully loaded weights with strict=True")
                except RuntimeError as e:
                    print(f"Error loading weights with strict=True: {e}")
                    print("Trying with strict=False...")
                    try:
                        transformer.load_state_dict(state_dict, strict=False)
                        print("Successfully loaded weights with strict=False, some weights may be missing or unused")
                    except Exception as e2:
                        print(f"Failed even with strict=False: {e2}")
                        raise
            
            return transformer
        except Exception as e:
            print(f"Error in create_fine_transformer_from_config: {e}")
            import traceback
            traceback.print_exc()
            raise

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
            "fine": self.weights_dir / "fine.transformer.24000.pt",  # Updated to match your actual file
        }

        self.device = 'cpu'
        self._initialize_models()

    def _initialize_models(self):
        """Initialize all required models and stages"""
        # Helper function to debug config structure
        def print_config_structure(config, prefix=""):
            """Print the structure of a config object to help debug"""
            if hasattr(config, "__dict__"):
                for key, value in config.__dict__.items():
                    if not key.startswith("_"):  # Skip private attributes
                        if hasattr(value, "__dict__"):
                            print(f"{prefix}{key}: {type(value)}")
                            print_config_structure(value, prefix + "  ")
                        else:
                            print(f"{prefix}{key}: {value}")
        
        # Print model config structure for debugging
        print("Model config structure:")
        print_config_structure(self.model_config)
        
        self.wav2vec = create_hubert_kmeans_from_config(
            self.my_model_config, self.checkpoint_paths["kmeans"], self.device
        )
        
        self.neural_codec = create_encodec_from_config(
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
        try:
            if Path(self.checkpoint_paths["fine"]).exists():
                print(f"Fine stage weights found at {self.checkpoint_paths['fine']}, attempting to initialize...")
                
                # First attempt - with our custom implementation
                try:
                    self.fine_transformer = create_fine_transformer_from_config(
                        self.model_config, self.checkpoint_paths["fine"], self.device
                    )
                except Exception as e:
                    print(f"Failed to initialize fine transformer with custom implementation: {e}")
                    print("Trying alternative model configurations...")
                    
                    # Try different model configurations
                    configs = [
                        # Large config
                        {"dim": 1024, "depth": 24, "heads": 16},
                        # Small_large config
                        {"dim": 1024, "depth": 24, "heads": 16},
                        # Small config
                        {"dim": 1024, "depth": 6, "heads": 8}
                    ]
                    
                    for i, cfg in enumerate(configs):
                        try:
                            print(f"Trying config {i+1}: {cfg}")
                            # Create a basic fine transformer with specific config
                            self.fine_transformer = create_fine_transformer(
                                dim=cfg["dim"],
                                depth=cfg["depth"],
                                heads=cfg["heads"],
                                clap_codebook_size=1024,
                                acoustic_codebook_size=1024,
                                num_clap_quantizers=12,
                                num_coarse_quantizers=4,
                                num_fine_quantizers=8,
                            ).to(self.device)
                            
                            # Try loading the weights with strict=False
                            state_dict = torch.load(self.checkpoint_paths["fine"], map_location=self.device)
                            self.fine_transformer.load_state_dict(state_dict, strict=False)
                            print(f"Successfully created transformer with config {i+1} using strict=False")
                            break
                        except Exception as e2:
                            print(f"Failed with config {i+1}: {e2}")
                            continue
                
                # If we have a transformer successfully created, initialize the fine stage
                if self.fine_transformer is not None:
                    self.fine_stage = FineStage(
                        fine_transformer=self.fine_transformer,
                        neural_codec=self.neural_codec,
                        clap=self.clap
                    )
                    print(f"Fine stage initialized successfully")
                else:
                    print("Failed to initialize fine transformer after multiple attempts")
            else:
                print(f"Fine stage weights not found at {self.checkpoint_paths['fine']}. Fine stage will be disabled.")
        except Exception as e:
            print(f"Warning: Failed to initialize fine stage: {str(e)}")
            import traceback
            traceback.print_exc()

        self.semcoarsetosem_stage = SemcoarsetosemStage(
            semcoarsetosem_transformer=self.semcoarsetosem_transformer,
            neural_codec=self.neural_codec,
            wav2vec=self.wav2vec,
        )
        self.coarse_stage = CoarseStage(
            coarse_transformer=self.coarse_transformer,
            neural_codec=self.neural_codec,
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

        audio_for_encodec = resample(data, sample_hz, self.neural_codec.sample_rate)
        audio_for_wav2vec = resample(normalized_data, sample_hz, self.wav2vec.target_sample_hz)

        audio_for_encodec = int16_to_float32(float32_to_int16(audio_for_encodec)).to(self.device)
        audio_for_wav2vec = int16_to_float32(float32_to_int16(audio_for_wav2vec)).to(self.device)

        vocals_semantic_token_ids = get_or_compute_semantic_token_ids(
            None, audio_for_wav2vec, self.wav2vec
        )
        vocals_coarse_token_ids, _ = get_or_compute_acoustic_token_ids(
            None, None, audio_for_encodec, self.neural_codec, 
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
                print("Using fine stage for higher quality generation...")
                # Generate fine tokens from coarse tokens
                print(f"Generating fine tokens from coarse tokens with shape: {generated_coarse_tokens.shape}")
                generated_fine_tokens = self.fine_stage.generate(
                    clap_token_ids=clap_token_ids,
                    coarse_token_ids=generated_coarse_tokens,
                    max_time_steps=duration * time_steps_factor,
                    include_eos_in_output=False,
                    append_eos_to_conditioning_tokens=True,
                    temperature=temperature * 0.5,  # Lower temperature for fine stage
                )
                
                print(f"Fine tokens generated successfully, shape: {generated_fine_tokens.shape}")
                
                # Try to reconstruct wave from combined tokens
                try:
                    print("Reconstructing audio from fine tokens...")
                    # Combine coarse and fine tokens for neural codec
                    all_tokens = torch.cat((generated_coarse_tokens, generated_fine_tokens), dim=-1)
                    print(f"Combined tokens shape: {all_tokens.shape}")
                    generated_wave = self.neural_codec.decode_from_codebook_indices(all_tokens)
                    print(f"Generated wave shape before reshape: {generated_wave.shape}")
                    
                    # Only reshape if it's 2D, otherwise keep as is
                    if len(generated_wave.shape) == 2:
                        generated_wave = rearrange(generated_wave, 'b n -> b 1 n').detach().cpu()
                    else:
                        generated_wave = generated_wave.detach().cpu()
                        
                    print(f"Final wave shape: {generated_wave.shape}")
                    print("Successfully reconstructed audio with fine tokens")
                except Exception as e:
                    print(f"Error reconstructing with fine tokens: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    print("Falling back to coarse reconstruction")
                    # Fall back to coarse only if fine stage reconstruction fails
                    generated_wave = self.coarse_stage.generate(
                        clap_token_ids=clap_token_ids,
                        semantic_token_ids=generated_inst_semantic_ids.squeeze(2),
                        max_time_steps=duration * time_steps_factor,
                        reconstruct_wave=True,
                    )
                    generated_wave = rearrange(generated_wave, 'b n -> b 1 n').detach().cpu()
            except Exception as e:
                print(f"Fine stage generation failed: {str(e)}. Falling back to coarse only.")
                import traceback
                traceback.print_exc()
                # Fall back to coarse only if fine stage reconstruction fails
                generated_wave = self.coarse_stage.generate(
                    clap_token_ids=clap_token_ids,
                    semantic_token_ids=generated_inst_semantic_ids.squeeze(2),
                    max_time_steps=duration * time_steps_factor,
                    reconstruct_wave=True,
                )
                
                print(f"Coarse wave shape before reshape: {generated_wave.shape}")
                # Only reshape if it's not already in the right format
                if len(generated_wave.shape) == 2:
                    generated_wave = rearrange(generated_wave, 'b n -> b 1 n').detach().cpu()
                else:
                    generated_wave = generated_wave.detach().cpu()
                    
                print(f"Final coarse wave shape: {generated_wave.shape}")
        else:
            # Just use coarse tokens for encoding
            print("Using coarse tokens only (fine stage not requested)")
            generated_wave = self.coarse_stage.generate(
                clap_token_ids=clap_token_ids,
                semantic_token_ids=generated_inst_semantic_ids.squeeze(2),
                max_time_steps=duration * time_steps_factor,
                reconstruct_wave=True,
            )
            
            # Only reshape if it's not already in the right format
            if len(generated_wave.shape) == 2:
                generated_wave = rearrange(generated_wave, 'b n -> b 1 n').detach().cpu()
            else:
                generated_wave = generated_wave.detach().cpu()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{audio_path.stem}_{timestamp}_generated.wav"
        output_path = output_dir / output_name

        torchaudio.save(output_path, generated_wave[0], self.neural_codec.sample_rate)

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
    parser.add_argument("--prompt", type=str, default="Add a bass line, harmony, and drums, and remove the piano melody to leave space for the soloist.")
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