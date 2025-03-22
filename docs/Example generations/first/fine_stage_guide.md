# ML Service for Harmony Explained

This directory contains the machine learning service of the Harmony Explained project. The ML service is responsible for generating musical accompaniments based on input audio and text prompts.

## Architecture

The ML service is built on top of the [Open-MusicLM](https://github.com/zhvng/open-musiclm) framework, an open-source implementation of Google's MusicLM model. The service uses several key components:

1. **CLAP**: Contrastive Language-Audio Pretraining model for processing text prompts
2. **HuBERT-KMeans**: For extracting semantic tokens from audio
3. **Encodec**: Neural audio codec for encoding/decoding audio data
4. **Transformer Models**: For token generation and sequence modeling

## Multi-Stage Generation Pipeline

The service implements a multi-stage generation pipeline:

1. **Semantic Stage**: Extracts musical meaning from input audio
2. **Coarse Stage**: Generates basic musical structure and instrument arrangement
3. **Fine Stage**: Adds detailed expression, timbre, and nuances (optional)

## Setup

### Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Model Weights

Download the required model weights and place them in the `model_weights/` directory:

#### Required
- `clap.rvq.950_no_fusion.pt`
- `kmeans_10s_no_fusion.joblib`
- `real_semcoarsetosem.transformer.5170.pt`
- `coarse.transformer.18000.pt`

#### Optional (for Fine Stage)
- `fine.transformer.24000.pt`

You can download these weights from the [OpenMusicLM experimental checkpoints](https://drive.google.com/drive/u/0/folders/1347glwEc-6XWulfU7NGrFrYTvTnjeVJE).

## Running the Service

```bash
uvicorn api.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.

## API Documentation

### POST /generate

Generates music based on an input audio file.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| audio_file | File | (required) | Input audio file to transform |
| semantic_steps | int | 5 | Number of semantic token generation steps |
| duration | int | 1 | Target duration in seconds |
| time_steps_factor | int | 4 | Temporal resolution multiplier |
| temperature | float | 0.85 | Controls randomness (higher = more creative) |
| prompt | string | "Add a bass line..." | Text description to guide generation |
| use_fine_stage | boolean | false | Whether to use fine stage for higher quality |
| save_for_eval | boolean | false | Whether to save results for evaluation |

**Response**:
- A WAV audio file containing the generated accompaniment

**Example**:
```python
import requests

url = "http://localhost:8000/generate"
files = {"audio_file": open("my_melody.wav", "rb")}
data = {
    "semantic_steps": "10",
    "duration": "15",
    "time_steps_factor": "6",
    "temperature": "0.9",
    "prompt": "Add a jazz trio accompaniment with piano, bass, and drums",
    "use_fine_stage": "true"
}

response = requests.post(url, files=files, data=data)
with open("generated_harmony.wav", "wb") as f:
    f.write(response.content)
```

### GET /health

Returns the health status of the service along with system information.

**Response**:
```json
{
  "status": "ok",
  "system_info": {
    "available_memory_gb": 6.42,
    "used_memory_percent": 58.3,
    "cpu_percent": 12.5,
    "torch_cuda_available": false,
    "torch_cuda_device_count": 0
  }
}
```

## Performance Considerations

### CPU vs GPU

The service runs on CPU by default, which is significantly slower than GPU. Consider using GPU if available, especially when enabling the Fine Stage.

### Memory Requirements

- **Minimum**: 8GB RAM
- **Recommended**: 16GB RAM
- **With Fine Stage**: 16GB+ RAM

### Generation Time Guidelines

| Configuration | Duration (sec) | Expected Processing Time |
|---------------|----------------|--------------------------|
| Coarse only   | 1-5            | 30-90 seconds            |
| Coarse only   | 5-20           | 2-5 minutes              |
| With Fine Stage | 1-5          | 1-3 minutes              |
| With Fine Stage | 5-20         | 5-15 minutes             |

## Advanced Configuration

### Customizing Model Parameters

The default parameters are set in `generator.py`. You can modify them to adjust the behavior of the model:

```python
self.checkpoint_paths = {
    "kmeans": self.weights_dir / "kmeans_10s_no_fusion.joblib",
    "clap": self.weights_dir / "clap.rvq.950_no_fusion.pt",
    "semcoarsetosem": self.weights_dir / "real_semcoarsetosem.transformer.5170.pt",
    "coarse": self.weights_dir / "coarse.transformer.18000.pt",
    "fine": self.weights_dir / "fine.transformer.24000.pt",
}
```

### Error Handling

The service includes robust error handling:

1. **Memory monitoring**: Warns if available memory is low
2. **Graceful degradation**: Falls back to coarse-only if fine stage fails
3. **Type checking**: Ensures parameters are of the correct type

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `duration`, `semantic_steps`, or disable Fine Stage
2. **Slow Generation**: Consider reducing parameters or disabling Fine Stage
3. **Model Loading Errors**: Verify weights files exist and have correct names

### Log Files

Check `app.log` for detailed error messages and diagnostics.

## References

- [OpenMusicLM Repository](https://github.com/zhvng/open-musiclm)
- [MusicLM Paper](https://arxiv.org/abs/2301.11325)
- [CLAP](https://github.com/LAION-AI/CLAP)
- [Encodec](https://github.com/facebookresearch/encodec)