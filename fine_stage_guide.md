# Fine Stage Implementation Guide for ML Web App Harmony Explained

This guide explains how to use the newly added Fine Stage functionality in the Harmony Explained application. The Fine Stage adds higher-quality audio refinement to your generated harmonies, adding more detailed timbre, harmonics, and nuances to the output.

## What is the Fine Stage?

As explained in the README, the audio generation process consists of several stages:

1. **HuBERT-KMeans**: Analyzes input audio and creates semantic tokens
2. **CLAP**: Processes text prompts into numerical embeddings
3. **Transformer**: Aligns audio tokens with text embeddings
4. **Coarse Stage**: Transforms tokens into coarse audio tokens (basic melody, rhythm, instrument structure)
5. **Fine Stage (NEW)**: Refines the coarse tokens to add detailed timbre, harmonics, and nuances
6. **Encodec**: Converts tokens into actual audio

The Fine Stage was disabled in the MVP version due to resource constraints, but is now available as an optional step.

## How to Use Fine Stage

### Backend Changes

The backend API now supports a new parameter:

```
use_fine_stage: boolean (default: false)
```

When set to `true`, the generation process will include the Fine Stage, producing higher-quality audio with more detailed timbre and expression.

### API Example

Here's how to call the API with fine stage enabled:

```javascript
// Create a form data object for the API request
const formData = new FormData();
formData.append('audio_file', audioBlob, 'recording.wav');
formData.append('semantic_steps', 20);
formData.append('duration', 10);
formData.append('time_steps_factor', 20);
formData.append('temperature', 0.85);
formData.append('prompt', 'Add a bass line, harmony, and drums with realistic expression');
formData.append('use_fine_stage', true); // Enable fine stage

// Make the request
const response = await fetch('http://localhost:8000/generate', {
  method: 'POST',
  body: formData,
});
```

## Expected Performance Impact

Adding the Fine Stage will significantly increase processing time and memory