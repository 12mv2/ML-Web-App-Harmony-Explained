# ML Web App Harmony Explained

## About

ML Web App Harmony Explained is a thoroughly documented ML-powered application that generates complementary musical accompaniment for your original melodies. This project serves as both a functional application and an educational resource showing how to integrate machine learning models into web applications.

The application lets you sing or upload a melody, and a machine learning model creates a harmonizing instrumental line that complements your vocal melody. You can save the original and generated melodies in the database.

This project emphasizes clear documentation, robust error handling, and detailed explanations of ML integration techniques.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [ML Service](#ml-service)
  - [How It Works](#how-it-works)
  - [Parameter Guidelines](#parameter-guidelines)
  - [Fine Stage Enhancement](#fine-stage-enhancement) 
  - [Processing Times & Timeouts](#processing-times--timeouts)
- [Features](#features)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
- [Current Limitations](#current-limitations)
- [Next Steps & Stretch Features](#next-steps--stretch-features)
- [Lessons Learned & Developer Guide](#lessons-learned--developer-guide)
- [Contributors](#contributors)

## Architecture Overview

HarmonyMaker consists of three main components:

1. **Frontend**: React application for user interaction and audio recording
2. **Backend API**: Node.js/Express server that handles file uploads and communication with the ML service
3. **ML Service**: Python/FastAPI service that processes audio files using machine learning models

## ML Service

The machine learning component of HarmonyMaker is based on [Open-MusicLM](https://github.com/zhvng/open-musiclm), an open-source implementation of Google's MusicLM model. Open-MusicLM uses CLAP (Contrastive Language-Audio Pretraining) for audio processing and Encodec for audio encoding/decoding.

### How It Works

1. **An audio file is uploaded**—a **piano playing a jazz riff in A minor.**
   - The **HuBERT-KMeans model analyzes the audio** and **creates semantic tokens.**
   - These tokens represent **musical features** like **melody, rhythm, and timbre**, but **they don't store actual sound—only structured musical meaning.**

2. **A text prompt is processed by the CLAP model, which converts it into numerical embeddings (vectors).**
   - These vectors represent the **musical meaning of the text in a format that can be matched with the semantic tokens from HuBERT-KMeans.**
   - Example prompt:  
     **"Add a bass line, harmony, and drums, and remove the piano melody to leave space for the soloist."**

3. **A Transformer model compares and aligns the audio tokens with the text embeddings.**
   - Instead of just comparing, the **Transformer modifies the semantic tokens based on the meaning of the text embeddings.**
   - **The result is a new set of refined tokens that preserve the essence of the original piano melody while incorporating harmony and rhythm from the text prompt.**

4. **The Coarse Stage transforms the refined tokens into coarse audio tokens.**
   - This step **converts the abstract musical meaning from the Transformer into structured acoustic representations.**
   - **These coarse tokens define**:  
     - 🎵 **Melody** (or its removal, as requested).  
     - 🥁 **Rhythm and groove**.  
     - 🎸 **Instrument structure of the backing track**.  
   - **At this stage, the music lacks:**  
     - 🎭 **Dynamics (loud vs. soft notes).**  
     - 🎼 **Articulation (staccato, legato, accents).**  
     - 🎻 **Realistic sound texture.**  

5. **The Fine Stage refines the coarse tokens into fine tokens.**
   - **This stage adds finer details such as:**  
     - 🎻 **Timbre (instrument texture and quality).**  
     - 🎶 **Harmonics (richness and overtones).**  
     - 🎚️ **Nuances (attack, decay, expressiveness).**  

6. **Encodec converts the coarse or fine tokens into real audio.**
   - If **only Coarse Tokens are used** → 🎵 **Lower-resolution output, faster generation**.  
   - If **Fine Tokens are used** → 🎶 **Higher-quality, more expressive music**.  

### Parameter Guidelines

The parameters used in the ML service are defined in `ml_service/api/main.py` within the `@app.post("/generate")` function. These parameters directly control different aspects of the music generation process:

```python
@app.post("/generate")
async def generate_music(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...),
    semantic_steps: int = Form(5)
    duration: Optional[int] = Form(1),  
    time_steps_factor: Optional[int] = Form(4),
    temperature: Optional[float] = Form(0.85),
    prompt: Optional[str] = Form("Add a bass line, harmony, and drums, and remove the piano melody to leave space for the soloist."),
    use_fine_stage: Optional[bool] = Form(False),
    save_for_eval: Optional[bool] = Form(False) 
):
```

Note: if no prompt parameter is given, then "Add a bass line, harmony, and drums, and remove the piano melody to leave space for the soloist." will be used.

Each of these parameters affects different stages of the ML pipeline, as explained below.

| Stage                      | Process                                         | Example in Our Jazz Harmony Task                               | Key Parameters Used                                           |
|----------------------------|-------------------------------------------------|------------------------------------------------|------------------------------------------------|
| **1. HuBERT-KMeans**       | Extracts semantic tokens from input audio      | Piano melody is converted into structured tokens | `semantic_steps` (affects detail of extracted tokens) |
| **2. CLAP**                | Converts text prompt into embeddings            | Text prompt (e.g., 'add a bass line, harmony, and drums, and remove the piano melody to leave space for the soloist.') converted into numerical representation | N/A (CLAP operates without direct tunable parameters) |
| **3. Transformer**         | Refines & modifies tokens to align text & audio | Adjusts the piano's tokens to match the requested jazz accompaniment | `temperature` (controls creativity vs. structure), `time_steps_factor` (affects how text modifies the audio tokens) |
| **4. Coarse Stage**        | Converts tokens into coarse audio representations | Generates rough versions of bass and drums, defining structure and instrument separation | `duration` (controls music length), `time_steps_factor` (affects rhythmic and melodic structure) |
| **5. Fine Stage** | Enhances tokens into higher-resolution sound   | Adds articulation, texture, and nuance to bass & drums for realistic sound | `use_fine_stage` (enables/disables this stage), `time_steps_factor` (adds expressive details like articulation and timbre) |
| **6. Encodec**             | Converts final tokens into real audio           | A full backing track is generated, ready for the pianist to solo over 🎵 | `save_for_eval` (determines if output is stored for analysis) |

### Fine Stage Enhancement

The Fine Stage is an optional enhancement that significantly improves the quality of generated music. When enabled, it adds:

- More realistic instrument timbre and texture
- Better articulation (staccato, legato, etc.)
- Natural dynamics and expressiveness
- Richer harmonics and overtones

To enable the Fine Stage, set the `use_fine_stage` parameter to `true` when making API requests:

```javascript
// Example JavaScript/FormData usage
const formData = new FormData();
formData.append('audio_file', audioBlob);
formData.append('semantic_steps', '5');
formData.append('duration', '10');
formData.append('use_fine_stage', 'true');  // Enable fine stage
```

**Important considerations:**

- Fine Stage generation takes approximately 2-3x longer than Coarse-only generation
- Memory requirements are higher (4GB+ RAM recommended)
- The system automatically falls back to Coarse-only generation if Fine Stage fails
- For high-quality music with detailed expression, enable the Fine Stage
- For quick previews or when performance is more important than quality, keep it disabled

### Processing Times & Timeouts

The ML service can take a significant amount of time to process audio, especially with higher parameter values and when the Fine Stage is enabled:
- Small coarse-only generations (duration: 1-5): ~30-90 seconds
- Medium coarse-only generations (duration: 5-20): ~2-5 minutes
- Small fine stage generations (duration: 1-5): ~1-3 minutes
- Medium fine stage generations (duration: 5-20): ~5-15 minutes

To accommodate these long processing times, the application implements:
1. Extended `headersTimeout` (20 minutes) using the `undici` library in the Node.js backend
2. Custom `fetchWithTimeout` wrapper with detailed error handling and logging
3. System resource monitoring in the ML service

## Features

- **Melody Generation**: Record your vocal melody and get an AI-generated accompaniment
- **Audio Recording**: Record directly in the browser or upload existing audio files
- **Audio Playback**: Listen to both original and generated audio with built-in player
- **Save Audios**: Save your original melody and generated audio for future reference (requires database setup)
- **High-Quality Mode**: Enable Fine Stage for more realistic, expressive audio (optional)

## Quick Start

### Prerequisites

- Node.js (v18+)
- Python 3.11
- npm or yarn
- 16GB RAM recommended (the ML service requires significant memory, especially with Fine Stage enabled)

### Installation

1. Install web application dependencies:

```bash
npm install
```

2. Install ML service dependencies (in a separate terminal):

```bash
cd ml_service
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Download model weights:
   - For basic functionality, you need:
     - clap.rvq.950_no_fusion.pt
     - kmeans_10s_no_fusion.joblib
     - real_semcoarsetosem.transformer.5170.pt
     - coarse.transformer.18000.pt
   - For enhanced quality (Fine Stage):
     - fine.transformer.24000.pt

   Place all weights in the `ml_service/model_weights/` directory.

### Running the Application

Start all services (frontend, backend, and tailwind):

```bash
npm run dev
```

Start ML service (in a separate terminal):

```bash
cd ml_service
source venv/bin/activate
uvicorn api.main:app --reload --port 8000
```

The application will be available at:

- **Frontend**: [http://localhost:3033](http://localhost:3033)
- **Backend API**: [http://localhost:4040](http://localhost:4040)
- **ML Service**: [http://localhost:8000](http://localhost:8000)
- **Health Check**: [http://localhost:4040/check-ml-service](http://localhost:4040/check-ml-service)

For development, use username: `123` and password: `123` to login. **Note:** Database connection is only required for saving audio pairs - you can still use the audio transformation features without database access.

## Contributors
- **Colin Rooney**