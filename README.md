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
   - These tokens represent **musical features** like **melody, rhythm, and timbre**, but **they don’t store actual sound—only structured musical meaning.**

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

5. **The Fine Stage (if used) refines the coarse tokens into fine tokens.**
   - **This stage adds finer details such as:**  
     - 🎻 **Timbre (instrument texture and quality).**  
     - 🎶 **Harmonics (richness and overtones).**  
     - 🎚️ **Nuances (attack, decay, expressiveness).**  

6. **Encodec converts the coarse or fine tokens into real audio.**
   - If **only Coarse Tokens are used** → 🎵 **Lower-resolution output, faster generation**.  
   - If **Fine Tokens are used** → 🎶 **Higher-quality, more expressive music**.  

## Parameter Guidelines

The parameters used in the ML service are defined in `ml_service/api/main.py` within the `@app.post("/generate")` function. These parameters directly control different aspects of the music generation process:

```python
@app.post("/generate")
async def generate_music(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...),
    semantic_steps: int = Form(6)
    duration: Optional[int] = Form(40),  
    time_steps_factor: Optional[int] = Form(6),
    temperature: Optional[float] = Form(0.85),
    prompt: Optional[str] = Form("Diverse kinds of instrument and richness"),
    save_for_eval: Optional[bool] = Form(False) 
):
```

Each of these parameters affects different stages of the ML pipeline, as explained below.

| Stage                      | Process                                         | Example in Our Jazz Harmony Task                               | Key Parameters Used                                           |
|----------------------------|-------------------------------------------------|------------------------------------------------|------------------------------------------------|
| **1. HuBERT-KMeans**       | Extracts semantic tokens from input audio      | Piano melody is converted into structured tokens | `semantic_steps` (affects detail of extracted tokens) |
| **2. CLAP**                | Converts text prompt into embeddings            | Text prompt (e.g., 'add a bass line, harmony, and drums, and remove the piano melody to leave space for the soloist.') converted into numerical representation | N/A (CLAP operates without direct tunable parameters) |
| **3. Transformer**         | Refines & modifies tokens to align text & audio | Adjusts the piano’s tokens to match the requested jazz accompaniment | `temperature` (controls creativity vs. structure), `time_steps_factor` (affects how text modifies the audio tokens) |
| **4. Coarse Stage**        | Converts tokens into coarse audio representations | Generates rough versions of bass and drums, defining structure and instrument separation | `duration` (controls music length), `time_steps_factor` (affects rhythmic and melodic structure) |
| **5. Fine Stage (Optional)** | Enhances tokens into higher-resolution sound   | Adds articulation, texture, and nuance to bass & drums for realistic sound | `time_steps_factor` (adds expressive details like articulation and timbre) |
| **6. Encodec**             | Converts final tokens into real audio           | A full backing track is generated, ready for the pianist to solo over 🎵 | `save_for_eval` (determines if output is stored for analysis) |

## Contributors
- **Colin Rooney**
- **Patrick Cromer**

