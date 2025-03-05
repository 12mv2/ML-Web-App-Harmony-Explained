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
- [Accessing Feature Branches](#accessing-feature-branches)
- [Known Issues & Future Work](#known-issues--future-work)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)

## Architecture Overview

HarmonyMaker consists of three main components:

1. **Frontend**: React application for user interaction and audio recording
2. **Backend API**: Node.js/Express server that handles file uploads and communication with the ML service
3. **ML Service**: Python/FastAPI service that processes audio files using machine learning models

## ML Service

The machine learning component of HarmonyMaker is based on [Open-MusicLM](https://github.com/zhvng/open-musiclm), an open-source implementation of Google's MusicLM model. Open-MusicLM uses CLAP (Contrastive Language-Audio Pretraining) for audio processing and Encodec for audio encoding/decoding.

### How It Works

1. The user records or uploads an audio melody
2. The audio is sent to the ML service via the backend API
3. The ML service processes the audio using multiple stages:
   - Semantic token generation (understanding the musical content)
   - Coarse token generation (creating complementary accompaniment)
   - Audio reconstruction (creating the final output)
4. The generated audio is returned to the frontend for playback

### Parameter Guidelines

The ML service accepts several parameters that control the generation process:

| Parameter | Description | Recommended Values | Impact on Processing |
|-----------|-------------|-------------------|----------------------|
| `duration` | Output audio length in seconds | 10-40? | Higher values increase processing time and memory usage (not strictly linear) |
| `semantic_steps` | Quality of semantic understanding | 4-8 | Higher values may improve quality but increase processing time |
| `time_steps_factor` | Time resolution multiplier | 4-6 | Controls "granularity" of audio generation (like frames per second) |
| `temperature` | Creative variation | 0.7-1.0 | Higher values produce more varied outputs |

**Understanding time_steps_factor:**
The `time_steps_factor` works as a multiplier with `duration` to determine how many discrete time steps the model will generate. It's somewhat analogous to the "frames per second" in video - higher values create more fine-grained generation with potentially smoother transitions. The actual calculation used is `max_time_steps = duration * time_steps_factor`. Increasing this value requires more memory and processing time but might result in more detailed audio output.

**Important Notes:**
- Setting `duration` above 40 may cause generation to fail due to memory constraints
- Total time steps (`duration * time_steps_factor`) over 240 may cause memory issues
- For faster iteration during development, use lower values (duration: 1-20, semantic_steps: 4-6)

### Processing Times & Timeouts

The ML service can take a significant amount of time to process audio, especially with higher parameter values:
- Small generations (duration: 10-20 / ~ 1 second audio output): ~3-8 minutes
- Medium generations (duration: 20-30): ~8-12 minutes
- Large generations (duration: 30-40): ~12-16 minutes
- Jumbo generations (duration: 30-40 / 7 min audio output): ~12-16 minutes

To accommodate these long processing times, the application implements:
1. Extended `headersTimeout` (20 minutes) using the `undici` library in the Node.js backend
2. Custom `fetchWithTimeout` wrapper with detailed error handling and logging
3. System resource monitoring in the ML service

## Features

- **Melody Generation**: Record your vocal melody and get an AI-generated accompaniment (Note: still in MVP stage, generation quality varies)
- **Audio Recording**: Record directly in the browser or upload existing audio files
- **Audio Playback**: Listen to both original and generated audio with built-in player
- **Save Audios**: Save your original melody and generated audio for future reference (requires database setup)

## Quick Start

### Prerequisites

- Node.js (v18+)
- Python 3.11
- npm or yarn
- 16GB RAM recommended (the ML service requires significant memory)

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

## Repository Information

This repository (ML Web App Harmony Explained) is a comprehensive, documented version of the HarmonyMaker project with enhanced explanations and error handling. It focuses on providing a clear understanding of how to integrate ML models into web applications, particularly for audio processing.

### Original Project

This project is based on the original HarmonyMaker project. The core functionality has been maintained while adding:

- Improved error handling
- Extended timeouts for ML processing
- Better documentation and explanations
- Enhanced parameter validation

### Getting Started

To use this project:

```bash
# Clone the repository
git clone https://github.com/[your-username]/ML-Web-App-Harmony-Explained.git

# Navigate to the project directory
cd ML-Web-App-Harmony-Explained
```

### Database Configuration

You don't need access to a database to use the ML service - the core audio transformation features work without it. However, database access is required for:

1. **Authentication**: To log in and access the full application
2. **Saving audio**: To store your original and transformed audio files for later retrieval

To get access to the Supabase API keys needed for database functionality:
1. Contact https://github.com/12mv2 and request the credentials
2. Place the provided keys in a `.env` file in the root directory

For developers who just want to experiment with the ML transformation functionality, you can skip this step and use the default development credentials (username: `123`, password: `123`).

## Known Issues & Future Work

- **Long Processing Times**: The ML service takes several minutes to generate audio, which can lead to a suboptimal user experience
- **Memory Constraints**: Setting duration above 40 seconds may cause generation to fail due to memory limitations
- **Network Timeouts**: Although we've implemented extended timeouts, some browsers might timeout client-side during long generations
- **Generation Quality**: The model sometimes produces unexpected or low-quality outputs

Future improvements:
- Implement a job queue + polling system for more reliable long-running operations
- Add progress tracking API for real-time updates during generation
- Optimize the ML model for faster processing and better quality
- Add more detailed error handling and recovery mechanisms

## Troubleshooting

### Common Issues

**"fetch failed" or timeout errors:**
- Verify ML service is running (`uvicorn api.main:app --reload --port 8000`)
- Check system memory availability (ML service needs ~14GB+ for longer generations)
- Try with shorter duration values
- Check the health of the ML service at [http://localhost:4040/check-ml-service](http://localhost:4040/check-ml-service)

**ML service returns empty or low-quality audio:**
- Check ML service logs for errors
- Verify audio input format (WAV format works best)
- Try with different parameter combinations
- Ensure your input audio is clear and well-recorded

**Server memory issues:**
- Restart the ML service to clear memory
- Close unnecessary applications
- Try with smaller parameter values

## Roadmap

- **Cloud-based GPU Service**: Move ML processing to cloud GPUs for faster generation times and reduced local resource requirements
- **ML Model Improvements**: Enhance generation quality and processing speed
- **Frontend UI Improvements**: Create more intuitive interface with better feedback during processing
- **Job Queue System**: Implement a more robust system for handling long-running tasks
- **OAuth Integration**: Enable Google OAuth for easier signup/login