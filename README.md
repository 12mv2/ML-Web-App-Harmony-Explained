# HarmonyMaker

## About

HarmonyMaker is a ML-powered application that generates complementary musical accompaniment for your original melodies. Simply sing your tune, and a machine learning model will create a harmonizing instrumental line that perfectly complements your vocal melody. You can save the original and generated melodies in database. 

## Features 

- **Instant Melody Generation**: Record your vocal melody and get an AI-generated accompaniment in seconds
- **Save Audios**: Save your original melody and generated audio for future reference

## Quick Start

### Prerequisites
- Node.js (v18+)
- Python 3.11
- npm or yarn

### Installation

1. Install web application dependencies:

npm install

2. Install ML Service dependencies in another termimal:

cd ml_service
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

## Running the Application

1. Start all services (frontend, backend, tailwind):

npm run dev

2. Start ML service (in a separate terminal):

cd ml_service
source venv/bin/activate
uvicorn api.main:app --reload --port 8000

The application will be available at:

Frontend: http://localhost:3033
Backend API: http://localhost:4040
ML Service: http://localhost:8000

For development, use username: '123' and password: '123' to login. Note: Database connection is only required for saving audio pairs - you can still use the audio transformation features without database access.

## Connecting to the db

Roadmap / Overview of Harmony Maker:

Enhanced ML Model: Improve generation accuracy and coherence by training on larger datasets and refining model architecture
Frontend UI Improvements: Create more intuitive interface with built-in audio playback and download capabilities
OAuth Integration: Enable Google OAuth for easier signup/login

## ML Service
The machine learning component of HarmonyMaker is based on [Open-MusicLM](https://github.com/zhvng/open-musiclm), an open-source implementation of Google's MusicLM model. Open-MusicLM uses CLAP (Contrastive Language-Audio Pretraining) for audio processing and Encodec for audio encoding/decoding. Our application wraps this model in a FastAPI service to provide audio transformation capabilities through a web interface.

The ML service and its pre-trained weights can process audio input and generate complementary musical accompaniment. For detailed information about the ML service and its setup, see the README in the `ml_service` directory.

## Resources to help you understand Harmony Maker

Andrej Karpathy makes excellent resources:

His deep dive into LLM's:
https://www.youtube.com/watch?v=7xTGNNLPyMI&t=6350s&ab_channel=AndrejKarpathy

