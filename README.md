# HarmonyMaker

## About

HarmonyMaker is a ML-powered application that generates complementary musical accompaniment for your original melodies. Simply sing your tune, and a machine learning model will create a harmonizing instrumental line that perfectly complements your vocal melody. You can save the original and generated melodies in database.

## ML Service

The machine learning component of HarmonyMaker is based on [Open-MusicLM](https://github.com/zhvng/open-musiclm), an open-source implementation of Google's MusicLM model. Open-MusicLM uses CLAP (Contrastive Language-Audio Pretraining) for audio processing and Encodec for audio encoding/decoding. Our application wraps this model in a FastAPI service to provide audio transformation capabilities through a web interface.

The ML service and its pre-trained weights can process audio input and generate complementary musical accompaniment. For detailed information about the ML service and its setup, see the README in the `ml_service` directory.

## Features

- **Instant Melody Generation**: Record your vocal melody and get an AI-generated accompaniment in seconds (still in MVP stage, generates a short ugly sound. We (That includes you!) are working to see if the model can generate a usable accompanyment)
- **Save Audios**: Save your original melody and generated audio for future reference (requires database setup. This feature exists but is on the backburner for now)

## Quick Start

### Prerequisites

- Node.js (v18+)
- Python 3.11
- npm or yarn

### Installation

1. Install web application dependencies:

npm install

2. Install ML service dependencies (in a separate terminal):

cd ml_service
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

### Running the Application

Start all services (frontend, backend, and tailwind):

npm run dev

Start ML service (in a separate terminal):

cd ml_service
source venv/bin/activate
uvicorn api.main:app --reload --port 8000

The application will be available at:

- **Frontend**: [http://localhost:3033](http://localhost:3033)
- **Backend API**: [http://localhost:4040](http://localhost:4040)
- **ML Service**: [http://localhost:8000](http://localhost:8000)

For development, use username: `123` and password: `123` to login. **Note:** Database connection is only required for saving audio pairs - you can still use the audio transformation features without database access.

## Accessing Feature Branches

If you want to access a feature branch that has been pushed to the remote repository, use the following commands:

git fetch origin <branch-name>
git checkout <branch-name>

### If You Cloned the Repository

If you cloned the repository (instead of forking), you can directly access all branches in the repository using:

git fetch origin

Then checkout the desired branch:

git checkout <branch-name>

### If You Forked the Repository

If you have forked the repository and the feature branch does not exist in your fork, follow these steps:

1. **Add the original repository as an upstream remote:**

   git remote add upstream <original-repo-url>

2. **Fetch branches from the upstream repository:**

   git fetch upstream

3. **Check out the feature branch from upstream:**

   git checkout -b <branch-name> upstream/<branch-name>


This will allow you to work with the latest version of the feature branch from the original repository.

## Roadmap

- **ML Model Integration**: Integrated an external ML model within a web application for music generation.
- **Frontend UI Improvements**: Create more intuitive interface with built-in audio playback and download capabilities
- **OAuth Integration**: Enable Google OAuth for easier signup/login

