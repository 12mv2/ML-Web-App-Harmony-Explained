# 🗺️ Harmony Explained: Development Roadmap

This document outlines the development plan for Harmony Explained, an ML-powered application that generates complementary musical accompaniment for original melodies.

## 🎯 MVP Goals

### 🔒 Completed Features

- ✅ Upload and record audio from frontend
- ✅ Process audio using HuBERT-KMeans to extract semantic tokens
- ✅ Convert prompt to embeddings with CLAP
- ✅ Transformer maps text + audio into refined semantic tokens
- ✅ Generate audio with Coarse stage
- ✅ Return generated `.wav` file to frontend
- ✅ Long fetch requests functioning (up to 14 hours)
- ✅ Basic parameter tuning

### 🚀 Current Development Focus

1. **Model Enhancements** - Improve audio quality by upgrading model architecture
2. **Parameter Optimization** - Find the optimal balance between quality and generation time
3. **GPU Acceleration** - Deploy ML service to cloud GPU for faster generation
4. **Documentation** - Create comprehensive examples and explanations

## 📋 Prioritized Task List

See our [GitHub Issues](https://github.com/yourusername/harmony-explained/issues) for the current prioritized tasks. Key issues include:

- #5: Upgrade Model Configuration for Higher Quality Audio
- #6: Optimize Fine Stage Parameters
- #7: Set Balanced Generation Parameters
- #8: Create Docker Container for ML Service
- #9: Deploy ML Service to Cloud with GPU
- #10: Document and Compare Audio Quality Improvements

## 🔮 Stretch Goals

After achieving our MVP, we plan to enhance the project with:

- Improved UI with model size selection
- Parameter sliders and visual feedback
- Progress indicators during generation
- Save/load functionality for previous generations
- User authentication and sharing features

## 📊 Performance Targets

- Generation Time: <0.5 hour for 7 seconds of high-quality audio on GPU
- Audio Quality: Clear harmonies, tolerable tibre, proper musical structure