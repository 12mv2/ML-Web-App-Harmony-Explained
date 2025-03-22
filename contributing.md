# Contributing to Harmony Explained

Thank you for your interest in contributing to Harmony Explained! This document provides guidelines and instructions for contributing to the project.

## 🐛 Issue Tracking

We use GitHub Issues to track tasks, enhancements, and bugs. Each issue is numbered and prioritized:

- Priority (1-2): Critical for MVP
- Priority (3-5): Important for user experience
- Priority (6+): Enhancements and optimizations

## 🔧 Development Setup

### Prerequisites

- Python 3.11+
- Node.js (v18+)
- npm or yarn
- 16GB RAM recommended

### Installation

1. Fork and clone the repository
2. Install dependencies:
   ```bash
   npm install
   cd ml_service
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Download model weights (see README.md)

## 🧪 Testing Changes

1. Make your changes in a feature branch
2. Test with at least one audio sample
3. Document performance impacts (generation time, quality)
4. Submit a pull request referencing the issue number

## 🔍 Code Review Process

Pull requests will be reviewed based on:

1. Code quality and maintainability
2. Performance impact
3. Documentation quality
4. Adherence to the project architecture

## 💻 Cloud GPU Testing

If you have access to a cloud GPU (H100/H200 recommended):

1. Follow the setup instructions in README.md
2. Document generation time and quality metrics
3. Share generated audio samples with your PR

## 📝 Documentation

When contributing, please update relevant documentation:

- Code comments for complex logic
- README updates for user-facing changes
- Parameter documentation for any new settings