# AI Video Generator 🎥✨

Transform your text prompts into stunning videos using **Stability AI Ultra**!  
This project provides a **beautiful Streamlit web interface** and a **robust FastAPI backend** for generating videos from text descriptions.

---

## ✨ Features

- 🎨 **Text-to-Video Generation:** Convert detailed text prompts into videos using Stability AI Ultra.  
- 🖥️ **Dual Interface:** Use either a Streamlit web app or FastAPI REST API.  
- 🎛️ **Customizable Settings:** Control video duration, resolution, style, and more.  
- 📊 **Generation History:** Keep track of all created videos with metadata.  
- 🔍 **Real-time Progress:** Monitor generation status with live updates.  
- 📁 **File Management:** Download, list, and manage generated videos easily.  
- 🛡️ **Content Safety:** Built-in prompt validation and moderation.  
- ⚡ **Enhanced Effects:** Dynamic zoom, rotation, and color effects for engaging results.  

---

## 🏗️ Project Structure


---

## 🚀 Quick Start

### 1️⃣ Prerequisites

- Python 3.8 or higher  
- Stability AI API key ([Get one here](https://stability.ai))  
- Git  

---

### 2️⃣ Installation

``bash
# Clone repository
git clone <your-repo-url>
cd ai-video-generator
``bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies

** install -r requirements.txt
3️⃣ Environment Configuration

Create a .env file in the root directory:
# Required
STABILITY_API_KEY=your_stability_ai_api_key_here

# Optional: Application Settings
OUTPUT_DIR=./outputs
LOG_LEVEL=INFO
LOG_FILE=video_generation.log

# Optional: API Server Settings
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false
RELOAD=true

# Optional: App Metadata
APP_TITLE=AI Video Generation API
APP_DESCRIPTION=Generate videos from text prompts using Stability AI
APP_VERSION=1.0.0

# Optional: Security Settings
MAX_PROMPT_LENGTH=2000
FORBIDDEN_WORDS=violence,explicit,harmful
CORS_ORIGINS=*
CORS_CREDENTIALS=true

4️⃣ Running the API
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

API endpoint: http://localhost:8000

Interactive docs: http://localhost:8000/docs

Alternative docs: http://localhost:8000/redoc

🚨 Troubleshooting

"API key not found"

Ensure .env contains a valid STABILITY_API_KEY

Check that .env is in the root directory

"Video generation failed"

Verify internet connection and API credits

Ensure the prompt adheres to content policies

"Module not found: chains"

Confirm chains.py exists in the backend/ folder

Run Python from the project root

"Connection timeout"

Check firewall and network settings

Try shorter prompts

🆘 Support

Issues: Report bugs or feature requests on GitHub

Documentation: Check /docs endpoint when running the API

Stability AI: Official Documentation

🙏 Acknowledgments

Stability AI – Ultra video generation API

Streamlit – Web interface framework

FastAPI – High-performance API backend

OpenCV – Video processing capabilities

