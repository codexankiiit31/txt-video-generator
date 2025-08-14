### AI Video Generator
Transform your text prompts into stunning videos using Stability AI Ultra! This application provides both a beautiful Streamlit web interface and a robust FastAPI backend for generating videos from text descriptions.
✨ Features

🎨 Text-to-Video Generation: Create videos from detailed text prompts using Stability AI Ultra
🖥️ Dual Interface: Choose between Streamlit web app or FastAPI REST API
🎛️ Customizable Settings: Control duration, resolution, style, and more
📊 Generation History: Track all your creations with metadata
🔍 Real-time Progress: Monitor generation progress with live updates
📁 File Management: Download, list, and manage generated videos
🛡️ Content Safety: Built-in prompt validation and content moderation
⚡ Enhanced Effects: Dynamic zoom, rotation, and color effects for engaging videos

🏗️ Project Structure
ai-video-generator/
├── frontend/
│   └── app.py              # Streamlit web application
├── backend/
│   ├── main.py             # FastAPI server
│   └── chains.py           # Core video generation logic
├── outputs/                # Generated video files (auto-created)
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables
└── README.md              # This file

└── README.md              # This file
🚀 Quick Start
1. Prerequisites

Python 3.8 or higher
Stability AI API key (Get one here)
Git (for cloning)

2. Installation
# Clone the repository
git clone <your-repo-url>
cd ai-video-generator

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

3. Dependencies
Create a requirements.txt file with the following dependencies:
streamlit>=1.28.0
fastapi>=0.104.0
uvicorn>=0.24.0
aiohttp>=3.8.0
aiofiles>=23.0.0
python-dotenv>=1.0.0
pydantic>=2.4.0
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0
requests>=2.31.0
python-multipart>=0.0.6

4. Environment Configuration
Create a .env file in the root directory:
# Required: Stability AI API Key
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

# Optional: Application Metadata
APP_TITLE=AI Video Generation API
APP_DESCRIPTION=Generate videos from text prompts using Stability AI
APP_VERSION=1.0.0

# Optional: Security Settings
MAX_PROMPT_LENGTH=2000
FORBIDDEN_WORDS=violence,explicit,harmful
CORS_ORIGINS=*
CORS_CREDENTIALS=true


API will be available at http://localhost:8000

Interactive API docs: http://localhost:8000/docs
Alternative docs: http://localhost:8000/redoc

🚨 Troubleshooting
Common Issues
❌ "API key not found"

Ensure your .env file contains STABILITY_API_KEY=your_key_here
Check that the .env file is in the root directory
Verify the API key is valid on Stability AI platform

❌ "Video generation failed"

Check your internet connection
Verify API key has sufficient credits
Ensure prompt doesn't violate content policy

❌ "Module not found: chains"

Ensure chains.py is in the backend directory
Check Python path configuration
Try running from the correct directory

❌ "Connection timeout"

Check firewall settings
Verify Stability AI API is accessible
Try with a shorter prompt

🆘 Support

Issues: Report bugs and feature requests on GitHub
Documentation: Check the /docs endpoint when running the API
Stability AI: Visit Stability AI Documentation for API details

🙏 Acknowledgments

Stability AI for providing the Ultra image generation API
Streamlit for the beautiful web interface framework
FastAPI for the high-performance API backend
OpenCV for video processing capabilities
