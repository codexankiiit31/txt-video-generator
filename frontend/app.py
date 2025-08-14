import streamlit as st
import asyncio
import os
import time
import json
from datetime import datetime
from typing import Optional
import base64
from pathlib import Path
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Fix the import - use the correct module name
try:
    from backend.chains import VideoGenerator, GenerationRequest, GenerationResponse
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False
    st.error("❌ Backend module 'chains.py' not found. Please ensure it's in the same directory.")

st.set_page_config(
    page_title="AI Video Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .status-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .error-card {
        background: #fff5f5;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    
    .info-card {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
        border: none;
    }
    
    .video-gallery {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        gap: 1rem;
        margin-top: 2rem;
    }
    
    .video-item {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'generator' not in st.session_state:
    st.session_state.generator = None
    st.session_state.generation_history = []
    st.session_state.current_generation = None
    st.session_state.api_tested = False

def initialize_generator():
    """Initialize the video generator"""
    if not BACKEND_AVAILABLE:
        st.error("❌ Backend module not available. Please check that 'chains.py' is present.")
        return None
    
    try:
        api_key = st.session_state.get('api_key', '')
        if not api_key:
            api_key = os.getenv('STABILITY_API_KEY')
        
        if not api_key:
            st.error("❌ No API key provided. Please enter your Stability AI API key.")
            return None
            
        # Set output directory
        output_dir = os.getenv("OUTPUT_DIR", "./outputs")
        generator = VideoGenerator(api_key=api_key, output_dir=output_dir)
        st.session_state.generator = generator
        return generator
        
    except Exception as e:
        st.error(f"❌ Failed to initialize generator: {str(e)}")
        return None

async def test_api_connection(generator):
    """Test API connection"""
    try:
        result = await generator.test_api_connection()
        return result
    except Exception as e:
        return {"status": "error", "message": f"Connection test failed: {str(e)}"}

def run_async(coro):
    """Helper to run async functions in Streamlit"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)

def main():
    if not BACKEND_AVAILABLE:
        st.stop()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎬 AI Video Generator</h1>
        <p>Transform your text prompts into stunning videos using Stability AI Ultra</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API Key input
        api_key = st.text_input(
            "Stability AI API Key",
            type="password",
            value=st.session_state.get('api_key', ''),
            help="Enter your Stability AI API key. You can also set the STABILITY_API_KEY environment variable."
        )
        
        if api_key:
            st.session_state.api_key = api_key
        
        # Initialize generator if API key is provided
        if api_key and not st.session_state.generator:
            st.session_state.generator = initialize_generator()
        
        # Test API connection
        if st.button("🔍 Test API Connection"):
            if not st.session_state.generator:
                generator = initialize_generator()
            else:
                generator = st.session_state.generator
                
            if generator:
                with st.spinner("Testing API connection..."):
                    result = run_async(test_api_connection(generator))
                    
                if result["status"] == "success":
                    st.success("✅ API connection successful!")
                    st.session_state.api_tested = True
                    with st.expander("API Response"):
                        st.json(result)
                else:
                    st.error(f"❌ API connection failed: {result['message']}")
                    st.session_state.api_tested = False
            else:
                st.error("❌ Please configure your API key first!")
        
        # Show API status
        if st.session_state.generator:
            st.success("✅ Generator initialized")
        else:
            st.warning("⚠️ Generator not initialized")
        
        # Generation settings
        st.header("🎥 Generation Settings")
        
        duration = st.slider(
            "Video Duration (seconds)",
            min_value=3,
            max_value=30,
            value=8,
            help="Length of the generated video"
        )
        
        resolution = st.selectbox(
            "Resolution",
            options=["512x512", "1024x1024", "1024x576", "576x1024"],
            index=1,
            help="Output video resolution"
        )
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            negative_prompt = st.text_area(
                "Negative Prompt (Optional)",
                placeholder="blurry, low quality, distorted, watermark...",
                help="Describe what you don't want to see in the image"
            )
            
            use_seed = st.checkbox("Use Custom Seed")
            seed = None
            if use_seed:
                seed = st.number_input(
                    "Seed",
                    min_value=0,
                    max_value=4294967294,
                    value=42,
                    help="Seed for reproducible results"
                )
        
        # Statistics
        st.header("📊 Statistics")
        total_generations = len(st.session_state.generation_history)
        successful_generations = len([g for g in st.session_state.generation_history if g.get('success', False)])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total", total_generations)
        with col2:
            st.metric("Success", successful_generations)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Create Your Video")
        
        # Main prompt input
        prompt = st.text_area(
            "Enter your video prompt",
            height=120,
            placeholder="A majestic mountain landscape with snow-capped peaks, golden hour lighting, cinematic composition, photorealistic, 8k quality...",
            help="Describe the image you want to generate. Be detailed and specific for better results."
        )
        
        # Style presets
        st.subheader("🎨 Style Presets")
        style_presets = {
            "Custom": "",
            "Cinematic": "cinematic lighting, film grain, dramatic composition, professional cinematography",
            "Artistic": "artistic, painterly, creative composition, vibrant colors, artistic lighting",
            "Photorealistic": "photorealistic, high detail, sharp focus, professional photography, 8k resolution",
            "Fantasy": "fantasy art, magical, ethereal, mystical atmosphere, enchanted",
            "Sci-Fi": "futuristic, sci-fi, cyberpunk, high-tech, neon lighting, digital art",
            "Nature": "natural lighting, organic, beautiful landscape, serene, peaceful"
        }
        
        selected_style = st.selectbox("Choose a style preset", list(style_presets.keys()))
        
        # Combine prompt with style
        final_prompt = prompt
        if selected_style != "Custom" and style_presets[selected_style]:
            final_prompt = f"{prompt}, {style_presets[selected_style]}" if prompt else style_presets[selected_style]
        
        if final_prompt != prompt:
            st.info(f"**Final prompt:** {final_prompt}")
        
        # Generate button
        generate_col1, generate_col2 = st.columns([3, 1])
        
        with generate_col1:
            # Check if generator is available and prompt is valid
            can_generate = (
                st.session_state.generator is not None and 
                final_prompt and 
                len(final_prompt.strip()) >= 10
            )
            
            generate_button = st.button(
                "🚀 Generate Video",
                disabled=not can_generate,
                use_container_width=True
            )
            
            if not st.session_state.generator and final_prompt:
                st.warning("⚠️ Please configure your API key first!")
            elif final_prompt and len(final_prompt.strip()) < 10:
                st.warning("⚠️ Prompt must be at least 10 characters long!")
        
        with generate_col2:
            if st.button("🔄 Clear", use_container_width=True):
                st.rerun()
    
    with col2:
        st.header("💡 Tips")
        
        st.markdown("""
        **For better results:**
        
        ✅ **Be specific** - Include details about lighting, style, composition
        
        ✅ **Use quality terms** - Add "8k", "high quality", "professional"
        
        ✅ **Describe the mood** - Include atmosphere and emotional tone
        
        ✅ **Mention camera angles** - "wide shot", "close-up", "aerial view"
        
        ❌ **Avoid contradictions** - Don't use conflicting descriptions
        
        ❌ **Keep it focused** - Don't include too many different elements
        """)
        
        st.markdown("""
        **Example prompts:**
        
        📸 *"A serene lake at sunset, golden hour lighting, mountains in background, photorealistic, 8k quality"*
        
        🎨 *"Abstract digital art, flowing colors, neon lights, futuristic, cyberpunk style"*
        
        🏰 *"Medieval castle on a hill, dramatic storm clouds, cinematic lighting, epic fantasy"*
        """)
    
    # Generation process
    if generate_button:
        if not st.session_state.generator:
            st.error("❌ Please configure your API key first!")
            return
        
        generator = st.session_state.generator
        
        # Create generation request
        try:
            request = GenerationRequest(
                prompt=final_prompt,
                duration=duration,
                resolution=resolution,
                negative_prompt=negative_prompt if negative_prompt else None,
                seed=seed
            )
            
            # Start generation
            st.markdown("---")
            st.header("🎬 Generating Your Video...")
            
            progress_container = st.container()
            status_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            # Simulate progress updates
            with status_container:
                with st.spinner("Initializing generation..."):
                    status_text.text("🔄 Preparing request...")
                    progress_bar.progress(10)
                    time.sleep(0.5)
                    
                    status_text.text("🎨 Generating image with Stability AI Ultra...")
                    progress_bar.progress(30)
                    
                    # Run the actual generation
                    try:
                        result = run_async(generator.generate_video(request))
                        
                        progress_bar.progress(80)
                        status_text.text("🎞️ Creating video from image...")
                        time.sleep(1)
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Generation complete!")
                        
                        # Store in history
                        generation_data = {
                            'timestamp': datetime.now().isoformat(),
                            'prompt': final_prompt,
                            'success': result.success,
                            'filename': result.filename,
                            'duration': result.generation_time,
                            'metadata': result.metadata
                        }
                        st.session_state.generation_history.append(generation_data)
                        
                        # Display result
                        if result.success:
                            st.success(f"🎉 Video generated successfully in {result.generation_time:.2f} seconds!")
                            
                            # Display video/image
                            if result.filename:
                                file_path = generator.get_output_path(result.filename)
                                if os.path.exists(file_path):
                                    # Check file extension
                                    if result.filename.lower().endswith(('.mp4', '.avi', '.mov')):
                                        with open(file_path, "rb") as f:
                                           video_bytes = f.read()
                                        if video_bytes:
                                         st.video(video_bytes)
                                    elif result.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                                        st.image(file_path, caption="Generated Image")
                                    
                                    # Download button
                                    with open(file_path, "rb") as file:
                                        file_data = file.read()
                                        mime_type = "video/mp4" if result.filename.lower().endswith('.mp4') else "image/png"
                                        st.download_button(
                                            label=f"📥 Download {result.filename}",
                                            data=file_data,
                                            file_name=result.filename,
                                            mime=mime_type,
                                            use_container_width=True
                                        )
                                else:
                                    st.warning("Generated file not found, but generation was reported as successful.")
                            
                            # Display metadata
                            with st.expander("📊 Generation Details"):
                                st.json(result.metadata)
                                
                        else:
                            st.error(f"❌ Generation failed: {result.message}")
                            
                    except Exception as e:
                        progress_bar.progress(100)
                        status_text.text("❌ Generation failed!")
                        st.error(f"Generation error: {str(e)}")
                        
                        # Store failed generation
                        generation_data = {
                            'timestamp': datetime.now().isoformat(),
                            'prompt': final_prompt,
                            'success': False,
                            'error': str(e)
                        }
                        st.session_state.generation_history.append(generation_data)
        
        except Exception as e:
            st.error(f"❌ Error creating request: {str(e)}")
    
    # Generation history
    if st.session_state.generation_history:
        st.markdown("---")
        st.header("📚 Generation History")
        
        # Filter and sort options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_successful = st.checkbox("✅ Show Successful", value=True)
        with col2:
            show_failed = st.checkbox("❌ Show Failed", value=True)
        with col3:
            limit_results = st.number_input("Show last N results", min_value=1, max_value=50, value=10)
        
        # Filter history
        filtered_history = []
        for item in reversed(st.session_state.generation_history[-limit_results:]):
            if (show_successful and item.get('success', False)) or (show_failed and not item.get('success', False)):
                filtered_history.append(item)
        
        # Display history
        for i, item in enumerate(filtered_history):
            with st.expander(
                f"{'✅' if item.get('success', False) else '❌'} {item['prompt'][:60]}... - {item['timestamp'][:19]}"
            ):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Prompt:** {item['prompt']}")
                    st.write(f"**Time:** {item['timestamp']}")
                    
                    if item.get('success', False):
                        st.write(f"**Duration:** {item.get('duration', 'N/A'):.2f}s")
                        st.write(f"**Filename:** {item.get('filename', 'N/A')}")
                        
                        # Show video/image if it exists
                        if item.get('filename') and st.session_state.generator:
                            file_path = st.session_state.generator.get_output_path(item['filename'])
                            if os.path.exists(file_path):
                                if item['filename'].lower().endswith(('.mp4', '.avi', '.mov')):
                                    st.video(file_path)
                                elif item['filename'].lower().endswith(('.png', '.jpg', '.jpeg')):
                                    st.image(file_path, width=300)
                    else:
                        st.error(f"**Error:** {item.get('error', 'Unknown error')}")
                
                with col2:
                    if item.get('success', False) and item.get('metadata'):
                        st.json(item['metadata'])

if __name__ == "__main__":
    main()