import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import re
import base64
import cv2
from dotenv import load_dotenv
from PIL import Image
import io
import numpy as np
import requests
import aiohttp
import aiofiles

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -------------------
# Request/Response Models
# -------------------
class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=10, max_length=2000, description="Text prompt for video generation")
    duration: Optional[int] = Field(default=5, ge=3, le=30, description="Video duration in seconds")
    resolution: Optional[str] = Field(default="1024x1024", description="Video resolution (512x512, 1024x1024)")
    style: Optional[str] = Field(default="realistic", description="Video style preference")
    negative_prompt: Optional[str] = Field(default=None, description="What to avoid in the image")
    seed: Optional[int] = Field(default=None, description="Seed for reproducible results")

class GenerationResponse(BaseModel):
    success: bool
    request_id: str
    message: str
    video_url: Optional[str] = None
    filename: Optional[str] = None
    generation_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

# -------------------
# Video Generator using Stability AI Ultra
# -------------------
class VideoGenerator:
    def __init__(self, api_key: Optional[str] = None, output_dir: Optional[str] = None):
        self.api_key = api_key or os.getenv("STABILITY_API_KEY")
        if not self.api_key:
            raise ValueError("Stability AI API key not found. Set STABILITY_API_KEY environment variable or pass api_key.")

        # Using the correct Stability AI Ultra endpoint
        self.image_endpoint = "https://api.stability.ai/v2beta/stable-image/generate/ultra"
        
        self.output_dir = output_dir or os.getenv("OUTPUT_DIR", "./outputs")
        os.makedirs(self.output_dir, exist_ok=True)
        self.max_prompt_length = int(os.getenv("MAX_PROMPT_LENGTH", "2000"))

        logger.info(f"VideoGenerator initialized. Output directory: {self.output_dir}")

    @classmethod
    def from_environment(cls, output_dir: Optional[str] = None):
        return cls(output_dir=output_dir)

    # -------------------
    # Prompt cleaning
    # -------------------
    def clean_prompt(self, prompt: str) -> str:
        cleaned = re.sub(r'\s+', ' ', prompt.strip())
        cleaned = re.sub(r'[<>{}[\]\\]', '', cleaned)
        if len(cleaned) > self.max_prompt_length:
            cleaned = cleaned[:self.max_prompt_length-3] + "..."
            logger.info(f"Prompt truncated to {self.max_prompt_length} characters")
        return cleaned

    # -------------------
    # Convert resolution to aspect ratio
    # -------------------
    def get_aspect_ratio(self, resolution: str) -> str:
        """Convert resolution to Stability AI aspect ratio format"""
        width, height = map(int, resolution.split("x"))
        
        # Map to supported aspect ratios
        ratio = width / height
        
        if ratio == 1.0:
            return "1:1"
        elif abs(ratio - 16/9) < 0.1:
            return "16:9"
        elif abs(ratio - 9/16) < 0.1:
            return "9:16"
        elif abs(ratio - 21/9) < 0.1:
            return "21:9"
        elif abs(ratio - 9/21) < 0.1:
            return "9:21"
        elif abs(ratio - 2/3) < 0.1:
            return "2:3"
        elif abs(ratio - 3/2) < 0.1:
            return "3:2"
        elif abs(ratio - 4/5) < 0.1:
            return "4:5"
        elif abs(ratio - 5/4) < 0.1:
            return "5:4"
        else:
            # Default to closest standard ratio
            if ratio > 1:
                return "16:9"
            else:
                return "9:16"

    # -------------------
    # Generate image using Stability AI Ultra (sync)
    # -------------------
    def generate_image_sync(self, prompt: str, resolution: str = "1024x1024", 
                          negative_prompt: Optional[str] = None, 
                          seed: Optional[int] = None) -> bytes:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "image/*"
            # Don't set content-type manually - requests handles multipart/form-data
        }
        
        aspect_ratio = self.get_aspect_ratio(resolution)

        # Prepare form data according to API documentation
        files = {'none': ''}  # Empty files dict as shown in documentation
        data = {
            'prompt': prompt,
            'aspect_ratio': aspect_ratio,
            'output_format': 'png'
        }
        
        # Add optional parameters if provided
        if negative_prompt:
            data['negative_prompt'] = negative_prompt
        if seed is not None:
            data['seed'] = seed

        try:
            logger.info(f"Sending request to: {self.image_endpoint}")
            logger.info(f"Prompt: {prompt[:100]}...")
            logger.info(f"Aspect ratio: {aspect_ratio}")
            
            response = requests.post(
                self.image_endpoint,
                headers=headers,
                files=files,
                data=data,
                timeout=120
            )
            
            logger.info(f"Response Status: {response.status_code}")
            
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '')
                logger.info(f"Content-Type: {content_type}")
                
                # With accept: image/*, should receive raw image bytes
                if 'image' in content_type or len(response.content) > 1000:
                    logger.info(f"Received binary image data, size: {len(response.content)} bytes")
                    return response.content
                else:
                    # Fallback: try to parse as JSON in case of base64 response
                    try:
                        json_response = response.json()
                        logger.info("Received JSON response, extracting base64 image")
                        
                        if 'image' in json_response:
                            return base64.b64decode(json_response['image'])
                        else:
                            raise Exception(f"Unexpected JSON response format: {json_response}")
                    except Exception as json_error:
                        logger.error(f"Failed to parse response: {json_error}")
                        raise Exception(f"Unexpected response format")
            
            # Handle specific error codes from API documentation
            elif response.status_code == 400:
                error_msg = f"Invalid parameters: {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
            elif response.status_code == 403:
                error_msg = "Request flagged by content moderation"
                logger.error(error_msg)
                raise Exception(error_msg)
            elif response.status_code == 413:
                error_msg = "Request too large (>10MB)"
                logger.error(error_msg)
                raise Exception(error_msg)
            elif response.status_code == 422:
                error_msg = f"Request rejected: {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
            elif response.status_code == 429:
                error_msg = "Rate limit exceeded (>150 requests in 10 seconds)"
                logger.error(error_msg)
                raise Exception(error_msg)
            elif response.status_code == 500:
                error_msg = "Internal server error"
                logger.error(error_msg)
                raise Exception(error_msg)
            else:
                error_text = response.text
                logger.error(f"API Error {response.status_code}: {error_text}")
                raise Exception(f"API Error {response.status_code}: {error_text}")

        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise Exception(f"Network error: {str(e)}")
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            raise

    # -------------------
    # Async wrapper for image generation
    # -------------------
    async def generate_initial_image(self, prompt: str, resolution: str = "1024x1024",
                                   negative_prompt: Optional[str] = None,
                                   seed: Optional[int] = None) -> bytes:
        """Generate image using Stability AI Ultra endpoint"""
        loop = asyncio.get_event_loop()
        
        try:
            logger.info("Generating image with Stability AI Ultra...")
            image_data = await loop.run_in_executor(
                None, 
                self.generate_image_sync, 
                prompt, 
                resolution,
                negative_prompt,
                seed
            )
            logger.info("Successfully generated image using Ultra endpoint")
            return image_data
            
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            raise Exception(f"Failed to generate image: {e}")

    # -------------------
    # Generate video from image (enhanced with more effects)
    # -------------------
    async def create_simple_video_from_image(self, image_data: bytes, duration: int = 5) -> str:
        """Create video from image with enhanced visual effects"""
        
        # Load and process image
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        height, width, _ = cv2_image.shape

        # Ensure dimensions are even for H.264 compatibility
        if width % 2 != 0:
            width -= 1
        if height % 2 != 0:
            height -= 1
        
        cv2_image = cv2.resize(cv2_image, (width, height))

        timestamp = int(time.time())
        filename = f"video_{timestamp}.mp4"
        video_path = os.path.join(self.output_dir, filename)

        # Use H.264 codec for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 30
        total_frames = duration * fps
        
        # Initialize video writer
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            logger.error("Failed to open video writer")
            raise Exception("Could not initialize video writer")

        logger.info(f"Creating video: {width}x{height}, {total_frames} frames at {fps}fps")

        try:
            for frame_idx in range(total_frames):
                # Create dynamic effects based on time
                time_factor = frame_idx / total_frames
                
                # Combine multiple effects for more dynamic video
                zoom_factor = 1.0 + 0.15 * np.sin(time_factor * 2 * np.pi * 0.5)  # Slow zoom
                rotation_angle = 3 * np.sin(time_factor * 2 * np.pi * 0.3)  # Gentle rotation
                
                # Apply transformations
                center = (width // 2, height // 2)
                M = cv2.getRotationMatrix2D(center, rotation_angle, zoom_factor)
                
                # Add subtle translation for breathing effect
                M[0, 2] += 5 * np.sin(time_factor * 2 * np.pi * 0.7)
                M[1, 2] += 3 * np.cos(time_factor * 2 * np.pi * 0.9)
                
                # Apply transformation
                frame = cv2.warpAffine(cv2_image, M, (width, height), 
                                     borderMode=cv2.BORDER_REFLECT_101)
                
                # Add subtle brightness and contrast variation
                brightness = 1.0 + 0.08 * np.sin(time_factor * 2 * np.pi * 0.4)
                contrast = 1.0 + 0.05 * np.cos(time_factor * 2 * np.pi * 0.6)
                frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=(brightness-1)*30)
                
                # Add subtle color temperature shift
                if frame_idx % 10 == 0:  # Every 10th frame for performance
                    temp_shift = 0.02 * np.sin(time_factor * 2 * np.pi * 0.2)
                    if temp_shift > 0:  # Warmer
                        frame[:, :, 0] = np.clip(frame[:, :, 0] * (1 - temp_shift), 0, 255)
                        frame[:, :, 2] = np.clip(frame[:, :, 2] * (1 + temp_shift), 0, 255)
                    else:  # Cooler
                        frame[:, :, 0] = np.clip(frame[:, :, 0] * (1 - temp_shift), 0, 255)
                        frame[:, :, 2] = np.clip(frame[:, :, 2] * (1 + temp_shift), 0, 255)
                
                video_writer.write(frame)
                
                # Progress logging
                if frame_idx % (fps * 2) == 0:  # Every 2 seconds
                    progress = (frame_idx / total_frames) * 100
                    logger.info(f"Video creation progress: {progress:.1f}%")

        finally:
            video_writer.release()

        # Verify video was created successfully
        if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
            logger.info(f"Enhanced video created successfully: {video_path}")
            logger.info(f"Video file size: {os.path.getsize(video_path)} bytes")
            return filename
        else:
            raise Exception("Video creation failed - output file is empty or missing")

    # -------------------
    # Main video generation pipeline
    # -------------------
    async def generate_video(self, request: GenerationRequest) -> GenerationResponse:
        start_time = time.time()
        request_id = f"{int(time.time())}_{abs(hash(request.prompt)) % 10000}"

        try:
            cleaned_prompt = self.clean_prompt(request.prompt)
            logger.info(f"Starting video generation for request: {request_id}")
            logger.info(f"Prompt: {cleaned_prompt[:100]}...")

            # Step 1: Generate image using Stability AI Ultra
            logger.info("Step 1: Generating initial image with Stability AI Ultra...")
            image_data = await self.generate_initial_image(
                prompt=cleaned_prompt,
                resolution=request.resolution,
                negative_prompt=request.negative_prompt,
                seed=request.seed
            )
            logger.info(f"Image generated successfully, size: {len(image_data)} bytes")

            # Step 2: Create video from image
            logger.info("Step 2: Creating enhanced video from image...")
            filename = await self.create_simple_video_from_image(image_data, request.duration)

            generation_time = time.time() - start_time
            logger.info(f"Video generation completed successfully in {generation_time:.2f}s")
            
            return GenerationResponse(
                success=True,
                request_id=request_id,
                message="Video generated successfully using Stability AI Ultra",
                video_url=f"/download/{filename}",
                filename=filename,
                generation_time=generation_time,
                metadata={
                    "original_prompt": request.prompt,
                    "cleaned_prompt": cleaned_prompt,
                    "duration": request.duration,
                    "resolution": request.resolution,
                    "style": request.style,
                    "negative_prompt": request.negative_prompt,
                    "seed": request.seed,
                    "generated_at": datetime.now().isoformat(),
                    "credits_used": 8,  # Ultra service uses 8 credits per image
                    "api_endpoint": "stability-ai-ultra"
                }
            )

        except Exception as e:
            generation_time = time.time() - start_time
            error_message = str(e)
            logger.error(f"Video generation failed for request {request_id}: {error_message}")
            
            return GenerationResponse(
                success=False,
                request_id=request_id,
                message=f"Video generation failed: {error_message}",
                generation_time=generation_time,
                metadata={
                    "error_type": type(e).__name__,
                    "original_prompt": request.prompt,
                    "failed_at": datetime.now().isoformat()
                }
            )

    # -------------------
    # Utility methods
    # -------------------
    async def test_api_connection(self) -> Dict[str, Any]:
        """Test the Stability AI connection with a simple request"""
        test_prompt = "A simple test image"
        
        try:
            logger.info("Testing Stability AI API connection...")
            image_data = await self.generate_initial_image(test_prompt, "512x512")
            
            return {
                "status": "success",
                "message": "API connection successful",
                "image_size": len(image_data),
                "endpoint": self.image_endpoint
            }
            
        except Exception as e:
            return {
                "status": "error", 
                "message": f"API connection failed: {str(e)}",
                "endpoint": self.image_endpoint
            }

    def get_output_path(self, filename: str) -> str:
        """Get full path to output file"""
        return os.path.join(self.output_dir, filename)

    def list_generated_videos(self) -> list:
        """List all generated video files"""
        try:
            files = [f for f in os.listdir(self.output_dir) if f.endswith('.mp4')]
            return sorted(files, key=lambda x: os.path.getctime(self.get_output_path(x)), reverse=True)
        except Exception as e:
            logger.error(f"Error listing videos: {e}")
            return []


# # -------------------
# # Example usage and testing
# # -------------------
# async def main():
#     """Example usage of the VideoGenerator"""
#     try:
#         # Initialize generator
#         generator = VideoGenerator.from_environment()
        
#         # Test API connection
#         connection_test = await generator.test_api_connection()
#         print(f"API Test: {connection_test}")
        
#         if connection_test["status"] != "success":
#             print("API connection failed, exiting...")
#             return
        
#         # Create a sample video
#         request = GenerationRequest(
#             prompt="A majestic mountain landscape with snow-capped peaks, golden hour lighting, cinematic composition",
#             duration=8,
#             resolution="1024x1024",
#             negative_prompt="blurry, low quality, distorted",
#             seed=42
#         )
        
#         print(f"Generating video with prompt: {request.prompt}")
#         result = await generator.generate_video(request)
        
#         if result.success:
#             print(f"✅ Video generated successfully!")
#             print(f"📁 Filename: {result.filename}")
#             print(f"⏱️  Generation time: {result.generation_time:.2f}s")
#             print(f"📊 Metadata: {result.metadata}")
#         else:
#             print(f"❌ Generation failed: {result.message}")
            
#     except Exception as e:
#         print(f"Error in main: {e}")

