"""Video and YouTube transcript processor"""

from pathlib import Path
from typing import Dict, Any, Optional
import re
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from .base_processor import BaseProcessor
from loguru import logger


class VideoProcessor(BaseProcessor):
    """Processor for video transcripts (YouTube URLs or local video files)"""

    SUPPORTED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.transcript_enabled = config.get('ingestion', {}).get('transcript_enabled', True)

    def can_process(self, file_path: Path) -> bool:
        """Check if file is a video or text file containing YouTube URL"""
        if file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
            return True

        # Check if it's a text file with YouTube URL
        if file_path.suffix.lower() in {'.txt', '.url'}:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    return 'youtube.com' in content or 'youtu.be' in content
            except Exception:
                return False

        return False

    def extract_text(self, file_path: Path) -> str:
        """Extract transcript from video"""
        if not self.transcript_enabled:
            logger.warning(f"Transcript extraction disabled, skipping: {file_path.name}")
            return ""

        # Check if it's a YouTube URL reference
        if file_path.suffix.lower() in {'.txt', '.url'}:
            return self._extract_youtube_transcript(file_path)

        # For local video files, you'd need whisper or other ASR
        # For now, we'll skip local video processing
        logger.warning(f"Local video processing not implemented yet: {file_path.name}")
        logger.info("Tip: Use OpenAI Whisper for local video transcription")
        return ""

    def _extract_youtube_transcript(self, file_path: Path) -> str:
        """Extract transcript from YouTube video"""
        try:
            # Read file to get YouTube URL
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract video ID
            video_id = self._extract_youtube_id(content)

            if not video_id:
                logger.warning(f"No valid YouTube URL found in {file_path.name}")
                return ""

            # Get transcript
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)

            # Format transcript
            text_parts = []
            for entry in transcript_list:
                timestamp = self._format_timestamp(entry['start'])
                text = entry['text']
                text_parts.append(f"[{timestamp}] {text}")

            transcript = '\n'.join(text_parts)

            logger.info(f"Extracted YouTube transcript: {video_id}")
            return transcript

        except TranscriptsDisabled:
            logger.error(f"Transcripts disabled for video: {video_id}")
            return ""

        except NoTranscriptFound:
            logger.error(f"No transcript found for video: {video_id}")
            return ""

        except Exception as e:
            logger.error(f"Failed to extract YouTube transcript from {file_path}: {e}")
            return ""

    def _extract_youtube_id(self, text: str) -> Optional[str]:
        """Extract YouTube video ID from URL"""
        # Pattern for youtube.com URLs
        patterns = [
            r'(?:youtube\.com/watch\?v=)([a-zA-Z0-9_-]{11})',
            r'(?:youtu\.be/)([a-zA-Z0-9_-]{11})',
            r'(?:youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)

        return None

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"

    def _create_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Create metadata with video-specific info"""
        metadata = super()._create_metadata(file_path)

        # Try to extract video ID if YouTube
        if file_path.suffix.lower() in {'.txt', '.url'}:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                video_id = self._extract_youtube_id(content)
                if video_id:
                    metadata['youtube_video_id'] = video_id
                    metadata['youtube_url'] = f"https://www.youtube.com/watch?v={video_id}"

            except Exception:
                pass

        return metadata
