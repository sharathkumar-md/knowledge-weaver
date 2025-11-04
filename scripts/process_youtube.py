"""
YouTube Video Processing Script
Extracts video metadata, description, and transcript using yt-dlp
"""

import json
from pathlib import Path
from typing import Dict, List
import yt_dlp
from loguru import logger
from datetime import datetime

# Configure paths
DATA_DIR = Path(__file__).parent.parent / 'data'
YOUTUBE_DIR = DATA_DIR / 'raw' / 'youtube'
WEBPAGES_DIR = DATA_DIR / 'raw' / 'webpages'
VIDEO_LOG = DATA_DIR / 'raw' / 'video_log.json'
PAGES_LOG = DATA_DIR / 'raw' / 'pages_log.json'


def get_saved_videos() -> List[Dict]:
    """Get all saved YouTube videos from logs"""
    videos = []

    # From video_log.json
    if VIDEO_LOG.exists():
        with open(VIDEO_LOG, 'r', encoding='utf-8') as f:
            videos.extend(json.load(f))

    # From pages_log.json (video type)
    if PAGES_LOG.exists():
        with open(PAGES_LOG, 'r', encoding='utf-8') as f:
            pages = json.load(f)
            videos.extend([p for p in pages if p.get('type') == 'video'])

    logger.info(f"Found {len(videos)} saved videos")
    return videos


def extract_video_info(url: str) -> Dict:
    """Extract video metadata and transcript using yt-dlp"""
    try:
        logger.info(f"Extracting info for: {url}")

        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en'],
            'skip_download': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            # Extract relevant information
            video_data = {
                'title': info.get('title', 'Unknown'),
                'description': info.get('description', ''),
                'channel': info.get('channel', info.get('uploader', 'Unknown')),
                'channel_id': info.get('channel_id', ''),
                'upload_date': info.get('upload_date', ''),
                'duration': info.get('duration', 0),
                'view_count': info.get('view_count', 0),
                'like_count': info.get('like_count', 0),
                'categories': info.get('categories', []),
                'tags': info.get('tags', []),
                'thumbnail': info.get('thumbnail', ''),
                'webpage_url': info.get('webpage_url', url),
                'video_id': info.get('id', ''),
            }

            # Get transcript/subtitles
            transcript = extract_transcript(info)
            video_data['transcript'] = transcript
            video_data['transcript_available'] = bool(transcript)

            logger.success(f"Extracted info for: {video_data['title']}")
            return video_data

    except Exception as e:
        logger.error(f"Failed to extract video info: {e}")
        return None


def extract_transcript(info: Dict) -> str:
    """Extract transcript from subtitles"""
    try:
        # Try automatic captions first (usually better formatted)
        if 'automatic_captions' in info and 'en' in info['automatic_captions']:
            subtitles = info['automatic_captions']['en']
        elif 'subtitles' in info and 'en' in info['subtitles']:
            subtitles = info['subtitles']['en']
        else:
            return ""

        # Find the best subtitle format (prefer json3 for timestamps)
        subtitle_url = None
        for sub in subtitles:
            if sub.get('ext') == 'json3':
                subtitle_url = sub.get('url')
                break

        if not subtitle_url:
            # Fallback to any available format
            for sub in subtitles:
                if 'url' in sub:
                    subtitle_url = sub['url']
                    break

        if subtitle_url:
            import urllib.request
            import json

            with urllib.request.urlopen(subtitle_url) as response:
                subtitle_data = json.loads(response.read())

            # Extract text from JSON3 format
            if 'events' in subtitle_data:
                transcript_lines = []
                for event in subtitle_data['events']:
                    if 'segs' in event:
                        line = ''.join(seg.get('utf8', '') for seg in event['segs'])
                        if line.strip():
                            transcript_lines.append(line.strip())

                return '\n'.join(transcript_lines)

        return ""

    except Exception as e:
        logger.warning(f"Could not extract transcript: {e}")
        return ""


def format_upload_date(upload_date: str) -> str:
    """Format YYYYMMDD to readable date"""
    try:
        if upload_date and len(upload_date) == 8:
            date_obj = datetime.strptime(upload_date, '%Y%m%d')
            return date_obj.strftime('%Y-%m-%d')
    except:
        pass
    return upload_date


def save_video_content(video_data: Dict):
    """Save extracted video data as markdown"""
    if not video_data:
        return

    title = video_data.get('title', 'Unknown Video')
    video_id = video_data.get('video_id', 'unknown')

    # Create safe filename
    safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in title)
    safe_title = safe_title[:100]  # Limit length
    filename = f"youtube_{video_id}_{safe_title}.md"
    filepath = WEBPAGES_DIR / filename

    # Format duration
    duration = video_data.get('duration', 0)
    duration_str = f"{duration // 60}:{duration % 60:02d}" if duration else "Unknown"

    # Create markdown content
    markdown = f"""# {title}

**URL:** {video_data.get('webpage_url', '')}
**Type:** video
**Channel:** {video_data.get('channel', 'Unknown')}
**Channel ID:** {video_data.get('channel_id', '')}
**Published:** {format_upload_date(video_data.get('upload_date', ''))}
**Duration:** {duration_str}
**Views:** {video_data.get('view_count', 0):,}
**Likes:** {video_data.get('like_count', 0):,}
**Video ID:** {video_id}
**Categories:** {', '.join(video_data.get('categories', []))}
**Tags:** {', '.join(video_data.get('tags', [])[:20])}
**Transcript Available:** {"Yes" if video_data.get('transcript_available') else "No"}

---

## Description

{video_data.get('description', 'No description available.')}

---

## Transcript

{video_data.get('transcript', 'No transcript available.')}
"""

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(markdown)

    logger.success(f"Saved video content to: {filepath}")

    # Also save transcript as separate file if available
    if video_data.get('transcript'):
        transcript_file = WEBPAGES_DIR / f"youtube_transcript_{video_id}.txt"
        with open(transcript_file, 'w', encoding='utf-8') as f:
            f.write(video_data['transcript'])
        logger.info(f"Saved transcript to: {transcript_file}")


def is_valid_youtube_url(url: str) -> bool:
    """Check if URL is a valid YouTube video URL"""
    import re
    patterns = [
        r'youtube\.com/watch\?v=[a-zA-Z0-9_-]{11}',
        r'youtu\.be/[a-zA-Z0-9_-]{11}',
        r'youtube\.com/embed/[a-zA-Z0-9_-]{11}',
    ]
    return any(re.search(pattern, url) for pattern in patterns)


def process_all_videos():
    """Process all saved YouTube videos"""
    videos = get_saved_videos()

    if not videos:
        logger.info("No videos to process")
        return

    # Get unique valid video URLs
    urls = set()
    for video in videos:
        url = video.get('url', '')
        if is_valid_youtube_url(url):
            urls.add(url)
        elif 'youtube.com' in url or 'youtu.be' in url:
            logger.warning(f"Skipping invalid YouTube URL: {url}")

    logger.info(f"Processing {len(urls)} unique valid videos")

    for url in urls:
        try:
            video_data = extract_video_info(url)
            if video_data:
                save_video_content(video_data)
        except Exception as e:
            logger.error(f"Failed to process {url}: {e}")


if __name__ == '__main__':
    logger.info("Starting YouTube video processing...")
    logger.info(f"Output directory: {WEBPAGES_DIR}")

    try:
        process_all_videos()
        logger.success("YouTube processing complete!")
    except Exception as e:
        logger.error(f"Error processing videos: {e}")
