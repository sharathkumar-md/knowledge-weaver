"""Simple Flask API server for browser extension integration"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import json
from datetime import datetime
from loguru import logger

app = Flask(__name__)
CORS(app)  # Enable CORS for browser extension

# Configure data directory
YOUTUBE_DIR = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'youtube'
WEBPAGES_DIR = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'webpages'
YOUTUBE_DIR.mkdir(parents=True, exist_ok=True)
WEBPAGES_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_LOG = YOUTUBE_DIR.parent / 'video_log.json'
PAGES_LOG = WEBPAGES_DIR.parent / 'pages_log.json'


@app.route('/api/save-video', methods=['POST'])
def save_video():
    """Save YouTube video URL for later processing"""
    try:
        data = request.json

        video_id = data.get('videoId')
        url = data.get('url')
        title = data.get('title', 'Unknown Title')
        saved_at = data.get('savedAt', datetime.now().isoformat())

        if not video_id or not url:
            return jsonify({'error': 'Missing videoId or url'}), 400

        # Create a text file with the video URL
        filename = f"youtube_{video_id}.txt"
        filepath = YOUTUBE_DIR / filename

        # Write video URL to file (for video processor to pick up)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(url)

        # Log the video
        log_entry = {
            'videoId': video_id,
            'url': url,
            'title': title,
            'savedAt': saved_at,
            'filename': str(filepath)
        }

        # Append to log
        logs = []
        if VIDEO_LOG.exists():
            with open(VIDEO_LOG, 'r', encoding='utf-8') as f:
                logs = json.load(f)

        logs.append(log_entry)

        with open(VIDEO_LOG, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved YouTube video: {title} ({video_id})")

        return jsonify({
            'success': True,
            'message': f'Video saved: {title}',
            'filename': filename
        }), 200

    except Exception as e:
        logger.error(f"Error saving video: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/save-page', methods=['POST'])
def save_page():
    """Save webpage content for later processing"""
    try:
        data = request.json

        url = data.get('url')
        title = data.get('title', 'Untitled Page')
        content = data.get('content', '')
        excerpt = data.get('excerpt', '')
        page_type = data.get('type', 'webpage')
        author = data.get('author')
        publish_date = data.get('publishDate')
        tags = data.get('tags', [])
        word_count = data.get('wordCount', 0)
        domain = data.get('domain', '')
        saved_at = data.get('savedAt', datetime.now().isoformat())

        if not url or not title:
            return jsonify({'error': 'Missing url or title'}), 400

        # Create a safe filename from the title
        safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in title)
        safe_title = safe_title[:100]  # Limit length
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{page_type}_{timestamp}_{safe_title}.md"
        filepath = WEBPAGES_DIR / filename

        # Create markdown content
        markdown_content = f"""# {title}

**URL:** {url}
**Type:** {page_type}
**Domain:** {domain}
**Author:** {author or 'Unknown'}
**Published:** {publish_date or 'Unknown'}
**Word Count:** {word_count}
**Saved At:** {saved_at}
**Tags:** {', '.join(tags) if tags else 'None'}

---

## Excerpt
{excerpt}

---

## Full Content
{content}
"""

        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        # Log the page
        log_entry = {
            'url': url,
            'title': title,
            'type': page_type,
            'domain': domain,
            'author': author,
            'publishDate': publish_date,
            'wordCount': word_count,
            'tags': tags,
            'savedAt': saved_at,
            'filename': str(filepath)
        }

        # Append to log
        logs = []
        if PAGES_LOG.exists():
            with open(PAGES_LOG, 'r', encoding='utf-8') as f:
                logs = json.load(f)

        logs.append(log_entry)

        with open(PAGES_LOG, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved webpage: {title} ({page_type}) from {domain}")

        return jsonify({
            'success': True,
            'message': f'Page saved: {title}',
            'filename': filename
        }), 200

    except Exception as e:
        logger.error(f"Error saving page: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get statistics about saved videos and pages"""
    try:
        # Video stats
        video_count = 0
        videos = []
        if VIDEO_LOG.exists():
            with open(VIDEO_LOG, 'r', encoding='utf-8') as f:
                videos = json.load(f)
                video_count = len(videos)

        # Page stats
        page_count = 0
        pages = []
        if PAGES_LOG.exists():
            with open(PAGES_LOG, 'r', encoding='utf-8') as f:
                pages = json.load(f)
                page_count = len(pages)

        return jsonify({
            'total_videos': video_count,
            'total_pages': page_count,
            'total_items': video_count + page_count,
            'videos': videos[-10:],  # Return last 10
            'pages': pages[-10:]     # Return last 10
        }), 200

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'service': 'Knowledge Weaver API',
        'version': '1.0.0'
    }), 200


if __name__ == '__main__':
    logger.info("Starting Knowledge Weaver API server...")
    logger.info(f"YouTube storage: {YOUTUBE_DIR}")
    logger.info(f"Webpages storage: {WEBPAGES_DIR}")
    print("\n" + "="*60)
    print("Knowledge Weaver API Server")
    print("="*60)
    print(f"\nServer running at: http://localhost:5000")
    print(f"\nData directories:")
    print(f"  YouTube videos: {YOUTUBE_DIR}")
    print(f"  Webpages: {WEBPAGES_DIR}")
    print("\nEndpoints:")
    print("  POST /api/save-video  - Save YouTube video")
    print("  POST /api/save-page   - Save webpage (articles, blogs, news, etc)")
    print("  GET  /api/stats       - Get statistics")
    print("  GET  /api/health      - Health check")
    print("\nPress Ctrl+C to stop\n")

    app.run(host='0.0.0.0', port=5000, debug=True)
