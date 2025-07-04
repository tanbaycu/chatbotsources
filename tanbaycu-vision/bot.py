import os
import logging
import tempfile
import time
import asyncio
import subprocess
import uuid
import datetime
import re
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
import requests
from google import genai
from deep_translator import GoogleTranslator
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Environment variables
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GOOGLE_DRIVE_CREDENTIALS = ""
GOOGLE_DRIVE_FOLDER_ID = "" 

# Video retention settings (in days)
VIDEO_RETENTION_DAYS = 7  # Delete videos older than 7 days

# Initialize Gemini client
genai_client = genai.Client(api_key=GEMINI_API_KEY)

# Initialize Google Drive API
try:
    credentials = service_account.Credentials.from_service_account_file(
        GOOGLE_DRIVE_CREDENTIALS, 
        scopes=['https://www.googleapis.com/auth/drive']
    )
    drive_service = build('drive', 'v3', credentials=credentials)
    google_drive_available = True
    logger.info("Google Drive API initialized successfully")
except Exception as e:
    logger.error(f"Google Drive initialization error: {e}")
    google_drive_available = False

# File size limits
TELEGRAM_FILE_SIZE_LIMIT = 50 * 1024 * 1024  # 50MB in bytes
GEMINI_FILE_SIZE_LIMIT = 20 * 1024 * 1024    # 20MB in bytes (Gemini's actual limit)

# Supported video MIME types
SUPPORTED_VIDEO_TYPES = [
    'video/mp4', 'video/mpeg', 'video/mov', 'video/avi', 
    'video/x-flv', 'video/mpg', 'video/webm', 'video/wmv', 'video/3gpp'
]

# Video URL patterns
VIDEO_URL_PATTERNS = [
    r'(https?://)?(www\.)?(youtube\.com|youtu\.be)/(watch\?v=|embed/|v/|shorts/)?([a-zA-Z0-9_-]{11})',  # YouTube
    r'(https?://)?(www\.)?(vimeo\.com)/([0-9]+)',  # Vimeo
    r'(https?://)?(www\.)?(dailymotion\.com|dai\.ly)/video/([a-zA-Z0-9]+)',  # Dailymotion
    r'(https?://)?(www\.)?(facebook\.com|fb\.watch)/([a-zA-Z0-9_\-.]+/videos/|watch/\?v=)([0-9]+)',  # Facebook
    r'(https?://)?(www\.)?(instagram\.com|instagr\.am)/(?:p|tv|reel)/([a-zA-Z0-9_-]+)',  # Instagram
    r'(https?://)?(www\.)?(tiktok\.com)/@([a-zA-Z0-9_\-.]+)/video/([0-9]+)',  # TikTok
    r'(https?://)?(drive\.google\.com)/file/d/([a-zA-Z0-9_-]+)',  # Google Drive
    r'(https?://)?([a-zA-Z0-9_-]+\.)*(mp4|mov|avi|webm|mkv|flv|wmv|mpg|mpeg|3gp)$'  # Direct video URLs
]

# Common query suggestions
QUERY_SUGGESTIONS = [
    "Summarize this video",
    "Describe what happens in the video",
    "Create a quiz based on this video",
    "Transcribe the audio in this video",
    "Analyze the visual elements in this video",
    "What is the main message of this video?",
    "Identify key moments in this video"
]

def translate_to_vietnamese(text):
    """Translate text from English to Vietnamese using deep_translator."""
    try:
        # Split text into chunks to handle long texts (Google Translator has a limit)
        max_chunk_size = 4000  # Reduced from 5000 to be safer
        chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        
        translated_chunks = []
        for chunk in chunks:
            translated = GoogleTranslator(source='en', target='vi').translate(chunk)
            translated_chunks.append(translated)
        
        return "".join(translated_chunks)
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return text + "\n\n(Translation failed. Original English response shown.)"

def is_video_url(text):
    """Check if the text contains a video URL."""
    for pattern in VIDEO_URL_PATTERNS:
        if re.search(pattern, text):
            return True
    return False

def extract_video_url(text):
    """Extract video URL from text."""
    for pattern in VIDEO_URL_PATTERNS:
        match = re.search(pattern, text)
        if match:
            # Return the full matched URL
            return match.group(0)
    return None

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    await update.message.reply_text(
        "👋 Welcome to the Video Analysis Bot!\n\n"
        "Send me a video or a video URL and I'll analyze it for you.\n\n"
        "⚠️ *File Size Handling*:\n"
        "• Videos under 20MB: Direct processing\n"
        "• Videos 20-50MB: Google Drive upload\n"
        "• Maximum size: 50MB (Telegram limit)\n\n"
        "✅ *URL Support*:\n"
        "• You can also send a video URL (YouTube, Vimeo, etc.)\n"
        "• Just paste the URL and I'll analyze it directly\n\n"
        "Supported formats: MP4, MPEG, MOV, AVI, FLV, MPG, WEBM, WMV, 3GPP\n\n"
        "After sending a video, you can ask questions about it or request analysis.\n\n"
        "Type /help for detailed instructions.\n\n"
        "Note: Analysis results will be automatically translated to Vietnamese! 🇻🇳",
        parse_mode="Markdown"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    help_text = (
        "📋 *Video Analysis Bot - Detailed Help*\n\n"
        
        "*Step 1: Send a Video or URL*\n"
        "• Send a video file (up to 50MB) OR\n"
        "• Send a video URL (YouTube, Vimeo, Google Drive, etc.)\n"
        "• For videos under 20MB: Direct processing\n"
        "• For videos 20-50MB: Google Drive upload\n"
        "• Supported formats: MP4, MPEG, MOV, AVI, FLV, MPG, WEBM, WMV, 3GPP\n\n"
        
        "*Step 2: Wait for Processing*\n"
        "• The bot will show download and processing progress\n"
        "• For larger videos, there will be additional steps (upload to Google Drive)\n"
        "• Processing time depends on video length, size, and complexity\n"
        "• Please be patient during processing\n\n"
        
        "*Step 3: Ask Questions*\n"
        "• After processing, you'll see suggested questions\n"
        "• Click on a suggestion or type your own question\n"
        "• To type your own question, click 'Type Own Question' button\n"
        "• When prompted, send your question as a new message\n\n"
        
        "*Example Questions:*\n"
        "• 'Summarize this video'\n"
        "• 'What happens at 01:05?'\n"
        "• 'Transcribe the audio'\n"
        "• 'Create a quiz based on this video'\n"
        "• 'Identify the main topics discussed'\n\n"
        
        "*Additional Commands:*\n"
        "• /start - Start the bot\n"
        "• /help - Show this help message\n"
        "• /suggestions - Show query suggestions\n"
        "• /info - Show bot information and version history\n"
        "• /cleanup - Manually trigger cleanup of old videos\n\n"
        
        "*Storage Information:*\n"
        "• Videos over 20MB are automatically stored in Google Drive\n"
        "• Local files are deleted after processing to save space\n"
        "• You can manually trigger cleanup with /cleanup\n\n"
        
        "*Language:*\n"
        "• All analysis results are automatically translated to Vietnamese\n"
        "• You can view the original English response using the 'Show Original' button\n\n"
        
        "*Troubleshooting:*\n"
        "• If your video is too large (>50MB), try trimming it or send a URL\n"
        "• If processing fails, try a different video format\n"
        "• If the bot doesn't respond, try sending /start to reset\n\n"
        
        "All analysis results will be automatically translated to Vietnamese! 🇻🇳"
    )
    
    await update.message.reply_text(
        help_text,
        parse_mode="Markdown"
    )

async def info_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Hiển thị thông tin về bot."""
    info_text = (
        "📱 *Video Analysis Bot*\n\n"
        
        "📋 *Thông tin bot:*\n"
        "Bot phân tích video sử dụng AI để trả lời các câu hỏi về nội dung video. "
        "Hỗ trợ nhiều định dạng video và tự động dịch kết quả phân tích sang tiếng Việt.\n\n"
        
        "📜 *Lịch sử cập nhật:* 09/03/2025\n\n"
        
        "1️⃣3️⃣ *Cập nhật v1.8.0 (Mới nhất)*\n"
        "   • 🔗 Thêm tính năng phân tích video từ URL\n"
        "   • 🧹 Tự động xóa file cục bộ sau khi xử lý\n"
        "   • 📱 Cải thiện xử lý lỗi kích thước file\n"
        "   • 🛠️ Tối ưu hóa quy trình xử lý\n\n"
        
        "1️⃣2️⃣ *Cập nhật v1.7.0*\n"
        "   • 🔄 Chuyển sang sử dụng Google Drive\n"
        "   • 📂 Lưu trữ video trong thư mục chỉ định\n"
        "   • 📱 Cải thiện trải nghiệm người dùng\n\n"
        
        "1️⃣1️⃣ *Cập nhật v1.6.0*\n"
        "   • 🌐 Thêm tính năng tải video lên cloud cho file lớn\n"
        "   • 🔗 Sử dụng URL video thay vì file trực tiếp\n"
        "   • 📈 Tăng giới hạn kích thước video lên 50MB\n"
        "   • 🛠️ Cải thiện xử lý lỗi\n"
        "   • 📱 Tối ưu hóa trải nghiệm người dùng\n\n"
        
        "🔟 *Cập nhật v1.5.0*\n"
        "   • 🎬 Thêm tính năng nén video tự động\n"
        "   • ⚠️ Cập nhật giới hạn dung lượng thực tế (20MB)\n"
        "   • 📊 Hiển thị thông tin chi tiết về lỗi\n"
        "   • 🛠️ Cải thiện xử lý file lớn\n"
        "   • 📱 Tối ưu hóa trải nghiệm người dùng\n\n"
        
        "💡 *Sắp tới:*\n"
        "• 📊 Cải thiện độ chính xác phân tích\n"
        "• 🎯 Thêm tính năng phân tích theo thời điểm cụ thể\n"
        "• 🔍 Nâng cao khả năng nhận diện đối tượng\n"
        "• 📱 Tối ưu hóa trải nghiệm người dùng\n\n"
        
        "📝 *Ghi chú:*\n"
        "• Bot sẽ tiếp tục được cập nhật và cải thiện\n"
        "• Báo cáo lỗi hoặc góp ý tại: @tanbaycu\n\n"
        
        "👨‍💻 *Tác giả:* tanbaycu - Powered by Gemini 2.0\n"
        "🆘 *Hỗ trợ:* @tanbaycu\n\n"
        
        "Sử dụng /help để xem hướng dẫn sử dụng chi tiết."
    )
    
    await update.message.reply_text(
        info_text,
        parse_mode="Markdown"
    )

async def cleanup_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Manually trigger cleanup of old videos."""
    if not google_drive_available:
        await update.message.reply_text(
            "⚠️ Google Drive is not available. Cannot perform cleanup."
        )
        return
    
    await update.message.reply_text("🧹 Starting cleanup of old videos...")
    
    try:
        # Calculate the cutoff date (videos older than this will be deleted)
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=VIDEO_RETENTION_DAYS)
        cutoff_timestamp = cutoff_date.strftime('%Y-%m-%dT%H:%M:%S')
        
        # Query for videos in the specified folder older than the cutoff date
        query = f"'{GOOGLE_DRIVE_FOLDER_ID}' in parents and mimeType contains 'video/' and modifiedTime < '{cutoff_timestamp}'"
        
        results = drive_service.files().list(
            q=query,
            spaces='drive',
            fields='files(id, name, modifiedTime)'
        ).execute()
        
        files = results.get('files', [])
        
        if not files:
            await update.message.reply_text("✅ No old videos found that need cleanup.")
            return
        
        # Delete the old files
        deleted_count = 0
        for file in files:
            try:
                drive_service.files().delete(fileId=file['id']).execute()
                deleted_count += 1
                logger.info(f"Deleted old video: {file['name']}")
            except Exception as e:
                logger.error(f"Error deleting file {file['name']}: {e}")
        
        await update.message.reply_text(
            f"✅ Cleanup completed. Deleted {deleted_count} old videos."
        )
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        await update.message.reply_text(
            f"❌ Error during cleanup: {str(e)}"
        )
    
async def suggestions_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show query suggestions as buttons."""
    # Check if there's a video file associated with this user
    if not context.user_data.get('video_files', {}).get('current') and not context.user_data.get('video_urls', {}).get('current'):
        await update.message.reply_text(
            "Please send a video first before requesting suggestions."
        )
        return
    
    keyboard = []
    for suggestion in QUERY_SUGGESTIONS:
        keyboard.append([InlineKeyboardButton(suggestion, callback_data=f"query:{suggestion}")])
    
    # Add "Type Own Question" button
    keyboard.append([InlineKeyboardButton("Type Own Question", callback_data="type_own_question")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "Select a query or type your own question:",
        reply_markup=reply_markup
    )

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle button callbacks."""
    query = update.callback_query
    await query.answer()
    
    data = query.data
    
    if data.startswith("query:"):
        selected_query = data[6:]  # Remove "query:" prefix
        
        # Send the selected query as a new message
        await query.message.reply_text(f"Processing: {selected_query}")
        
        # Process the query
        await process_text(query.message, context, selected_query)
    
    elif data == "more_suggestions":
        # Show more suggestions
        keyboard = []
        for suggestion in QUERY_SUGGESTIONS:
            keyboard.append([InlineKeyboardButton(suggestion, callback_data=f"query:{suggestion}")])
        
        # Add "Type Own Question" button
        keyboard.append([InlineKeyboardButton("Type Own Question", callback_data="type_own_question")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            "Select a query or type your own question:",
            reply_markup=reply_markup
        )
    
    elif data == "type_own_question":
        # Prompt user to type their own question
        await query.edit_message_text(
            "Please type and send your question about the video now."
        )
        
        # Set conversation state to wait for user's question
        context.user_data['waiting_for_question'] = True
    
    elif data.startswith("original:"):
        # Show original English text
        response_hash = data.split(":")[1]
        original_text = context.user_data.get('original_responses', {}).get(response_hash)
        
        if not original_text:
            await query.edit_message_text("Original text not found.")
            return
        
        # Create button to go back to translated version
        keyboard = [
            [InlineKeyboardButton("Back to Vietnamese", callback_data=f"translate:{response_hash}")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Edit message to show original text
        await query.edit_message_text(
            f"📊 *Analysis Results* (Original English)\n\n"
            f"{original_text}",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )
    
    elif data.startswith("translate:"):
        # Show translated Vietnamese text
        response_hash = data.split(":")[1]
        original_text = context.user_data.get('original_responses', {}).get(response_hash)
        
        if not original_text:
            await query.edit_message_text("Original text not found.")
            return
        
        # Translate the text
        translated_text = translate_to_vietnamese(original_text)
        
        # Create button to go back to original version
        keyboard = [
            [InlineKeyboardButton("Show Original (English)", callback_data=f"original:{response_hash}")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Edit message to show translated text
        await query.edit_message_text(
            f"📊 *Kết Quả Phân Tích* (Đã dịch sang tiếng Việt)\n\n"
            f"{translated_text}",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

async def upload_to_google_drive(file_path, user_id):
    """Upload video to Google Drive and return public URL."""
    if not google_drive_available:
        return None
    
    try:
        # Create a unique filename
        file_name = f"video_{user_id}_{int(time.time())}{os.path.splitext(file_path)[1]}"
        
        # Create file metadata
        file_metadata = {
            'name': file_name,
            'parents': [GOOGLE_DRIVE_FOLDER_ID]
        }
        
        # Create media
        media = MediaFileUpload(
            file_path,
            mimetype='video/mp4',
            resumable=True
        )
        
        # Upload file to Google Drive
        file = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id,webViewLink'
        ).execute()
        
        # Make the file accessible via link
        permission = {
            'type': 'anyone',
            'role': 'reader'
        }
        drive_service.permissions().create(
            fileId=file['id'],
            body=permission
        ).execute()
        
        # Get the web view link
        file = drive_service.files().get(
            fileId=file['id'],
            fields='webViewLink'
        ).execute()
        
        # Return file info
        return {
            'id': file['id'],
            'url': file['webViewLink'],
            'name': file_name,
            'upload_time': datetime.datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error uploading to Google Drive: {e}")
        return None

async def process_video_url(update: Update, context: ContextTypes.DEFAULT_TYPE, url) -> None:
    """Process a video URL."""
    # Notify user that processing has started
    status_message = await update.message.reply_text(
        f"🔗 Processing video URL: {url}\n"
        f"This may take a moment..."
    )
    
    try:
        # Store the video URL in user data for later use
        if not context.user_data.get('video_urls'):
            context.user_data['video_urls'] = {}
        context.user_data['video_urls']['current'] = url
        context.user_data['video_duration'] = "Unknown"  # Duration unknown for URLs
        context.user_data['video_type'] = 'url'  # Mark as URL-based
        
        # Create a prompt that includes the video URL
        prompt = f"Analyze this video: {url}\n\nPlease confirm you can access and view this video."
        
        # Generate content using Gemini API with the URL in the prompt
        response = genai_client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=prompt
        )
        
        # Check if Gemini could access the video
        if "cannot access" in response.text.lower() or "cannot view" in response.text.lower() or "unable to access" in response.text.lower():
            await status_message.edit_text(
                "❌ Gemini API cannot access the video URL.\n"
                "Please try a different video URL or upload a video file directly."
            )
            return
        
        # Show success message with query options
        await show_query_options(update, context, status_message)
        
    except Exception as e:
        error_message = str(e)
        logger.error(f"Gemini API URL processing error: {error_message}")
        
        await status_message.edit_text(
            f"⚠️ Error processing video URL with Gemini API: {error_message}\n"
            f"Please try a different video URL or upload a video file."
        )

async def process_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process video files sent by the user."""
    # Check if the message contains a video
    if not update.message.video:
        await update.message.reply_text("Please send a valid video file.")
        return

    # Get video file information
    video = update.message.video
    file_size = video.file_size
    mime_type = video.mime_type
    duration = video.duration if hasattr(video, 'duration') else "Unknown"
    
    # Check file size (Telegram API limit is 50MB for bots)
    if file_size > TELEGRAM_FILE_SIZE_LIMIT:
        await update.message.reply_text(
            f"⚠️ Video is too large. Maximum size is {format_size(TELEGRAM_FILE_SIZE_LIMIT)}.\n\n"
            f"Your video: {format_size(file_size)}\n\n"
            f"Please trim your video or reduce its quality to stay under the limit."
        )
        return
    
    # Check if mime type is supported
    if mime_type not in SUPPORTED_VIDEO_TYPES:
        await update.message.reply_text(
            f"⚠️ Unsupported video format: {mime_type}\n\n"
            f"Supported formats: MP4, MPEG, MOV, AVI, FLV, MPG, WEBM, WMV, 3GPP"
        )
        return
    
    # Determine processing method based on file size BEFORE downloading
    use_google_drive = file_size > GEMINI_FILE_SIZE_LIMIT
    
    # Notify user that processing has started
    status_message = await update.message.reply_text(
        f"⏳ Downloading video...\n"
        f"Size: {format_size(file_size)}\n"
        f"Duration: {format_duration(duration)}\n"
        f"Processing method: {'Google Drive' if use_google_drive else 'Direct Gemini API'}"
    )
    
    try:
        # Get file from Telegram
        file = await context.bot.get_file(update.message.video.file_id)
        
        # Create a temporary file to store the video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_path = temp_file.name
        
        # Download the file with progress updates
        start_time = time.time()
        await file.download_to_drive(temp_path)
        download_time = time.time() - start_time
        
        await status_message.edit_text(
            f"✅ Video downloaded in {download_time:.1f}s\n"
            f"Size: {format_size(file_size)}"
        )
        
        # Process based on the determined method
        if not use_google_drive:
            # Small video - direct processing with Gemini
            await status_message.edit_text(
                f"✅ Video downloaded in {download_time:.1f}s\n"
                f"📤 Uploading to Gemini API..."
            )
            
            # Process directly with Gemini
            await process_with_gemini(update, context, status_message, temp_path, file_size, duration)
            
            # Delete the local file after processing
            try:
                os.unlink(temp_path)
                logger.info(f"Deleted local file: {temp_path}")
            except Exception as e:
                logger.error(f"Error deleting local file: {e}")
            
        else:
            # Large video - upload to Google Drive
            if google_drive_available:
                await status_message.edit_text(
                    f"✅ Video downloaded in {download_time:.1f}s\n"
                    f"⚙️ Video size ({format_size(file_size)}) exceeds Gemini's limit of {format_size(GEMINI_FILE_SIZE_LIMIT)}.\n"
                    f"Uploading to Google Drive... This may take a while."
                )
                
                # Upload to Google Drive
                upload_start_time = time.time()
                drive_info = await upload_to_google_drive(temp_path, update.effective_user.id)
                upload_time = time.time() - upload_start_time
                
                if drive_info and drive_info.get('url'):
                    # Google Drive upload successful
                    await status_message.edit_text(
                        f"✅ Video uploaded to Google Drive in {upload_time:.1f}s\n"
                        f"🔗 Using video URL for analysis\n"
                        f"📤 Sending to Gemini API..."
                    )
                    
                    # Store the drive info in user data
                    if not context.user_data.get('drive_videos'):
                        context.user_data['drive_videos'] = []
                    context.user_data['drive_videos'].append(drive_info)
                    
                    # Process with URL
                    await process_with_url(update, context, status_message, drive_info['url'], file_size, duration)
                    
                    # Delete the local file after processing
                    try:
                        os.unlink(temp_path)
                        logger.info(f"Deleted local file: {temp_path}")
                    except Exception as e:
                        logger.error(f"Error deleting local file: {e}")
                    
                else:
                    # Google Drive upload failed
                    await status_message.edit_text(
                        f"⚠️ Google Drive upload failed. Please try a smaller video or contact support."
                    )
                    
                    # Clean up temporary files
                    try:
                        os.unlink(temp_path)
                    except Exception as e:
                        logger.error(f"Error deleting local file: {e}")
            else:
                # Google Drive not available
                await status_message.edit_text(
                    f"⚠️ Video is too large for Gemini API ({format_size(file_size)} > {format_size(GEMINI_FILE_SIZE_LIMIT)}) and Google Drive is not available.\n"
                    f"Please send a smaller video (under {format_size(GEMINI_FILE_SIZE_LIMIT)})."
                )
                
                # Clean up temporary files
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    logger.error(f"Error deleting local file: {e}")
        
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        await status_message.edit_text(
            f"❌ Error processing video: {str(e)}\n\n"
            f"Please try again with a different video or contact support if the issue persists."
        )

async def process_with_gemini(update, context, status_message, video_path, file_size, duration):
    """Process video file with Gemini API."""
    try:
        # Double-check file size before uploading to Gemini
        actual_file_size = os.path.getsize(video_path)
        if actual_file_size > GEMINI_FILE_SIZE_LIMIT:
            await status_message.edit_text(
                f"⚠️ Video is too large for direct processing with Gemini API.\n"
                f"Maximum size: {format_size(GEMINI_FILE_SIZE_LIMIT)}\n"
                f"Your video: {format_size(actual_file_size)}\n\n"
                f"Switching to Google Drive upload..."
            )
            
            # Switch to Google Drive upload
            if google_drive_available:
                upload_start_time = time.time()
                drive_info = await upload_to_google_drive(video_path, update.effective_user.id)
                upload_time = time.time() - upload_start_time
                
                if drive_info and drive_info.get('url'):
                    await status_message.edit_text(
                        f"✅ Video uploaded to Google Drive in {upload_time:.1f}s\n"
                        f"🔗 Using video URL for analysis\n"
                        f"📤 Sending to Gemini API..."
                    )
                    
                    # Store the drive info in user data
                    if not context.user_data.get('drive_videos'):
                        context.user_data['drive_videos'] = []
                    context.user_data['drive_videos'].append(drive_info)
                    
                    # Process with URL
                    await process_with_url(update, context, status_message, drive_info['url'], actual_file_size, duration)
                else:
                    await status_message.edit_text(
                        f"⚠️ Google Drive upload failed. Please try a smaller video."
                    )
            else:
                await status_message.edit_text(
                    f"⚠️ Video is too large for Gemini API and Google Drive is not available.\n"
                    f"Please send a smaller video (under {format_size(GEMINI_FILE_SIZE_LIMIT)})."
                )
            
            return
        
        # Upload to Gemini API
        video_file = genai_client.files.upload(file=video_path)
        
        # Store the video file URI in user data for later use
        if not context.user_data.get('video_files'):
            context.user_data['video_files'] = {}
        context.user_data['video_files']['current'] = video_file.uri
        context.user_data['video_file_name'] = video_file.name
        context.user_data['video_duration'] = duration
        context.user_data['video_type'] = 'file'  # Mark as file-based
        
        # Check if the file is ready for processing
        await status_message.edit_text("🔄 Processing video... This may take a moment.")
        
        video_file_status = genai_client.files.get(name=video_file.name)
        processing_start_time = time.time()
        
        while video_file_status.state.name == "PROCESSING":
            # Wait for processing to complete with progress updates
            elapsed_time = time.time() - processing_start_time
            if elapsed_time > 5:  # Update message every 5 seconds
                await status_message.edit_text(
                    f"🔄 Processing video... ({int(elapsed_time)}s)\n"
                    "This may take a few minutes depending on video length."
                )
                processing_start_time = time.time()  # Reset timer
            
            await asyncio.sleep(2)
            video_file_status = genai_client.files.get(name=video_file.name)
        
        if video_file_status.state.name == "FAILED":
            await status_message.edit_text(
                "❌ Video processing failed. Please try again with a different video.\n"
                "Possible reasons:\n"
                "- Video format not supported by Gemini\n"
                "- Video content issues\n"
                "- Processing timeout"
            )
            return
        
        # Show success message with query options
        await show_query_options(update, context, status_message)
        
    except Exception as e:
        error_message = str(e)
        logger.error(f"Gemini API error: {error_message}")
        
        if "file size" in error_message.lower():
            # If the error is about file size, try Google Drive instead
            await status_message.edit_text(
                f"⚠️ Gemini API rejected the video due to file size limitations.\n"
                f"Switching to Google Drive upload..."
            )
            
            # Try Google Drive upload as fallback
            if google_drive_available:
                upload_start_time = time.time()
                drive_info = await upload_to_google_drive(video_path, update.effective_user.id)
                upload_time = time.time() - upload_start_time
                
                if drive_info and drive_info.get('url'):
                    await status_message.edit_text(
                        f"✅ Video uploaded to Google Drive in {upload_time:.1f}s\n"
                        f"🔗 Using video URL for analysis\n"
                        f"📤 Sending to Gemini API..."
                    )
                    
                    # Store the drive info in user data
                    if not context.user_data.get('drive_videos'):
                        context.user_data['drive_videos'] = []
                    context.user_data['drive_videos'].append(drive_info)
                    
                    # Process with URL
                    await process_with_url(update, context, status_message, drive_info['url'], file_size, duration)
                else:
                    await status_message.edit_text(
                        f"⚠️ Google Drive upload failed. Please try a smaller video."
                    )
            else:
                await status_message.edit_text(
                    f"⚠️ Video is too large for Gemini API and Google Drive is not available.\n"
                    f"Please send a smaller video (under {format_size(GEMINI_FILE_SIZE_LIMIT)})."
                )
        else:
            await status_message.edit_text(
                f"⚠️ Error with Gemini API: {error_message}\n"
                f"Please try again with a different video."
            )

async def process_with_url(update, context, status_message, video_url, file_size, duration):
    """Process video URL with Gemini API."""
    try:
        # Store the video URL in user data for later use
        if not context.user_data.get('video_urls'):
            context.user_data['video_urls'] = {}
        context.user_data['video_urls']['current'] = video_url
        context.user_data['video_duration'] = duration
        context.user_data['video_type'] = 'url'  # Mark as URL-based
        
        await status_message.edit_text("🔄 Processing video from URL... This may take a moment.")
        
        # Create a prompt that includes the video URL
        prompt = f"Analyze this video: {video_url}\n\nPlease confirm you can access and view this video."
        
        # Generate content using Gemini API with the URL in the prompt
        response = genai_client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=prompt
        )
        
        # Check if Gemini could access the video
        if "cannot access" in response.text.lower() or "cannot view" in response.text.lower() or "unable to access" in response.text.lower():
            await status_message.edit_text(
                "❌ Gemini API cannot access the video URL.\n"
                "Please try a different video or use a smaller video for direct upload."
            )
            return
        
        # Show success message with query options
        await show_query_options(update, context, status_message)
        
    except Exception as e:
        error_message = str(e)
        logger.error(f"Gemini API URL processing error: {error_message}")
        
        await status_message.edit_text(
            f"⚠️ Error processing video URL with Gemini API: {error_message}\n"
            f"Please try again with a different video."
        )

async def show_query_options(update, context, status_message):
    """Show query options after successful video processing."""
    # Create suggestion buttons
    keyboard = []
    for suggestion in QUERY_SUGGESTIONS[:4]:  # Show first 4 suggestions
        keyboard.append([InlineKeyboardButton(suggestion, callback_data=f"query:{suggestion}")])
    
    # Add "Type Own Question" button
    keyboard.append([InlineKeyboardButton("Type Own Question", callback_data="type_own_question")])
    
    # Add a button to see more suggestions
    keyboard.append([InlineKeyboardButton("More suggestions", callback_data="more_suggestions")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await status_message.edit_text(
        "✅ Video processed successfully!\n\n"
        "Now you can ask questions about the video or request analysis.\n"
        "Select a suggestion below or type your own question:",
        reply_markup=reply_markup
    )

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle text messages from users."""
    text = update.message.text
    
    # Check if the text contains a video URL
    if is_video_url(text):
        # Extract the URL
        video_url = extract_video_url(text)
        if video_url:
            # Process the video URL
            await process_video_url(update, context, video_url)
            return
    
    # Check if we're waiting for a custom question
    if context.user_data.get('waiting_for_question'):
        # Reset the waiting flag
        context.user_data['waiting_for_question'] = False
        
        # Process the user's custom question
        await process_text(update.message, context)
    else:
        # Check if there's a video file or URL associated with this user
        if context.user_data.get('video_files', {}).get('current') or context.user_data.get('video_urls', {}).get('current'):
            # Process the text as a query about the current video
            await process_text(update.message, context)
        else:
            # No video file or URL, check if it might be a URL that wasn't detected
            if "http" in text.lower():
                await update.message.reply_text(
                    "I couldn't recognize a valid video URL in your message.\n"
                    "Please send a supported video URL (YouTube, Vimeo, etc.) or upload a video file."
                )
            else:
                # Prompt user to send a video
                await update.message.reply_text(
                    "Please send a video or a video URL first before asking questions."
                )

async def process_text(update: Update, context: ContextTypes.DEFAULT_TYPE, custom_query=None) -> None:
    """Process text messages as queries about the current video."""
    # Check if there's a video associated with this user
    video_type = context.user_data.get('video_type')
    
    if not video_type or (video_type == 'file' and not context.user_data.get('video_files', {}).get('current')) or (video_type == 'url' and not context.user_data.get('video_urls', {}).get('current')):
        await update.reply_text(
            "Please send a video first before asking questions."
        )
        return
    
    # Get the query from the user or use custom query if provided
    query = custom_query if custom_query else update.text
    
    # Notify user that processing has started
    status_message = await update.reply_text("🔍 Analyzing video with your query... This may take a moment.")
    
    try:
        if video_type == 'file':
            # Get the video file reference
            video_file = genai_client.files.get(name=context.user_data['video_file_name'])
            
            # Generate content using Gemini API with file
            start_time = time.time()
            response = genai_client.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=[video_file, query]
            )
            analysis_time = time.time() - start_time
        else:  # video_type == 'url'
            # Get the video URL
            video_url = context.user_data['video_urls']['current']
            
            # Create a prompt that includes the video URL and the query
            combined_prompt = f"Video URL: {video_url}\n\nQuery: {query}\n\nPlease analyze the video at the URL and answer the query."
            
            # Generate content using Gemini API with URL in prompt
            start_time = time.time()
            response = genai_client.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=combined_prompt
            )
            analysis_time = time.time() - start_time
        
        # Translate the response to Vietnamese
        await status_message.edit_text("🔄 Translating response to Vietnamese...")
        
        original_text = response.text
        translated_text = translate_to_vietnamese(original_text)
        
        # Create a unique hash for this response
        response_hash = f"{hash(original_text) % 10000}"
        
        # Create buttons for additional actions
        keyboard = [
            [InlineKeyboardButton("Show Original (English)", callback_data=f"original:{response_hash}")],
            [InlineKeyboardButton("Ask another question", callback_data="more_suggestions")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Store the original text in context for retrieval
        if not context.user_data.get('original_responses'):
            context.user_data['original_responses'] = {}
        context.user_data['original_responses'][response_hash] = original_text
        
        # Send the translated response to the user
        await status_message.edit_text(
            f"📊 *Kết Quả Phân Tích* (Đã dịch sang tiếng Việt)\n\n"
            f"{translated_text}\n\n"
            f"⏱️ Thời gian phân tích: {analysis_time:.1f}s",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )
        
    except Exception as e:
        logger.error(f"Error analyzing video: {e}")
        await status_message.edit_text(f"❌ Error analyzing video: {str(e)}")

def format_size(size_bytes):
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes/1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes/(1024*1024):.1f} MB"
    else:
        return f"{size_bytes/(1024*1024*1024):.2f} GB"

def format_duration(seconds):
    """Format duration in minutes:seconds."""
    if seconds == "Unknown":
        return "Unknown"
    
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"

def main() -> None:
    """Start the bot."""
    # Create the Application
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("suggestions", suggestions_command))
    application.add_handler(CommandHandler("info", info_command))
    application.add_handler(CommandHandler("cleanup", cleanup_command))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_handler(MessageHandler(filters.VIDEO, process_video))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
