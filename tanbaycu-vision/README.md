# 🎥 tanbaycu Vision Bot

An AI-powered video analysis bot with multilingual support, cloud storage integration, and comprehensive video understanding capabilities.

## ✨ Features

### 🎬 Video Processing
- **Multi-Platform Support**: Direct video file upload or URL processing
- **Wide Format Compatibility**: MP4, MPEG, MOV, AVI, FLV, MPG, WEBM, WMV, 3GPP
- **Intelligent Size Handling**: Automatic optimization for different file sizes
- **Cloud Storage Integration**: Google Drive upload for large files (20-50MB)

### 🔗 URL Support
- **YouTube**: Full support for YouTube videos and shorts
- **Vimeo**: Complete Vimeo video analysis
- **Social Media**: Instagram, TikTok, Facebook video support
- **Google Drive**: Direct Google Drive video links
- **Direct URLs**: Support for direct video file URLs

### 🤖 AI-Powered Analysis
- **Content Summarization**: Comprehensive video content summaries
- **Visual Analysis**: Detailed analysis of visual elements and scenes
- **Audio Transcription**: Extract and transcribe audio content
- **Key Moment Identification**: Identify and timestamp important moments
- **Custom Queries**: Ask specific questions about video content

### 🌐 Multilingual Features
- **Vietnamese Translation**: Automatic translation of all analysis results
- **Bilingual Responses**: Option to view original English responses
- **Cultural Adaptation**: Context-aware responses for Vietnamese users
- **Language Preservation**: Technical terms preserved with explanations

### ☁️ Cloud Integration
- **Google Drive Storage**: Automatic upload for large videos
- **Automatic Cleanup**: Scheduled deletion of old videos to save space
- **Retention Management**: 7-day retention policy for uploaded videos
- **Storage Optimization**: Efficient file management and organization

## 🎯 Main Functions

### Video Upload & Processing
- **Direct Upload**: Send video files up to 50MB through Telegram
- **URL Processing**: Paste video URLs for direct analysis
- **Progress Tracking**: Real-time processing status updates
- **Error Handling**: Comprehensive error recovery and user guidance

### Analysis Capabilities
- **Content Summary**: Generate detailed video summaries
- **Scene Analysis**: Break down video into key scenes and moments
- **Audio Transcription**: Extract and transcribe spoken content
- **Visual Description**: Detailed description of visual elements
- **Quiz Generation**: Create educational quizzes based on video content
- **Topic Identification**: Identify and categorize main topics

### Interactive Features
- **Query Suggestions**: Pre-defined analysis options for quick access
- **Custom Questions**: Type your own specific questions about the video
- **Progressive Analysis**: Build on previous analysis with follow-up questions
- **Results History**: Access previous analysis results during the session

### Available Commands
- `/start` - Initialize bot and show welcome message
- `/help` - Comprehensive help and usage instructions
- `/suggestions` - Show available query suggestions
- `/info` - Bot information and version details
- `/cleanup` - Manually trigger cleanup of old videos

## 🔑 Required Environment Variables

The bot requires the following environment variables to be set:

```bash
TELEGRAM_TOKEN=your_telegram_bot_token
GEMINI_API_KEY=your_google_gemini_api_key
```

### Optional Configuration
- **Google Drive Integration**: Requires `googledrive-credentials.json` file for cloud storage
- **Drive Folder ID**: Configurable Google Drive folder for video storage

## 🛠️ Technical Stack

- **AI Model**: Google Generative AI (Gemini) for video analysis
- **Translation**: Google Translator via deep-translator
- **Telegram Framework**: python-telegram-bot with async support
- **Cloud Storage**: Google Drive API v3 with service account authentication
- **Video Processing**: Native video handling with format detection
- **URL Processing**: Multi-platform video URL extraction and processing

## 📏 File Size Handling

### Processing Tiers
- **Under 20MB**: Direct processing with Gemini AI
- **20-50MB**: Upload to Google Drive, then process
- **Over 50MB**: Telegram limit - requires URL or file compression

### Supported Formats
- **Video**: MP4, MPEG, MOV, AVI, FLV, MPG, WEBM, WMV, 3GPP
- **URLs**: YouTube, Vimeo, Instagram, TikTok, Facebook, Google Drive, direct links

## 🔄 Analysis Workflow

1. **Input**: Upload video file or send video URL
2. **Processing**: Automatic format detection and size optimization
3. **Storage**: Large files uploaded to Google Drive automatically
4. **Analysis**: AI-powered video content analysis
5. **Translation**: Automatic Vietnamese translation of results
6. **Interaction**: Follow-up questions and additional analysis
7. **Cleanup**: Automatic cleanup of temporary files and old videos

## 🌟 Key Innovations

1. **Intelligent Size Management**: Automatic handling of different file sizes with optimal processing
2. **Multi-Platform URL Support**: Comprehensive support for major video platforms
3. **Seamless Cloud Integration**: Transparent Google Drive integration for large files
4. **Automatic Vietnamese Localization**: Professional translation while preserving technical accuracy
5. **Progressive Analysis**: Build comprehensive understanding through iterative questioning
6. **Resource Optimization**: Efficient storage management with automatic cleanup

This bot provides enterprise-grade video analysis capabilities through a simple Telegram interface, making advanced AI video understanding accessible to Vietnamese users with full language support and cloud integration.

## 📞 Contact & Support

- **Personal Website**: [https://tanbaycu.is-a.dev](https://tanbaycu.is-a.dev)
- **Social Links**: [linktr.ee/tanbaycu](https://linktr.ee/tanbaycu)

For support, feature requests, or bug reports, please visit the links above.