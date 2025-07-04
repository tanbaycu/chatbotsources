# 🧠 tanbaycu Assistant Bot

An advanced AI-powered Telegram assistant bot featuring intelligent conversation, image analysis, and generation capabilities with smart memory management.

## ✨ Features

### 🤖 Core AI Capabilities
- **Multi-Model AI Integration**: Powered by Google Gemini 2.0 Flash Lite and G4F
- **Advanced Memory System**: Intelligent short-term and long-term memory with TF-IDF vectorization
- **Context-Aware Responses**: Smart conversation context analysis with cosine similarity
- **Multi-Personality Modes**: Adaptive personality based on conversation topic and sentiment

### 🖼️ Image Capabilities
- **Image Analysis**: Advanced image understanding and description
- **Image Generation**: AI-powered image creation with multiple models (Flux, SDXL, etc.)
- **Image Optimization**: Automatic quality optimization for efficient processing
- **Cache System**: Smart caching for improved performance

### 🧠 Intelligence Features
- **Smart Prompt Generation**: Dynamic prompt optimization based on conversation context
- **Sentiment Detection**: Automatic emotion and tone analysis
- **Topic Classification**: Intelligent categorization of conversation subjects
- **Search Integration**: Advanced web search capabilities with result synthesis

### 🔧 Technical Features
- **Dual API Key Management**: Automatic failover between primary and backup API keys
- **Rate Limiting Protection**: Smart request throttling and error handling
- **Async Processing**: High-performance asynchronous operations
- **Memory Persistence**: Automatic conversation history saving and loading

## 🎯 Main Functions

### Chat & Conversation
- Natural language conversation with context awareness
- Multiple personality modes (professional, casual, educational)
- Smart memory system that remembers important information
- Conversation stage detection and appropriate responses

### Image Processing
- Upload images for AI analysis and description
- Generate images from text descriptions
- Support for multiple AI image generation models
- Real-time processing progress updates

### Search & Research
- Web search with intelligent result compilation
- Context-aware search query optimization
- Comprehensive answer synthesis from multiple sources

### Utility Commands
- `/start` - Initialize bot and show welcome message
- `/info` - Display bot capabilities and statistics
- `/clear` - Clear conversation memory
- `/search <query>` - Perform web search
- `/image <prompt>` - Generate images with AI
- `/imagehelp` - Show image generation help
- `/imagestats` - Display image generation statistics
- `/update` - Show version history and updates

## 🔑 Required Environment Variables

The bot requires the following environment variables to be set:

```bash
BOT_TOKEN_NE=your_telegram_bot_token
GEMINI_KEY=your_primary_gemini_api_key
GEMINI_KEY_BACKUP=your_backup_gemini_api_key
```

## 🛠️ Technical Stack

- **AI Models**: Google Gemini 2.0 Flash Lite, G4F
- **Machine Learning**: scikit-learn (TF-IDF, cosine similarity)
- **Image Processing**: Pillow (PIL)
- **Async Framework**: aiohttp, asyncio
- **Telegram API**: pyTelegramBotAPI
- **Data Processing**: numpy

## 🚀 Key Innovations

1. **Advanced Memory Management**: Combines short-term and long-term memory with semantic similarity search
2. **Dynamic Personality Adaptation**: Automatically adjusts response style based on conversation context
3. **Intelligent Error Handling**: Robust API key rotation and rate limit management
4. **Multi-Modal Processing**: Seamless integration of text and image processing capabilities
5. **Performance Optimization**: Efficient caching, async processing, and resource management

This bot represents a sophisticated AI assistant capable of maintaining meaningful, context-aware conversations while providing advanced image processing and generation capabilities.

## 📞 Contact & Support

- **Personal Website**: [https://tanbaycu.is-a.dev](https://tanbaycu.is-a.dev)
- **Social Links**: [linktr.ee/tanbaycu](https://linktr.ee/tanbaycu)

For support, feature requests, or bug reports, please visit the links above.