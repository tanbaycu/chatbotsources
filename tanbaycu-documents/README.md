# 📄 tanbaycu Document Analyzer Bot

An intelligent Telegram bot for PDF document analysis and comparison with multilingual support and advanced AI-powered insights.

## ✨ Features

### 📚 Document Processing
- **PDF Upload & Analysis**: Direct PDF file upload with intelligent content extraction
- **Multi-Document Comparison**: Compare multiple PDFs side-by-side with detailed analysis
- **Smart Document Management**: Organized file storage with easy access and management
- **Large File Support**: Handles complex PDF documents with efficient processing

### 🤖 AI-Powered Analysis
- **Intelligent Q&A**: Ask specific questions about document content
- **Content Summarization**: Generate comprehensive summaries of PDF documents
- **Key Information Extraction**: Identify and highlight important information
- **Context-Aware Responses**: Maintain conversation context across multiple queries

### 🌐 Multilingual Support
- **Vietnamese Translation**: Automatic translation of analysis results to Vietnamese
- **Bilingual Interface**: Support for both English and Vietnamese user interactions
- **Language Selection**: User-configurable language preferences
- **Cultural Context Adaptation**: Responses adapted to Vietnamese cultural context

### 💬 Interactive Conversation
- **Conversation State Management**: Sophisticated state handling for smooth user experience
- **Menu-Driven Interface**: Intuitive navigation with inline keyboards
- **Progress Tracking**: Real-time updates during document processing
- **Error Recovery**: Robust error handling with helpful user guidance

## 🎯 Main Functions

### Document Operations
- **Upload Documents**: Send PDF files directly through Telegram
- **Analyze Content**: Get comprehensive analysis of document content
- **Compare Documents**: Side-by-side comparison of multiple PDFs
- **Ask Questions**: Interactive Q&A about specific document content

### Analysis Types
- **Content Summary**: Generate detailed summaries of document content
- **Key Points Extraction**: Identify and list main points and conclusions
- **Topic Analysis**: Categorize and analyze document topics
- **Custom Queries**: Ask specific questions about document content

### Management Features
- **File Organization**: Manage multiple uploaded documents
- **Session Persistence**: Maintain document access across conversations
- **Language Switching**: Change interface language at any time
- **Message Cleanup**: Automatic cleanup of old messages for better performance

### Available Commands
- `/start` - Initialize bot and show main menu
- `/menu` - Display main navigation menu
- `/language` - Change interface language
- `/help` - Show detailed help and instructions
- `/back` - Return to previous menu
- `/cancel` - Cancel current operation

## 🔑 Required Environment Variables

The bot requires the following environment variables to be set:

```bash
TELEGRAM_TOKEN=your_telegram_bot_token
GEMINI_API_KEY=your_google_gemini_api_key
```

## 🛠️ Technical Stack

- **AI Model**: Google Generative AI (Gemini 1.5 Flash)
- **Translation**: Google Translator via deep-translator
- **Telegram Framework**: python-telegram-bot with ConversationHandler
- **HTTP Client**: httpx for efficient async operations
- **Document Processing**: Native PDF handling with AI analysis

## 🔄 Conversation Flow

1. **Start**: User initiates conversation with `/start`
2. **Language Selection**: Choose preferred language (English/Vietnamese)
3. **Main Menu**: Access to upload, analyze, or compare documents
4. **Document Upload**: Send PDF files for processing
5. **Analysis Options**: Choose from predefined analysis types or custom queries
6. **Interactive Q&A**: Ask follow-up questions about the documents
7. **Results**: Receive comprehensive analysis with Vietnamese translation

## 🌟 Key Innovations

1. **Intelligent Document Understanding**: Advanced AI comprehension of PDF content
2. **Seamless Multilingual Experience**: Automatic translation preserving technical accuracy
3. **State-Aware Conversations**: Sophisticated conversation flow management
4. **Efficient Resource Management**: Optimized file handling and processing
5. **User-Centric Design**: Intuitive interface with helpful guidance and error recovery

This bot provides a professional-grade document analysis solution that makes complex PDF analysis accessible through a simple Telegram interface, with full Vietnamese language support for local users.

## 📞 Contact & Support

- **Personal Website**: [https://tanbaycu.is-a.dev](https://tanbaycu.is-a.dev)
- **Social Links**: [linktr.ee/tanbaycu](https://linktr.ee/tanbaycu)

For support, feature requests, or bug reports, please visit the links above.