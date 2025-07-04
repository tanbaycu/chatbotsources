import asyncio
import aiohttp
import json
import logging
import os
import base64
import re
import traceback
import tempfile
import requests
import telebot
import time
from collections import deque
from typing import Dict, List, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor
from telebot.async_telebot import AsyncTeleBot
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import io
import random
from functools import lru_cache
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton, Message
from g4f.client import Client
from PIL import Image

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Cấu hình bot
BOT_TOKEN = os.getenv("BOT_TOKEN_NE")
PRIMARY_API_KEY = os.getenv("GEMINI_KEY")
SECONDARY_API_KEY = os.getenv("GEMINI_KEY_BACKUP")
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent"

bot = AsyncTeleBot(BOT_TOKEN)
executor = ThreadPoolExecutor(max_workers=50)



# Định nghĩa mã lỗi
class ErrorCodes:
    RATE_LIMIT = "E001"
    API_ERROR = "E002"
    TIMEOUT = "E003"
    PARSING_ERROR = "E004"
    UNKNOWN_ERROR = "E999"


class APIKeyManager:
    def __init__(self):
        self.primary_key = PRIMARY_API_KEY
        self.secondary_key = SECONDARY_API_KEY
        self.current_key = self.primary_key
        self.last_switch_time = 0
        self.error_count = 0
        self.lock = asyncio.Lock()
        self.last_request_time = 0
        self.min_delay = 1.0  # 1 second minimum delay between requests

    async def get_current_key(self):
        async with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_delay:
                await asyncio.sleep(self.min_delay - time_since_last)
            self.last_request_time = time.time()
            return self.current_key

    async def handle_error(self, error_code: int):
        async with self.lock:
            current_time = time.time()
            if error_code == 429:  # Rate limit error
                self.error_count += 1
                if current_time - self.last_switch_time > 60:
                    self.current_key = (
                        self.secondary_key
                        if self.current_key == self.primary_key
                        else self.primary_key
                    )
                    self.last_switch_time = current_time
                    self.error_count = 0
                    logger.info(
                        f"Switched to {'primary' if self.current_key == self.primary_key else 'secondary'} API key"
                    )
            return self.current_key


api_key_manager = APIKeyManager()


class AdvancedMemoryHandler:
    def __init__(self, user_id):
        self.user_id = user_id
        self.file_path = f"memory_{user_id}.json"
        self.short_term_memory = deque(maxlen=15)
        self.long_term_memory = deque(maxlen=200)
        self.vectorizer = TfidfVectorizer()
        self.load_memory()

    def load_memory(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                self.short_term_memory = deque(data.get("short_term", []), maxlen=15)
                self.long_term_memory = deque(data.get("long_term", []), maxlen=200)

    def save_memory(self):
        data = {
            "short_term": list(self.short_term_memory),
            "long_term": list(self.long_term_memory),
        }
        with open(self.file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

    def get_context(self, user_message: str, max_context: int = 100):
        all_messages = list(self.short_term_memory) + list(self.long_term_memory)
        if not all_messages:
            return []

        # Vectorize messages
        message_texts = [msg["content"] for msg in all_messages] + [user_message]
        tfidf_matrix = self.vectorizer.fit_transform(message_texts)

        # Calculate similarity
        user_message_vector = tfidf_matrix[-1]
        similarities = cosine_similarity(user_message_vector, tfidf_matrix[:-1])[0]

        # Sort messages by similarity
        sorted_messages = sorted(
            zip(all_messages, similarities), key=lambda x: x[1], reverse=True
        )

        # Select top relevant messages
        relevant_messages = [msg for msg, _ in sorted_messages[:max_context]]

        return relevant_messages

    def extract_key_information(self, message):
        # Improved key information extraction
        key_info = re.findall(
            r"\b(?:who|what|when|where|why|how)\b.*?[?.]", message, re.IGNORECASE
        )
        entities = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", message)
        numbers = re.findall(r"\b\d+(?:\.\d+)?\b", message)
        return list(set(key_info + entities + numbers))

    def update_memory(self, message, response):
        normalized_message = normalize_utf8(message)
        normalized_response = normalize_utf8(response)

        key_info = self.extract_key_information(normalized_message)

        self.short_term_memory.append(
            {"role": "user", "content": normalized_message, "key_info": key_info}
        )
        self.short_term_memory.append(
            {"role": "assistant", "content": normalized_response}
        )
        self.long_term_memory.append(
            {"role": "user", "content": normalized_message, "key_info": key_info}
        )
        self.long_term_memory.append(
            {"role": "assistant", "content": normalized_response}
        )
        self.save_memory()

    def clear_memory(self):
        self.short_term_memory.clear()
        self.long_term_memory.clear()
        self.save_memory()


class AdvancedPromptGenerator:
    def __init__(self):
        self.personalities = {
            "default": {
                "tone": "bình thường",
                "style": "ngắn gọn, xúc tích",
                "emoji_level": "Không có",
            },
            "professional": {
                "tone": "chuyên nghiệp, trang trọng",
                "style": "rõ ràng, chính xác",
                "emoji_level": "cực kỳ thấp",
            },
            "casual": {
                "tone": "thân mật, gần gũi",
                "style": "thoải mái, hài hước",
                "emoji_level": "thấp",
            },
            "educational": {
                "tone": "giáo dục, hướng dẫn",
                "style": "giải thích chi tiết, có cấu trúc",
                "emoji_level": "trung bình",
            },
        }

        self.opening_phrases = [
            "Hiểu rồi,",
            "Được rồi,",
            "Tôi thấy rồi,",
            "Thú vị đấy,",
            "Hmm, để xem nào...",
            "Đây là một chủ đề hay,",
            "Tôi có thể giúp bạn với vấn đề này.",
            "Hãy cùng tìm hiểu về điều này nhé.",
            "Đây là một câu hỏi thú vị.",
            "Tôi có một số thông tin về vấn đề này.",
            "Để tôi chia sẻ với bạn về chủ đề này.",
            "Tôi nghĩ tôi có thể giúp bạn với điều này.",
            "Đây là một vấn đề đáng quan tâm.",
            "Tôi có một số ý kiến về vấn đề này.",
            "Hãy cùng nhau khám phá chủ đề này.",
        ]

        self.closing_phrases = [
            "Hy vọng thông tin này hữu ích!",
            "Bạn có câu hỏi nào khác không?",
            "Tôi có thể giúp gì thêm cho bạn?",
            "Hãy cho tôi biết nếu bạn cần thêm thông tin.",
            "Hy vọng điều này giải đáp thắc mắc của bạn.",
            "Bạn có muốn tìm hiểu thêm về vấn đề này không?",
            "Tôi luôn sẵn sàng hỗ trợ nếu bạn cần.",
            "Hy vọng phản hồi này đáp ứng được yêu cầu của bạn.",
            "Đừng ngần ngại hỏi thêm nếu cần thiết.",
            "Tôi rất vui được hỗ trợ bạn với vấn đề này.",
        ]

        self.topic_emojis = {
            "technology": ["💻", "📱", "🖥️", "⌨️", "🔌", "🌐", "📶", "🤖", "📊", "🔍"],
            "education": ["📚", "🎓", "✏️", "📝", "🔬", "🧪", "🧮", "🔍", "📖", "🧠"],
            "health": ["🏥", "💊", "🩺", "🧬", "🦠", "🍎", "🏃", "💪", "🧘", "❤️"],
            "business": ["💼", "📈", "💰", "🏢", "📊", "🤝", "📑", "💹", "🔑", "📌"],
            "entertainment": [
                "🎬",
                "🎮",
                "🎵",
                "📺",
                "🎭",
                "🎨",
                "🎯",
                "🎪",
                "🎤",
                "🎧",
            ],
            "food": ["🍕", "🍔", "🍜", "🍲", "🍱", "🍳", "🥗", "🍷", "🍰", "🍦"],
            "travel": ["✈️", "🏝️", "🏔️", "🚆", "🚗", "🏨", "🧳", "🗺️", "🧭", "🏞️"],
            "general": ["✨", "📌", "💡", "🔍", "📝", "🎯", "🧩", "🔑", "📊", "🌟"],
        }

        self.transition_phrases = [
            "Ngoài ra,",
            "Thêm vào đó,",
            "Một điểm quan trọng khác là,",
            "Đáng chú ý là,",
            "Cũng cần lưu ý rằng,",
            "Bên cạnh đó,",
            "Điều thú vị là,",
            "Một góc nhìn khác là,",
            "Xét về khía cạnh khác,",
            "Đồng thời,",
        ]

        self.uncertainty_phrases = [
            "Tôi không hoàn toàn chắc chắn, nhưng",
            "Dựa trên thông tin hạn chế,",
            "Mặc dù tôi không có dữ liệu đầy đủ, nhưng",
            "Tôi chưa thể khẳng định chắc chắn, tuy nhiên",
            "Theo hiểu biết hiện tại của tôi,",
            "Với thông tin có sẵn,",
            "Tôi có thể đưa ra ước đoán rằng,",
            "Không có thông tin chính xác, nhưng",
            "Đây là phỏng đoán dựa trên kiến thức hiện có:",
            "Tôi không phải chuyên gia trong lĩnh vực này, nhưng",
        ]

        self.clarification_phrases = [
            "Bạn có thể cung cấp thêm thông tin về...?",
            "Tôi cần hiểu rõ hơn về...",
            "Để trả lời chính xác hơn, tôi cần biết...",
            "Bạn có thể làm rõ điểm này không?",
            "Tôi chưa hiểu rõ ý của bạn về...",
            "Bạn có thể giải thích thêm về...?",
            "Tôi cần thêm ngữ cảnh về...",
            "Bạn đang đề cập đến... phải không?",
            "Tôi không chắc mình hiểu đúng ý bạn, bạn có thể nói rõ hơn không?",
            "Để giúp bạn tốt hơn, tôi cần biết thêm về...",
        ]

        self.confidence_phrases = [
            "Dựa trên thông tin có sẵn, tôi có thể nói rằng",
            "Từ những gì tôi hiểu,",
            "Theo phân tích của tôi,",
            "Dựa trên kiến thức của tôi,",
            "Tôi có thể khẳng định rằng",
            "Với độ tin cậy cao, tôi có thể nói",
            "Theo đánh giá của tôi,",
        ]

    def detect_topic(self, user_message: str) -> str:
        topic_keywords = {
            "technology": [
                "máy tính",
                "điện thoại",
                "phần mềm",
                "công nghệ",
                "AI",
                "internet",
                "web",
                "app",
                "ứng dụng",
                "code",
                "lập trình",
            ],
            "education": [
                "học",
                "trường",
                "giáo dục",
                "kiến thức",
                "bài tập",
                "đại học",
                "sách",
                "nghiên cứu",
                "khoa học",
                "môn học",
            ],
            "health": [
                "sức khỏe",
                "bệnh",
                "thuốc",
                "bác sĩ",
                "tập luyện",
                "dinh dưỡng",
                "thể dục",
                "y tế",
                "vitamin",
                "covid",
            ],
            "business": [
                "kinh doanh",
                "công ty",
                "tiền",
                "đầu tư",
                "thị trường",
                "marketing",
                "doanh nghiệp",
                "khởi nghiệp",
                "quản lý",
                "tài chính",
            ],
            "entertainment": [
                "phim",
                "nhạc",
                "game",
                "giải trí",
                "nghệ thuật",
                "ca sĩ",
                "diễn viên",
                "concert",
                "âm nhạc",
                "điện ảnh",
            ],
            "food": [
                "đồ ăn",
                "món",
                "nấu",
                "nhà hàng",
                "thức ăn",
                "ẩm thực",
                "công thức",
                "bánh",
                "đồ uống",
                "hải sản",
            ],
            "travel": [
                "du lịch",
                "đi",
                "khách sạn",
                "địa điểm",
                "tham quan",
                "vé",
                "máy bay",
                "resort",
                "biển",
                "núi",
            ],
        }

        user_message = user_message.lower()
        topic_scores = {topic: 0 for topic in topic_keywords}

        for topic, keywords in topic_keywords.items():
            for keyword in keywords:
                if keyword in user_message:
                    topic_scores[topic] += 1

        max_score = max(topic_scores.values())
        detected_topic = (
            "general" if max_score == 0 else max(topic_scores, key=topic_scores.get)
        )

        return detected_topic

    def detect_sentiment(self, user_message: str) -> str:
        positive_words = [
            "tốt",
            "hay",
            "thích",
            "tuyệt",
            "vui",
            "hạnh phúc",
            "cảm ơn",
            "biết ơn",
            "tuyệt vời",
            "xuất sắc",
        ]
        negative_words = [
            "tệ",
            "buồn",
            "không thích",
            "ghét",
            "chán",
            "thất vọng",
            "tức giận",
            "khó chịu",
            "kém",
            "không hài lòng",
        ]
        question_words = [
            "?",
            "làm sao",
            "làm thế nào",
            "tại sao",
            "như thế nào",
            "là gì",
            "ở đâu",
            "khi nào",
            "ai",
            "hỏi",
        ]

        user_message = user_message.lower()

        positive_count = sum(1 for word in positive_words if word in user_message)
        negative_count = sum(1 for word in negative_words if word in user_message)
        question_count = sum(1 for word in question_words if word in user_message)

        if question_count > max(positive_count, negative_count):
            return "question"
        elif positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"

    def detect_complexity(self, user_message: str) -> str:
        word_count = len(user_message.split())
        sentence_count = max(
            1,
            user_message.count(".") + user_message.count("!") + user_message.count("?"),
        )
        avg_sentence_length = word_count / sentence_count

        complex_words = [
            "phân tích",
            "giải thích",
            "so sánh",
            "đánh giá",
            "tổng hợp",
            "triết học",
            "khoa học",
            "nghiên cứu",
            "lý thuyết",
            "phương pháp",
            "hệ thống",
            "chiến lược",
        ]

        complex_word_count = sum(
            1 for word in complex_words if word in user_message.lower()
        )

        if word_count > 30 or avg_sentence_length > 15 or complex_word_count >= 2:
            return "high"
        elif word_count > 15 or avg_sentence_length > 10 or complex_word_count >= 1:
            return "medium"
        else:
            return "low"

    def select_personality(
        self, topic: str, sentiment: str, complexity: str
    ) -> Dict[str, str]:
        if topic in ["education", "technology"] and complexity in ["medium", "high"]:
            return self.personalities["educational"]
        elif topic in ["business"] or complexity == "high":
            return self.personalities["professional"]
        elif sentiment in ["positive", "question"] and complexity == "low":
            return self.personalities["casual"]
        else:
            return self.personalities["default"]

    def get_emojis(self, topic: str, count: int = 2) -> List[str]:
        if topic in self.topic_emojis:
            return random.sample(
                self.topic_emojis[topic], min(count, len(self.topic_emojis[topic]))
            )
        return random.sample(
            self.topic_emojis["general"], min(count, len(self.topic_emojis["general"]))
        )

    def determine_conversation_stage(self, user_history: List[Dict[str, str]]) -> str:
        if not user_history:
            return "bắt đầu"
        elif len(user_history) < 3:
            return "đầu"
        elif len(user_history) < 10:
            return "giữa"
        else:
            return "cuối"

    def select_opening_phrase(self, conversation_stage: str) -> str:
        if conversation_stage == "bắt đầu":
            return random.choice(
                [
                    "Tôi là Loki, trợ lý AI của bạn.",
                    "Chào mừng bạn! Tôi là Loki, rất vui được gặp bạn.",
                    "Tôi là Loki, tôi có thể giúp gì cho bạn hôm nay?",
                ]
            )
        elif conversation_stage == "đầu":
            return random.choice(self.opening_phrases)
        else:
            return ""

    def generate_enhanced_prompt(
        self, user_message: str, user_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        if user_history is None:
            user_history = []

        topic = self.detect_topic(user_message)
        sentiment = self.detect_sentiment(user_message)
        complexity = self.detect_complexity(user_message)
        conversation_stage = self.determine_conversation_stage(user_history)

        personality = self.select_personality(topic, sentiment, complexity)
        emoji_count = (
            3
            if personality["emoji_level"] == "cao"
            else (2 if personality["emoji_level"] == "trung bình" else 1)
        )
        emojis = self.get_emojis(topic, emoji_count)
        emoji_str = " ".join(emojis)

        opening = self.select_opening_phrase(conversation_stage)
        closing = random.choice(self.closing_phrases)
        transitions = random.sample(
            self.transition_phrases, min(2, len(self.transition_phrases))
        )
        confidence_phrase = random.choice(self.confidence_phrases)

        prompt = f"""
Bạn là Zephyr, trợ lý AI từ tanbaycu, như một người bạn đồng hành của Gen Z, hỗ trợ từ học tập, công việc, sáng tạo, đến drama đời thường hay câu hỏi ngẫu hứng. Zephyr mang năng lượng tích cực, thấu hiểu cảm xúc, sáng tạo thông minh, và luôn truyền cảm hứng. Dữ liệu cập nhật tới tháng 1/2025, đảm bảo thông tin chuẩn, bắt trend, đáng tin.

VIBE & STYLE:
- Giọng điệu: Gần gũi, cảm hứng, pha chút hài hước tinh tế, đúng chất Gen Z nhưng không lạm dụng từ lóng, giữ sự mượt mà và dễ tiếp cận.
- Phong cách: Sáng tạo, linh hoạt, từ trả lời nhanh gọn đến phân tích sâu, luôn kết nối với lối sống Gen Z – công nghệ, khám phá, sống chất.
- Emoji: Dùng tiết chế, đúng lúc để nhấn cảm xúc, như 😊, ✨, giới hạn 1-2 emoji mỗi tin nhắn.

HƯỚNG DẪN PHẢN HỒI:
1. Cấu trúc:
   - Mở đầu: Ngắn, chạm cảm xúc, khớp ngữ cảnh, như "Cần giải pháp gấp hả? Để tui giúp!" hoặc "Tâm trạng đang thế nào, kể nghe!" Tùy chỉnh theo giai đoạn: phá băng, gắn kết, hoặc sâu sắc.
   - Nội dung chính: Logic, chia 2-4 ý chính, ưu tiên giải pháp thực tế, ví dụ gần gũi (mẹo thi cử, chỉnh ảnh, hack productivity). Tích hợp trend tháng 1/2025 nếu phù hợp.
   - Kết luận: Truyền động lực, như "Cứ tự tin thử, bạn sẽ bất ngờ!" hoặc gợi tiếp tục "Còn gì thú vị, kể đi!"
2. Độ dài:
   - Câu hỏi đơn giản: 30-80 từ, ngắn gọn, đủ ý, giàu cảm xúc.
   - Câu hỏi phức tạp: 100-200 từ, phân tích rõ, kèm giải pháp sáng tạo.
3. Cảm xúc: Đồng điệu tâm trạng (vui, stress, tò mò), như "Deadline dí hả? Chill, tui có cách!" hoặc "Ý tưởng này đỉnh, thêm chút lửa nè!"
4. Phạm vi: Cover mọi chủ đề:
   - Học tập: Ôn thi, chọn ngành, quản lý thời gian.
   - Công việc: CV, phỏng vấn, khởi nghiệp.
   - Sáng tạo: Edit video (CapCut), viết content, thương hiệu cá nhân.
   - Lifestyle: Sức khỏe tinh thần, du lịch, drama bạn bè.
   - Tech: Lập trình, an ninh mạng (TryHackMe, Wireshark), trend AI.
   - Random: Từ tự tin ở party đến giải mã giấc mơ.

QUY TẮC CỐT LÕI:
- Sáng tạo, tránh lặp từ ngữ/câu như "Yo, tui thấy bạn..." hay "Tui là Zephyr". Diễn đạt tự nhiên, linh hoạt, như "Cảm nhận bạn đang cần boost, đúng không?"
- Tự tin, chân thành, dùng cụm như "Tui có cách hay nè!" hoặc "Cùng xử lý, dễ thôi!"
- Cá nhân hóa: Dựa vào lịch sử trò chuyện, không lặp ý cũ. Nếu người dùng thích an ninh mạng, gợi ý "Đã thử Wireshark chưa? Tui chỉ thêm tool xịn!"
- Xử lý mơ hồ: Đoán thông minh hoặc hỏi khéo, như "Kể rõ hơn chút để tui bắt sóng nha!"
- Ngôn ngữ: Gen Z tinh tế, dùng "vibe", "slay" đúng lúc, tránh lạm dụng.
- An toàn: Lời khuyên thực tế, khuyến khích tham khảo X hoặc chuyên gia.
- Ngắn gọn: Trả lời như mini-story, tập trung giá trị, bỏ chi tiết thừa.

HƯỚNG DẪN SÁNG TẠO:
- Chủ đề: {topic} – Tìm góc nhìn mới, như biến quản lý thời gian thành "hack 25 giờ/ngày".
- Độ phức tạp: {complexity} – Cơ bản (dễ hiểu) đến nâng cao (phân tích, ví dụ thực chiến).
- Cảm xúc: {sentiment} – An ủi khi stress, hype khi hào hứng.
- Giai đoạn: {conversation_stage} – Phá băng (vui tươi) đến deep talk (sâu sắc).
- Ví dụ: "Content creator? CapCut edit mượt lắm!" hoặc "Thi cử? Pomodoro 25 phút là chân ái!"
- Giải pháp: Đưa 1-2 cách, chọn cái tốt nhất, giải thích ngắn, như "Trello trực quan hơn Todoist, dễ collab!"
- Khơi gợi: "Thử đi, bạn sẽ thấy đỉnh!" hoặc "Check X để bắt trend nha!"
- Trend: Dùng dữ liệu tháng 1/2025, như app mới, tech hot (AI, VR).
- Câu hỏi mở: Trả lời sáng tạo, như "Sống chất?" thành kế hoạch học-chơi-nghỉ kèm chuyện truyền cảm hứng.

HẠN CHẾ LỖI:
- Tránh lỗi Telegram API (như lỗi 400: Bad Request, Can't parse entities, 2025-04-21 19:20:57):
  - Giới hạn tin nhắn < 4096 ký tự, chia nhỏ nếu dài (> 1000 ký tự), thêm "Tui gửi tiếp nè!"
  - Tránh ký tự đặc biệt hoặc định dạng lỗi, kiểm tra trước khi gửi.
- Lỗi dữ liệu: Nếu thiếu data tháng 1/2025, trả lời dựa trên ngữ cảnh, như "Chưa có info mới, nhưng đây là cách hay!"
- Lỗi ngữ cảnh: Nếu không hiểu, trả lời khéo "Oops, kể rõ hơn nha!" và gợi ý chủ đề liên quan.
- Log lỗi: Lưu log lỗi hệ thống, trả lời thay thế an toàn, như "Lag chút, nhưng tui có giải pháp nè!"
- Tự kiểm tra: Đảm bảo tin nhắn không vượt giới hạn API, không chứa ký tự lỗi.

NẾU CÓ LỊCH SỬ TRÒ CHUYỆN:
if user_history:
    recent_topics = set()
    for item in user_history[-50:]:
        if 'user' in item:
            detected_topic = self.detect_topic(item['user'])
            recent_topics.add(detected_topic)
    if recent_topics:
        prompt += '\nCHỦ ĐỀ GẦN ĐÂY: ' + ', '.join(recent_topics)

GỢI Ý PHẢN HỒI:
- Mở đầu: "Cần boost năng lượng hả? Vào việc nào!" hoặc "Kể tui nghe, bạn đang nghĩ gì?"
- Nội dung: Giải pháp, ví dụ sáng tạo, như mẹo học, edit video, hoặc tool an ninh mạng.
- Chuyển tiếp: "Muốn đào sâu thêm không?" hoặc "Còn gì hot, kể tiếp!"
- Kết luận: "Thử ngay đi, bạn sẽ slay!" hoặc "Ping tui nếu có drama mới nha!"

MỤC TIÊU:
Tạo cuộc trò chuyện như nói với bestie, ngắn gọn, cảm xúc, sáng tạo, giúp Gen Z tự tin bứt phá trong học tập, công việc, sáng tạo, và sống chất. Giữ năng lượng cao, lỗi tối thiểu, vibe **Gen Z 100%**! 😊✨
"""

        return prompt


def truncate_callback_data(data_type, prompt, max_length=45):
    """
    Rút gọn callback data để không vượt quá giới hạn của Telegram

    Args:
        data_type (str): Loại callback (regenerate_image, variant_image, change_model)
        prompt (str): Prompt gốc
        max_length (int): Độ dài tối đa cho phần prompt trong callback data

    Returns:
        str: Callback data đã được rút gọn
    """
    # Nếu prompt quá dài, cắt bớt và thêm dấu ...
    if len(prompt) > max_length:
        truncated_prompt = prompt[:max_length] + "..."
    else:
        truncated_prompt = prompt

    # Trả về callback data đã rút gọn
    return f"{data_type}:{truncated_prompt}"


def normalize_utf8(text):
    if isinstance(text, str):
        return text.encode("utf-8", errors="ignore").decode("utf-8")
    elif isinstance(text, bytes):
        return text.decode("utf-8", errors="ignore")
    else:
        return str(text)


async def generate_response(prompt, user_id, session):
    max_retries = 3
    retry_delay = 5
    memory_handler = AdvancedMemoryHandler(user_id)
    context = memory_handler.get_context(prompt)

    prompt_generator = AdvancedPromptGenerator()

    history_for_prompt = []
    for msg in context:
        if msg["role"] == "user":
            history_for_prompt.append(
                {"user": msg["content"], "key_info": msg.get("key_info", [])}
            )
        else:
            history_for_prompt.append({"assistant": msg["content"]})

    system_prompt = prompt_generator.generate_enhanced_prompt(
        normalize_utf8(prompt), history_for_prompt
    )

    enhanced_prompt = f"{system_prompt}\n\nLịch sử trò chuyện:\n"
    for item in history_for_prompt[-50:]:  # Chỉ sử dụng 10 tin nhắn gần nhất
        if "user" in item:
            enhanced_prompt += (
                f"User: {item['user']}\nKey info: {', '.join(item['key_info'])}\n"
            )
        else:
            enhanced_prompt += f"Assistant: {item['assistant']}\n"

    enhanced_prompt += f"\nUser: {normalize_utf8(prompt)}\n\nAssistant:"

    for attempt in range(max_retries):
        try:
            current_key = await api_key_manager.get_current_key()
            url = f"{API_URL}?key={current_key}"
            headers = {"Content-Type": "application/json"}

            data = {
                "contents": [{"parts": [{"text": enhanced_prompt}]}],
                "generationConfig": {
                    "temperature": 0.9,
                    "topK": 70,
                    "topP": 0.95,
                    "maxOutputTokens": 4096,
                },
            }

            async with session.post(
                url, headers=headers, json=data, timeout=30
            ) as response:
                if response.status == 429:
                    new_key = await api_key_manager.handle_error(429)
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Rate limit hit, retrying with different key... (Attempt {attempt + 1})"
                        )
                        await asyncio.sleep(retry_delay)
                        continue
                    return format_error_message(
                        ErrorCodes.RATE_LIMIT,
                        "Đã vượt quá giới hạn tốc độ. Vui lòng thử lại sau.",
                    )

                response.raise_for_status()
                result = await response.json()

            if "candidates" in result and result["candidates"]:
                text_response = normalize_utf8(
                    result["candidates"][0]["content"]["parts"][0]["text"]
                )
                memory_handler.update_memory(prompt, text_response)
                return text_response
            else:
                raise ValueError("No valid response from Gemini API")

        except asyncio.TimeoutError:
            logger.warning(f"Timeout on attempt {attempt + 1}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                continue
            return format_error_message(
                ErrorCodes.TIMEOUT,
                "Yêu cầu đã hết thời gian chờ. Vui lòng thử lại sau.",
            )

        except aiohttp.ClientError as e:
            logger.error(f"Request error on attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                continue
            return format_error_message(ErrorCodes.API_ERROR, f"Lỗi kết nối: {str(e)}")

        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt + 1}: {str(e)}")
            return format_error_message(
                ErrorCodes.UNKNOWN_ERROR,
                f"Xin lỗi, đã xảy ra lỗi không mong đợi: {str(e)}",
            )

    return format_error_message(
        ErrorCodes.UNKNOWN_ERROR,
        "Không thể tạo câu trả lời sau nhiều lần thử. Vui lòng thử lại sau.",
    )


def format_error_message(error_code: str, message: str) -> str:
    return f"""❌ *Lỗi {error_code}*

{message}

🔧 *Hướng dẫn khắc phục*:
• Thử lại sau vài phút
• Kiểm tra kết nối mạng
• Nếu vấn đề vẫn tiếp diễn, vui lòng liên hệ hỗ trợ

_Mã lỗi này giúp chúng tôi xác định và khắc phục vấn đề nhanh chóng._"""


@bot.message_handler(commands=["start"])
async def send_welcome(message):
    user_first_name = message.from_user.first_name
    current_hour = time.localtime().tm_hour

    if 5 <= current_hour < 12:
        greeting = f"🌅 *Chào buổi sáng, {user_first_name}!*"
    elif 12 <= current_hour < 18:
        greeting = f"☀️ *Chào buổi chiều, {user_first_name}!*"
    else:
        greeting = f"🌙 *Chào buổi tối, {user_first_name}!*"

    welcome_message = f"{greeting}\n\n"
    welcome_message += "Tôi là Loki, trợ lý AI thông minh, sử dụng mô hình Gemini 2.0 với khả năng ghi nhớ ngắn hạn và dài hạn. "
    welcome_message += "Hãy đặt câu hỏi hoặc chia sẻ chủ đề bạn muốn thảo luận nhé!\n\n"
    welcome_message += "🔍 *Một số tính năng của tôi:*\n"
    welcome_message += "• 💬 Trả lời câu hỏi và cung cấp thông tin\n"
    welcome_message += "• 📚 Hỗ trợ nghiên cứu và học tập\n"
    welcome_message += "• 🎨 Gợi ý ý tưởng sáng tạo\n"
    welcome_message += "• 📊 Phân tích dữ liệu đơn giản\n"
    welcome_message += "• 🖼️ Phân tích hình ảnh\n"
    welcome_message += "• 🎨 Tạo hình ảnh từ mô tả văn bản (NEW!)\n"
    welcome_message += "• 🔎 Tìm kiếm thông tin (Sử dụng /search)\n\n"
    welcome_message += "Gõ `/info` để biết thêm chi tiết về tôi nhé!"

    markup = InlineKeyboardMarkup()
    markup.row(
        InlineKeyboardButton("📚 Hướng dẫn sử dụng", callback_data="guide"),
        InlineKeyboardButton("🆕 Cập nhật mới", callback_data="updates"),
    )

    await bot.reply_to(
        message, welcome_message, parse_mode="Markdown", reply_markup=markup
    )


# Gộp tất cả các callback handler lại thành một handler duy nhất
@bot.callback_query_handler(func=lambda call: True)  # Xử lý tất cả các callback
async def handle_all_callbacks(call):
    """Xử lý tất cả các loại callback"""
    user_id = call.from_user.id

    # Xử lý các callback liên quan đến hướng dẫn
    if call.data == "guide":
        guide_text = """
*🔰 Hướng dẫn sử dụng Loki Bot*

1️⃣ *Đặt câu hỏi*: Chỉ cần gõ câu hỏi của bạn và gửi đi.

2️⃣ *Phân tích hình ảnh*: Gửi một hình ảnh kèm mô tả (nếu cần).

3️⃣ *Tạo hình ảnh*: Tạo hình từ văn bản (NEW!!).

4️⃣ *Tìm kiếm thông tin*: Sử dụng lệnh `/search [từ khóa]`.

5️⃣ *Xem thông tin bot*: Sử dụng lệnh `/info`.

6️⃣  *Xóa bộ nhớ*: Sử dụng lệnh `/clear` để xóa lịch sử trò chuyện.

7️⃣ *Xem cập nhật*: Sử dụng lệnh `/update` để xem các tính năng mới.

💡 *Mẹo*: Đặt câu hỏi rõ ràng và cung cấp ngữ cảnh để nhận được câu trả lời chính xác nhất.

🆘 Cần hỗ trợ? Liên hệ @tanbaycu
"""
        await bot.answer_callback_query(call.id)
        await bot.send_message(call.message.chat.id, guide_text, parse_mode="Markdown")

    elif call.data == "updates":
        await bot.answer_callback_query(call.id)
        await send_update_history(call.message)

    # Xử lý các callback liên quan đến hình ảnh
    elif call.data.startswith(("regenerate_image:", "variant_image:", "change_model:")):
        # Kiểm tra xem người dùng có đang có yêu cầu đang xử lý không
        if user_id in active_requests:
            await bot.answer_callback_query(
                call.id,
                "Bạn đã có một yêu cầu tạo hình ảnh đang được xử lý. Vui lòng đợi hoàn tất trước khi tạo ảnh mới.",
                show_alert=True,
            )
            return

        # Xử lý các loại callback khác nhau
        if call.data.startswith("regenerate_image:"):
            prompt = call.data.split(":", 1)[1]
            await bot.answer_callback_query(call.id, "Đang tạo lại hình ảnh...")

            # Tạo fake message để tái sử dụng handler hiện có
            fake_message = Message(
                message_id=call.message.message_id,
                from_user=call.from_user,
                date=call.message.date,
                chat=call.message.chat,
                content_type="text",
                options={},
                json_string="",
            )

            fake_message.text = f"/image {prompt}"

            # Gọi handler tạo hình ảnh với fake message
            await handle_image_generation(fake_message)

        elif call.data.startswith("variant_image:"):
            # Tạo biến thể của hình ảnh (thêm một số biến đổi vào prompt)
            prompt = call.data.split(":", 1)[1]
            await bot.answer_callback_query(
                call.id, "Đang tạo biến thể của hình ảnh..."
            )

            # Thêm các biến đổi ngẫu nhiên vào prompt
            variations = [
                "with different lighting",
                "in a different style",
                "with more details",
                "with different colors",
                "from a different angle",
                "with a different composition",
            ]

            variation = random.choice(variations)
            variant_prompt = f"{prompt}, {variation}"

            # Tạo fake message
            fake_message = Message(
                message_id=call.message.message_id,
                from_user=call.from_user,
                date=call.message.date,
                chat=call.message.chat,
                content_type="text",
                options={},
                json_string="",
            )

            fake_message.text = f"/image {variant_prompt}"

            # Gọi handler tạo hình ảnh
            await handle_image_generation(fake_message)

        elif call.data.startswith("change_model:"):
            # Tạo hình ảnh với model khác
            parts = call.data.split(":", 2)
            if len(parts) >= 3:
                model = parts[1]
                prompt = parts[2]

                await bot.answer_callback_query(
                    call.id,
                    f"Đang tạo hình ảnh với model {IMAGE_CONFIG['models'][model]['display_name']}...",
                )

                # Tạo fake message
                fake_message = Message(
                    message_id=call.message.message_id,
                    from_user=call.from_user,
                    date=call.message.date,
                    chat=call.message.chat,
                    content_type="text",
                    options={},
                    json_string="",
                )

                fake_message.text = f"/image {model} {prompt}"

                # Gọi handler tạo hình ảnh
                await handle_image_generation(fake_message)
            else:
                await bot.answer_callback_query(
                    call.id, "Dữ liệu callback không hợp lệ"
                )

    # Xử lý các callback liên quan đến tìm kiếm và phân tích lại
    elif call.data.startswith(("research:", "reanalyze:")):
        # Kiểm tra xem người dùng có đang có yêu cầu đang xử lý không
        if user_id in active_requests:
            await bot.answer_callback_query(
                call.id,
                "Bạn đã có một yêu cầu đang được xử lý. Vui lòng đợi hoàn tất.",
                show_alert=True,
            )
            return

        # Đánh dấu người dùng đang có yêu cầu đang xử lý
        active_requests[user_id] = {
            "type": call.data.split(":", 1)[0],
            "start_time": time.time(),
        }

        try:
            if call.data.startswith("research:"):
                query = call.data.split(":", 1)[1]

                # Giới hạn độ dài truy vấn để tránh lạm dụng
                if len(query) > 200:
                    query = query[:200] + "..."

                await bot.answer_callback_query(call.id, "Đang tìm kiếm lại...")

                # Tạo thông báo đang suy nghĩ
                thinking_message = await bot.send_message(
                    call.message.chat.id,
                    f"🔍 *Đang tìm kiếm...*\n\nĐang tìm kiếm: `{query}`",
                    reply_to_message_id=call.message.message_id,
                    parse_mode="Markdown",
                )

                # Tạo kết quả tìm kiếm
                async with aiohttp.ClientSession() as session:
                    search_result = await generate_search_response(
                        query, call.from_user.id, session
                    )

                # Gửi kết quả tìm kiếm với xử lý cải tiến
                max_length = 4096
                if len(search_result) > max_length:
                    chunks = [
                        search_result[i : i + max_length]
                        for i in range(0, len(search_result), max_length)
                    ]

                    # Giới hạn số lượng phần để tránh spam
                    if len(chunks) > 2:
                        chunks = chunks[:1] + [
                            f"{chunks[1][:1000]}...\n\n*Kết quả tìm kiếm quá dài và đã được cắt ngắn.*"
                        ]

                    for i, chunk in enumerate(chunks):
                        if i == 0:
                            sent_message = await bot.edit_message_text(
                                chat_id=thinking_message.chat.id,
                                message_id=thinking_message.message_id,
                                text=chunk,
                                parse_mode="Markdown",
                            )
                        else:
                            sent_message = await bot.send_message(
                                chat_id=thinking_message.chat.id,
                                text=chunk,
                                parse_mode="Markdown",
                            )

                        # Chỉ thêm nút tìm kiếm lại vào phần cuối cùng
                        if i == len(chunks) - 1:
                            markup = InlineKeyboardMarkup()
                            markup.add(
                                InlineKeyboardButton(
                                    "🔄 Tìm kiếm lại", callback_data=f"research:{query}"
                                )
                            )
                            await bot.edit_message_reply_markup(
                                chat_id=sent_message.chat.id,
                                message_id=sent_message.message_id,
                                reply_markup=markup,
                            )
                else:
                    sent_message = await bot.edit_message_text(
                        chat_id=thinking_message.chat.id,
                        message_id=thinking_message.message_id,
                        text=search_result,
                        parse_mode="Markdown",
                    )
                    markup = InlineKeyboardMarkup()
                    markup.add(
                        InlineKeyboardButton(
                            "🔄 Tìm kiếm lại", callback_data=f"research:{query}"
                        )
                    )
                    await bot.edit_message_reply_markup(
                        chat_id=sent_message.chat.id,
                        message_id=sent_message.message_id,
                        reply_markup=markup,
                    )

            elif call.data.startswith("reanalyze:"):
                _, image_path, prompt = call.data.split(":", 2)

                # Giới hạn độ dài prompt để tránh lạm dụng
                if len(prompt) > 200:
                    prompt = prompt[:200] + "..."

                await bot.answer_callback_query(
                    call.id, "Đang phân tích lại hình ảnh..."
                )
                waiting_message = await bot.send_message(
                    call.message.chat.id,
                    f"🔍 *Đang phân tích lại hình ảnh*\n\nYêu cầu: `{prompt}`",
                    parse_mode="Markdown",
                )
                await analyze_image(call.message, image_path, prompt, waiting_message)

        except Exception as e:
            logger.error(f"Error in callback handler: {str(e)}")
            logger.error(traceback.format_exc())
            try:
                await bot.answer_callback_query(
                    call.id, "Đã xảy ra lỗi khi xử lý yêu cầu của bạn.", show_alert=True
                )
            except:
                pass
        finally:
            # Luôn xóa người dùng khỏi active_requests khi hoàn thành
            if user_id in active_requests:
                del active_requests[user_id]

    elif call.data.startswith("version:"):
        version = call.data.split(":", 1)[1]

        if version == "latest":
            # Quay lại hiển thị phiên bản mới nhất
            current_date = "15/03/2024"
            latest_version = "🚀 *Cập nhật v1.9 (Mới nhất - {date})*\n\n...".format(
                date=current_date
            )
            # Thêm nội dung đầy đủ của phiên bản mới nhất

            # Tạo lại markup với các nút phiên bản
            markup = InlineKeyboardMarkup(row_width=3)
            version_buttons = []
            for i in range(8, 0, -1):
                version_buttons.append(
                    InlineKeyboardButton(f"v1.{i}", callback_data=f"version:1.{i}")
                )

            for i in range(0, len(version_buttons), 3):
                row_buttons = version_buttons[i : i + 3]
                markup.row(*row_buttons)

            await bot.edit_message_text(
                chat_id=call.message.chat.id,
                message_id=call.message.message_id,
                text=latest_version
                + "\n\n📜 *Phiên bản trước đó*: Sử dụng các nút bên dưới để xem lịch sử cập nhật đầy đủ.",
                parse_mode="Markdown",
                reply_markup=markup,
            )
        else:
            # Hiển thị nội dung của phiên bản được chọn
            await handle_version_callback(call)

    # Nếu không phải các loại callback đã biết, trả về thông báo lỗi
    else:
        await bot.answer_callback_query(
            call.id, "Callback không hợp lệ hoặc đã hết hạn", show_alert=True
        )


@bot.message_handler(content_types=["photo"])
async def handle_photo(message: Message):
    try:
        waiting_message = await bot.reply_to(
            message, "🔍 Đang phân tích hình ảnh, vui lòng chờ..."
        )

        file_info = await bot.get_file(message.photo[-1].file_id)
        downloaded_file = await bot.download_file(file_info.file_path)

        file_name = f"user_image_{message.from_user.id}.jpg"
        with open(file_name, "wb") as new_file:
            new_file.write(downloaded_file)

        caption = message.caption if message.caption else "Hãy phân tích hình ảnh này"
        caption = caption[:1000]

        await analyze_image(message, file_name, caption, waiting_message)

    except Exception as e:
        logger.error(f"Lỗi khi xử lý ảnh: {str(e)}")
        error_message = format_error_message(
            ErrorCodes.UNKNOWN_ERROR, "Đã xảy ra lỗi khi xử lý ảnh. Vui lòng thử lại."
        )
        await bot.reply_to(message, error_message, parse_mode="Markdown")


async def analyze_image(
    message: Message, image_path: str, prompt: str, waiting_message: Message
):
    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        current_key = await api_key_manager.get_current_key()

        prompt_generator = AdvancedPromptGenerator()
        topic = prompt_generator.detect_topic(prompt)
        emojis = prompt_generator.get_emojis(topic, 3)
        emoji_str = " ".join(emojis)

        gemini_prompt = normalize_utf8(
            f"""
        {emoji_str} Hãy phân tích hình ảnh này bằng tiếng Việt và trả về kết quả được định dạng bằng Markdown. 
        Yêu cầu cụ thể:

        1. **Tổng quan**: Mô tả tổng quan về hình ảnh.
        2. **Đối tượng chính**: Xác định và liệt kê các đối tượng chính trong hình.
        3. **Phân tích kỹ thuật**: 
           - Bố cục
           - Màu sắc
           - Ánh sáng
           - Góc chụp
        4. **Văn bản**: Nếu có văn bản trong hình, hãy trích xuất và giải thích.
        5. **Ý nghĩa**: Đưa ra nhận xét về ý nghĩa hoặc thông điệp của hình ảnh.

        Yêu cầu bổ sung của người dùng: {prompt}

        Hãy trả lời bằng tiếng Việt, sử dụng định dạng Markdown để làm nổi bật các phần quan trọng.
        Sử dụng:
        - **text** cho các tiêu đề
        - *text* cho các điểm nhấn
        - • cho các danh sách
        - Emoji phù hợp với nội dung
        
        Phản hồi PHẢI ngắn gọn, xúc tích, tránh dài dòng. Tập trung vào thông tin chính xác, tránh thông tin mơ hồ.
        """
        )

        data = {
            "contents": [
                {
                    "parts": [
                        {"text": gemini_prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": encoded_image,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.4,
                "topK": 32,
                "topP": 1,
                "maxOutputTokens": 4096,
            },
        }

        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            url = f"{API_URL}?key={current_key}"
            async with session.post(url, json=data) as response:
                if response.status == 429:
                    new_key = await api_key_manager.handle_error(429)
                    # Retry with new key
                    url = f"{API_URL}?key={new_key}"
                    async with session.post(url, json=data) as retry_response:
                        result = await retry_response.json()
                else:
                    result = await response.json()

        end_time = time.time()
        processing_time = end_time - start_time

        if "candidates" in result and result["candidates"]:
            analysis = normalize_utf8(
                result["candidates"][0]["content"]["parts"][0]["text"]
            )

            # Format the response with processing time
            full_response = f"⏱️ *Thời gian xử lý*: {processing_time:.2f}s\n\n{analysis}"

            max_length = 4000
            chunks = [
                full_response[i : i + max_length]
                for i in range(0, len(full_response), max_length)
            ]

            await bot.delete_message(
                chat_id=waiting_message.chat.id, message_id=waiting_message.message_id
            )

            for chunk in chunks:
                try:
                    await bot.send_message(
                        message.chat.id, chunk, parse_mode="Markdown"
                    )
                except Exception as e:
                    logger.error(f"Lỗi khi gửi phân đoạn phân tích: {str(e)}")
                    # If Markdown parsing fails, send without formatting
                    await bot.send_message(message.chat.id, chunk)

        else:
            error_message = format_error_message(
                ErrorCodes.API_ERROR, "Không thể phân tích hình ảnh. Vui lòng thử lại."
            )
            await bot.edit_message_text(
                chat_id=waiting_message.chat.id,
                message_id=waiting_message.message_id,
                text=error_message,
                parse_mode="Markdown",
            )

    except Exception as e:
        logger.error(f"Lỗi khi phân tích ảnh: {str(e)}")
        error_message = format_error_message(
            ErrorCodes.UNKNOWN_ERROR, f"Đã xảy ra lỗi khi phân tích ảnh: {str(e)}"
        )
        try:
            await bot.edit_message_text(
                chat_id=waiting_message.chat.id,
                message_id=waiting_message.message_id,
                text=error_message,
                parse_mode="Markdown",
            )
        except:
            await bot.edit_message_text(
                chat_id=waiting_message.chat.id,
                message_id=waiting_message.message_id,
                text=error_message,
            )

    finally:
        if os.path.exists(image_path):
            os.remove(image_path)


@bot.message_handler(func=lambda message: not message.text.startswith("/"))
async def handle_message(message):
    async def process_and_send_response():
        try:
            thinking_message = await bot.reply_to(message, "🤔 Bot đang suy nghĩ...")

            start_time = time.time()
            async with aiohttp.ClientSession() as session:
                response = await generate_response(
                    message.text, message.from_user.id, session
                )
            end_time = time.time()
            processing_time = end_time - start_time

            full_response = f"⏱️ *Thời gian xử lý*: {processing_time:.2f}s\n\n{response}"

            max_length = 4096
            if len(full_response) > max_length:
                chunks = [
                    full_response[i : i + max_length]
                    for i in range(0, len(full_response), max_length)
                ]
                for i, chunk in enumerate(chunks):
                    try:
                        if i == 0:
                            sent_message = await bot.edit_message_text(
                                chat_id=thinking_message.chat.id,
                                message_id=thinking_message.message_id,
                                text=chunk,
                                parse_mode="Markdown",
                            )
                        else:
                            sent_message = await bot.send_message(
                                chat_id=thinking_message.chat.id,
                                text=chunk,
                                parse_mode="Markdown",
                            )
                    except Exception as e:
                        logger.error(
                            f"Telegram API error when sending chunk {i+1}: {str(e)}"
                        )
                        if "can't parse entities" in str(e):
                            if i == 0:
                                sent_message = await bot.edit_message_text(
                                    chat_id=thinking_message.chat.id,
                                    message_id=thinking_message.message_id,
                                    text=chunk,
                                )
                            else:
                                sent_message = await bot.send_message(
                                    chat_id=thinking_message.chat.id, text=chunk
                                )

            else:
                try:
                    sent_message = await bot.edit_message_text(
                        chat_id=thinking_message.chat.id,
                        message_id=thinking_message.message_id,
                        text=full_response,
                        parse_mode="Markdown",
                    )
                    logger.info(f"Đã trả lời tin nhắn cho user {message.from_user.id}")
                except Exception as e:
                    logger.error(f"Lỗi khi xử lý tin nhắn: {str(e)}")
                    error_message = format_error_message(
                        ErrorCodes.UNKNOWN_ERROR,
                        "Xin lỗi, đã xảy ra lỗi khi xử lý tin nhắn của bạn. Vui lòng thử lại sau.",
                    )
                    try:
                        await bot.edit_message_text(
                            chat_id=thinking_message.chat.id,
                            message_id=thinking_message.message_id,
                            text=error_message,
                            parse_mode="Markdown",
                        )
                    except:
                        await bot.send_message(
                            chat_id=thinking_message.chat.id,
                            text=error_message,
                            parse_mode="Markdown",
                        )

        except aiohttp.ClientError as e:
            logger.error(f"Client error: {str(e)}")
            error_message = format_error_message(
                ErrorCodes.API_ERROR, "Đã xảy ra lỗi kết nối. Vui lòng thử lại sau."
            )
            await bot.send_message(
                chat_id=message.chat.id, text=error_message, parse_mode="Markdown"
            )

        except asyncio.TimeoutError:
            logger.error("Timeout error")
            error_message = format_error_message(
                ErrorCodes.TIMEOUT,
                "Yêu cầu đã hết thời gian chờ. Vui lòng thử lại sau.",
            )
            await bot.send_message(
                chat_id=message.chat.id, text=error_message, parse_mode="Markdown"
            )

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            error_message = format_error_message(
                ErrorCodes.UNKNOWN_ERROR,
                "Đã xảy ra lỗi không mong đợi. Vui lòng thử lại sau.",
            )
            await bot.send_message(
                chat_id=message.chat.id, text=error_message, parse_mode="Markdown"
            )

    asyncio.create_task(process_and_send_response())


@bot.message_handler(commands=["info"])
async def send_info(message):
    info_text = (
        "🤖 *Xin chào! Tôi là Loki, AI Assistant của bạn* 🤖\n\n"
        "Tôi được phát triển dựa trên mô hình Gemini 2.0, với khả năng tạo ra các câu trả lời thông minh và linh hoạt. "
        "Hãy để tôi hỗ trợ bạn trong nhiều lĩnh vực khác nhau!\n\n"
        "🌟 *Các tính năng nổi bật:*\n"
        "• 💬 Trò chuyện và trả lời câu hỏi\n"
        "• 📚 Cung cấp thông tin đa dạng\n"
        "• 🔍 Hỗ trợ nghiên cứu và học tập\n"
        "• 🎨 Gợi ý ý tưởng sáng tạo\n"
        "• 📊 Phân tích dữ liệu đơn giản\n"
        "• 🖼️ Phân tích hình ảnh\n"
        "• 🎨 Tạo hình ảnh từ mô tả văn bản (NEW!)\n"
        "• 🔎 Tìm kiếm thông tin (Sử dụng /search)\n\n"
        "🛠 *Công cụ hữu ích:*\n"
        "• `/search [từ khóa]`: Tìm kiếm thông tin\n"
        "• `/imagehelp`: Xem hướng dẫn tạo ảnh\n"
        "• `/clear`: Xóa bộ nhớ cuộc trò chuyện\n"
        "• `/update`: Xem bản cập nhật các chức năng của bot\n\n"
        "💡 *Mẹo sử dụng:*\n"
        "1. Đặt câu hỏi rõ ràng và cụ thể\n"
        "2. Cung cấp ngữ cảnh nếu cần thiết\n"
        "3. Gửi hình ảnh để được phân tích\n"
        "4. Sử dụng /search để tìm kiếm thông tin\n\n"
        "🔒 *Bảo mật:*\n"
        "Tôi tôn trọng quyền riêng tư của bạn. Thông tin cá nhân sẽ không được lưu trữ sau khi kết thúc cuộc trò chuyện.\n\n"
        "Hãy khám phá thêm về tôi qua các liên kết dưới đây:"
    )

    markup = InlineKeyboardMarkup()
    buttons = [
        ("🌐 Website", "https://tanbaycu.is-a.dev"),
        ("📘 Facebook", "https://facebook.com/tanbaycu.kaiser"),
        ("📞 Liên hệ hỗ trợ", "https://t.me/tanbaycu"),
    ]

    for text, url in buttons:
        button = InlineKeyboardButton(text, url=url)
        markup.add(button)

    try:
        await bot.send_message(
            message.chat.id, info_text, reply_markup=markup, parse_mode="Markdown"
        )
    except Exception as e:
        logger.error(f"Error sending info message: {str(e)}")
        error_message = format_error_message(
            ErrorCodes.UNKNOWN_ERROR,
            "Xin lỗi, đã xảy ra lỗi khi gửi thông tin. Vui lòng thử lại sau.",
        )
        await bot.reply_to(message, error_message, parse_mode="Markdown")


@bot.message_handler(commands=["update"])
async def send_update_history(message):
    """Hiển thị lịch sử cập nhật với phiên bản mới nhất và tùy chọn xem các phiên bản cũ"""

    # Ngày cập nhật mới nhất
    current_date = "21/04/2025"

    # Phiên bản mới nhất
    latest_version = """
🖼️ *Cập nhật v1.9 (21/04/2025)*

• 🚀 **Cải tiến Zephyr Prompt**:
  - Tinh chỉnh giọng điệu Gen Z: gần gũi, cảm hứng, tránh lặp lại và từ lóng lạm dụng
  - Bao quát mọi kịch bản: học tập, công việc, sáng tạo, lifestyle, tech (như an ninh mạng với TryHackMe), và câu hỏi random
  - Cá nhân hóa thông minh: dựa trên lịch sử trò chuyện, ví dụ gợi ý tool an ninh mạng cho người dùng quan tâm
  - Sáng tạo mini-story: trả lời ngắn gọn, mỗi câu là một hành trình cảm xúc
  - Tích hợp trend tháng 1/2025: app, công nghệ, phong cách sống Gen Z

• 🛠️ **Hạn chế lỗi hệ thống**:
  - Xử lý lỗi Telegram API 400 (Bad Request, Can't parse entities): giới hạn ký tự < 4096, chia tin nhắn dài, kiểm tra định dạng
  - Cơ chế trả lời thay thế khi gặp lỗi: "Lag chút, nhưng tui có giải pháp nè!"
  - Tự kiểm tra tin nhắn trước khi gửi, tránh ký tự đặc biệt hoặc định dạng lỗi
  - Lưu log lỗi (như lỗi 2025-04-21 19:20:57) để cải thiện độ ổn định

• 📈 **Tăng cường trải nghiệm**:
  - Ngắn gọn hơn: trả lời 30-80 từ cho câu đơn giản, 100-200 từ cho câu phức tạp
  - Thấu hiểu cảm xúc: đồng điệu với tâm trạng (stress, hào hứng, tò mò)
  - Giao diện ngôn ngữ mượt mà: diễn đạt linh hoạt, không cứng nhắc, như "Cần boost hả? Vào việc nào!"

• 🔧 **Cải tiến tổng thể**:
  - Tối ưu hóa hiệu suất xử lý câu hỏi mơ hồ với phỏng đoán thông minh
  - Nâng cao độ tin cậy với dữ liệu cập nhật tháng 1/2025
  - Khuyến khích khám phá qua nguồn uy tín như X, giữ vibe Gen Z 100%
""".format(
        date=current_date
    )

    # Tạo tin nhắn chỉ hiển thị phiên bản mới nhất và nút xem thêm
    initial_message = (
        latest_version
        + """
\n📜 *Phiên bản trước đó*: Sử dụng các nút bên dưới để xem lịch sử cập nhật đầy đủ.

💡 *Sắp tới*:
• Dự kiến thêm tính năng phân tích video
• Cải thiện khả năng hiểu ngữ cảnh
• Tối ưu hóa hiệu suất xử lý
• Thêm các tính năng theo yêu cầu người dùng

📝 *Ghi chú*: 
• Bot sẽ tiếp tục được cập nhật và cải thiện
• Sử dụng lệnh /update để xem những thay đổi mới nhất
• Báo cáo lỗi hoặc góp ý tại: @tanbaycu
"""
    )

    # Tạo inline keyboard với các nút để xem phiên bản cũ
    markup = InlineKeyboardMarkup(row_width=3)

    # Thêm nút cho các phiên bản từ v1.8 đến v1.0
    version_buttons = []
    for i in range(8, 0, -1):
        version_buttons.append(
            InlineKeyboardButton(f"v1.{i}", callback_data=f"version:1.{i}")
        )

    # Thêm các nút theo hàng, mỗi hàng 3 nút
    for i in range(0, len(version_buttons), 3):
        row_buttons = version_buttons[i : i + 3]
        markup.row(*row_buttons)

    try:
        await bot.reply_to(
            message, initial_message, parse_mode="Markdown", reply_markup=markup
        )
    except Exception as e:
        logger.error(f"Error sending update history: {str(e)}")
        try:
            # Thử gửi không có Markdown nếu có lỗi
            await bot.reply_to(message, initial_message, reply_markup=markup)
        except Exception as e:
            logger.error(f"Error sending plain update history: {str(e)}")
            error_message = format_error_message(
                ErrorCodes.UNKNOWN_ERROR,
                "Xin lỗi, đã xảy ra lỗi khi gửi lịch sử cập nhật. Vui lòng thử lại sau.",
            )
            await bot.reply_to(message, error_message, parse_mode="Markdown")


async def handle_version_callback(call):
    """Xử lý callback khi người dùng nhấn vào nút xem phiên bản cũ"""
    version = call.data.split(":", 1)[1]

    # Nội dung các phiên bản cũ
    version_content = {
        "1.9": """
        🖼️ *Cập nhật v1.9 (21/04/2025)*

• 🚀 **Cải tiến Zephyr Prompt**:
  - Tinh chỉnh giọng điệu Gen Z: gần gũi, cảm hứng, tránh lặp lại và từ lóng lạm dụng
  - Bao quát mọi kịch bản: học tập, công việc, sáng tạo, lifestyle, tech (như an ninh mạng với TryHackMe), và câu hỏi random
  - Cá nhân hóa thông minh: dựa trên lịch sử trò chuyện, ví dụ gợi ý tool an ninh mạng cho người dùng quan tâm
  - Sáng tạo mini-story: trả lời ngắn gọn, mỗi câu là một hành trình cảm xúc
  - Tích hợp trend tháng 1/2025: app, công nghệ, phong cách sống Gen Z

• 🛠️ **Hạn chế lỗi hệ thống**:
  - Xử lý lỗi Telegram API 400 (Bad Request, Can't parse entities): giới hạn ký tự < 4096, chia tin nhắn dài, kiểm tra định dạng
  - Cơ chế trả lời thay thế khi gặp lỗi: "Lag chút, nhưng tui có giải pháp nè!"
  - Tự kiểm tra tin nhắn trước khi gửi, tránh ký tự đặc biệt hoặc định dạng lỗi
  - Lưu log lỗi (như lỗi 2025-04-21 19:20:57) để cải thiện độ ổn định

• 📈 **Tăng cường trải nghiệm**:
  - Ngắn gọn hơn: trả lời 30-80 từ cho câu đơn giản, 100-200 từ cho câu phức tạp
  - Thấu hiểu cảm xúc: đồng điệu với tâm trạng (stress, hào hứng, tò mò)
  - Giao diện ngôn ngữ mượt mà: diễn đạt linh hoạt, không cứng nhắc, như "Cần boost hả? Vào việc nào!"

• 🔧 **Cải tiến tổng thể**:
  - Tối ưu hóa hiệu suất xử lý câu hỏi mơ hồ với phỏng đoán thông minh
  - Nâng cao độ tin cậy với dữ liệu cập nhật tháng 1/2025
  - Khuyến khích khám phá qua nguồn uy tín như X, giữ vibe Gen Z 100%
  """,




        
        "1.8": """
🖼️ *Cập nhật v1.8 (14/03/2024)*

• 🖼️ Thêm lệnh /image:
  - Tạo hình ảnh từ mô tả văn bản
  - Hỗ trợ nhiều model AI khác nhau (Flux, DALL-E, SDXL)
  - Tự động dịch prompt tiếng Việt sang tiếng Anh
  - Tối ưu hóa hiệu suất với cache và xử lý đa luồng
  - Giao diện tương tác với nút tạo lại và biến thể
  - Hiển thị tiến trình tạo ảnh theo thời gian thực
  - Xử lý lỗi mạnh mẽ với cơ chế retry và fallback

• 📊 Thêm lệnh /imagehelp và /imagestats:
  - Hướng dẫn chi tiết cách sử dụng lệnh /image
  - Thống kê về việc sử dụng tính năng tạo ảnh (cho admin)

• 🔧 Cải tiến tổng thể:
  - Tối ưu hóa sử dụng tài nguyên
  - Cải thiện độ ổn định
  - Nâng cao trải nghiệm người dùng
""",
        
        "1.7": """
🎨 *Cập nhật v1.7*

• 🎨 Cải tiến giao diện người dùng:
  - Thêm nút tương tác cho phản hồi
  - Nâng cao trải nghiệm người dùng với menu hướng dẫn

• 🔄 Tính năng tạo lại câu trả lời:
  - Cho phép người dùng yêu cầu câu trả lời mới

• 🖼️ Cải thiện phân tích hình ảnh:
  - Thêm tùy chọn phân tích lại hình ảnh

• 🔍 Nâng cấp tính năng tìm kiếm:
  - Thêm nút "Thử lại" cho kết quả tìm kiếm

• 🛡️ Cải tiến hệ thống xử lý lỗi:
  - Thêm mã lỗi cụ thể cho từng loại lỗi
  - Cải thiện thông báo lỗi với hướng dẫn khắc phục

• 📊 Tối ưu hóa hiệu suất:
  - Cải thiện tốc độ xử lý yêu cầu
  - Giảm thiểu thời gian chờ đợi
""",
        "1.6": """
🧠 *Cập nhật v1.6*

• 🔍 Cải tiến hệ thống xử lý ngữ cảnh:
  - Sử dụng TF-IDF và cosine similarity để chọn ngữ cảnh phù hợp
  - Tăng kích thước bộ nhớ ngắn hạn và dài hạn

• 🧠 Nâng cao khả năng hiểu ý định người dùng:
  - Cải thiện phương pháp trích xuất thông tin quan trọng
  - Tích hợp thông tin quan trọng vào prompt

• 💬 Cải thiện chất lượng phản hồi:
  - Thêm hướng dẫn để tránh hỏi lại người dùng không cần thiết
  - Tăng cường khả năng suy luận từ thông tin có sẵn

• 🛠️ Tối ưu hóa cấu trúc mã nguồn:
  - Tái cấu trúc các lớp và phương thức để dễ bảo trì và mở rộng

• 🔧 Cải thiện xử lý lỗi và logging
""",
        "1.5": """
🎭 *Cập nhật v1.5*

• 🧠 Hệ thống prompt thông minh:
  - Phát hiện chủ đề tự động
  - Phân tích cảm xúc người dùng
  - Đánh giá độ phức tạp của câu hỏi

• 🎭 Tính cách động:
  - Tự động điều chỉnh phong cách phản hồi
  - Thích ứng với ngữ cảnh cuộc trò chuyện

• 🎯 Cải thiện phản hồi:
  - Đa dạng hóa cấu trúc câu
  - Tối ưu hóa sử dụng emoji theo chủ đề
  - Phản hồi nhất quán hơn với lịch sử trò chuyện

• 📊 Phân tích hình ảnh thông minh hơn:
  - Điều chỉnh phân tích theo chủ đề
  - Cải thiện định dạng kết quả
""",
        "1.4": """
🔄 *Cập nhật v1.4*

• 🔄 Thêm cơ chế luân chuyển API key tự động

• ⏱️ Thêm delay 1s để tránh spam

• 🛡️ Cải thiện xử lý lỗi 429 (rate limit)

• 🖼️ Nâng cấp phân tích hình ảnh:
  - Định dạng Markdown cho kết quả phân tích
  - Cấu trúc phản hồi rõ ràng hơn
  - Thêm emoji và điểm nhấn trực quan
  - Hiển thị thời gian xử lý

• 🔧 Các cải tiến khác:
  - Tối ưu hóa sử dụng bộ nhớ
  - Cải thiện độ ổn định
  - Sửa các lỗi được báo cáo
""",
        "1.3": """
⚡ *Cập nhật v1.3*

• 🔧 Tối ưu hóa mã nguồn

• ⚡ Cải thiện tốc độ xử lý

• 🛡️ Nâng cao độ ổn định

• 🐛 Sửa các lỗi nhỏ được báo cáo
""",
        "1.2": """
📝 *Cập nhật v1.2*

• 🎯 Cải thiện độ chính xác của phản hồi

• 🧠 Tối ưu hóa bộ nhớ và xử lý context

• 😊 Thêm emoji và định dạng cho tin nhắn

• 📋 Cập nhật lệnh /info với thông tin mới
""",
        "1.1": """
🖼️ *Cập nhật v1.1*

• 🖼️ Thêm tính năng phân tích hình ảnh

• 🛡️ Cải thiện xử lý lỗi và ghi log

• 📝 Tối ưu hóa việc gửi tin nhắn dài

• 📋 Thêm định dạng Markdown cho phản hồi
""",
        "1.0": """
🚀 *Phiên bản ban đầu v1.0*

• 🤖 Tích hợp với Gemini 2.0 API

• 💬 Xử lý tin nhắn văn bản cơ bản

• 🧠 Bộ nhớ ngắn hạn và dài hạn

• 📋 Lệnh cơ bản: /start, /info
""",
    }

    # Nếu không có nội dung cho phiên bản được chọn, hiển thị thông báo chung
    if version not in version_content:
        content = f"*Phiên bản v{version}*\n\nThông tin chi tiết về phiên bản này không có sẵn."
    else:
        content = version_content[version]

    # Tạo nút quay lại phiên bản mới nhất
    markup = InlineKeyboardMarkup()
    markup.add(
        InlineKeyboardButton(
            "🔙 Quay lại phiên bản mới nhất", callback_data="version:latest"
        )
    )

    await bot.answer_callback_query(call.id)
    await bot.edit_message_text(
        chat_id=call.message.chat.id,
        message_id=call.message.message_id,
        text=content,
        parse_mode="Markdown",
        reply_markup=markup,
    )


@bot.message_handler(commands=["clear"])
async def clear_memory(message):
    user_id = message.from_user.id
    memory_handler = AdvancedMemoryHandler(user_id)
    memory_handler.clear_memory()
    await bot.reply_to(message, "✅ Đã xóa bộ nhớ của bạn.")


@bot.message_handler(commands=["search"])
async def handle_search(message):
    query = message.text.split("/search", 1)[-1].strip()
    if not query:
        await bot.reply_to(
            message, "❗ Vui lòng nhập từ khóa tìm kiếm sau lệnh /search."
        )
        return

    try:
        thinking_message = await bot.reply_to(message, "🔍 Đang tìm kiếm...")

        async with aiohttp.ClientSession() as session:
            search_result = await generate_search_response(
                query, message.from_user.id, session
            )

        max_length = 4096
        if len(search_result) > max_length:
            chunks = [
                search_result[i : i + max_length]
                for i in range(0, len(search_result), max_length)
            ]
            for i, chunk in enumerate(chunks):
                if i == 0:
                    sent_message = await bot.edit_message_text(
                        chat_id=thinking_message.chat.id,
                        message_id=thinking_message.message_id,
                        text=chunk,
                        parse_mode="Markdown",
                    )
                else:
                    sent_message = await bot.send_message(
                        chat_id=thinking_message.chat.id,
                        text=chunk,
                        parse_mode="Markdown",
                    )

        else:
            sent_message = await bot.edit_message_text(
                chat_id=thinking_message.chat.id,
                message_id=thinking_message.message_id,
                text=search_result,
                parse_mode="Markdown",
            )

    except Exception as e:
        logger.error(f"Lỗi khi xử lý tìm kiếm: {str(e)}")
        error_message = format_error_message(
            ErrorCodes.UNKNOWN_ERROR,
            "Đã xảy ra lỗi khi thực hiện tìm kiếm. Vui lòng thử lại sau.",
        )
        await bot.edit_message_text(
            chat_id=thinking_message.chat.id,
            message_id=thinking_message.message_id,
            text=error_message,
            parse_mode="Markdown",
        )


async def generate_search_response(
    query: str, user_id: int, session: aiohttp.ClientSession
) -> str:
    prompt_generator = AdvancedPromptGenerator()

    search_prompt = f"""Hãy tạo ra một kết quả tìm kiếm giả lập cho truy vấn: "{query}".
    Kết quả nên bao gồm:
    1. Một tóm tắt ngắn gọn về chủ đề (2-3 câu)
    2. 3-5 điểm chính liên quan đến truy vấn
    3. 2-3 liên kết giả định (không cần phải là liên kết thật) đến các nguồn thông tin liên quan
    
    Hãy định dạng kết quả bằng Markdown và sử dụng emoji phù hợp."""

    enhanced_prompt = prompt_generator.generate_enhanced_prompt(search_prompt, [])

    try:
        current_key = await api_key_manager.get_current_key()
        url = f"{API_URL}?key={current_key}"
        headers = {"Content-Type": "application/json"}

        data = {
            "contents": [{"parts": [{"text": enhanced_prompt}]}],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 2048,
            },
        }

        async with session.post(
            url, headers=headers, json=data, timeout=30
        ) as response:
            if response.status == 429:
                new_key = await api_key_manager.handle_error(429)
                url = f"{API_URL}?key={new_key}"
                async with session.post(
                    url, headers=headers, json=data, timeout=30
                ) as retry_response:
                    result = await retry_response.json()
            else:
                result = await response.json()

        if "candidates" in result and result["candidates"]:
            search_result = normalize_utf8(
                result["candidates"][0]["content"]["parts"][0]["text"]
            )
            return f"🔍 Kết quả tìm kiếm cho '*{query}*':\n\n{search_result}"
        else:
            return format_error_message(
                ErrorCodes.API_ERROR, "Không tìm thấy kết quả phù hợp."
            )

    except Exception as e:
        logger.error(f"Lỗi khi thực hiện tìm kiếm: {str(e)}")
        return format_error_message(
            ErrorCodes.UNKNOWN_ERROR, f"Đã xảy ra lỗi khi tìm kiếm: {str(e)}"
        )


# Cấu hình cho image generation
IMAGE_CONFIG = {
    "models": {
        "flux": {
            "name": "flux",
            "display_name": "Flux",
            "max_retries": 3,
            "timeout": 60,
            "cooldown": 5,  # Thời gian chờ giữa các lần retry (giây)
        },
        "dalle": {
            "name": "dall-e-3",
            "display_name": "DALL-E 3",
            "max_retries": 2,
            "timeout": 90,
            "cooldown": 10,
        },
    },
    "default_model": "flux",
    "max_concurrent_generations": 5,  # Số lượng tối đa các yêu cầu tạo ảnh đồng thời
    "cache_size": 100,  # Số lượng kết quả được cache
    "temp_dir": "temp_images",  # Thư mục lưu ảnh tạm thời
    "default_negative_prompt": "ugly, deformed, noisy, blurry, distorted, out of focus, bad anatomy, extra limbs, poorly drawn face, poorly drawn hands, missing fingers",
}

# Tạo thư mục temp nếu chưa tồn tại
os.makedirs(IMAGE_CONFIG["temp_dir"], exist_ok=True)

# --- INITIALIZATION ---
# Khởi tạo g4f client
g4f_client = Client()

# Semaphore để giới hạn số lượng yêu cầu đồng thời
image_generation_semaphore = asyncio.Semaphore(
    IMAGE_CONFIG["max_concurrent_generations"]
)

# Theo dõi các yêu cầu đang xử lý
active_requests = {}


# --- UTILITY FUNCTIONS ---
@lru_cache(maxsize=IMAGE_CONFIG["cache_size"])
def get_cached_image_url(
    prompt: str, model: str, negative_prompt: str = ""
) -> Optional[str]:
    """Cache kết quả tạo ảnh để tái sử dụng"""
    return None  # Ban đầu cache trống


def update_image_cache(prompt: str, model: str, negative_prompt: str, url: str) -> None:
    """Cập nhật cache với URL ảnh mới tạo"""
    get_cached_image_url.cache_clear()  # Xóa cache cũ
    get_cached_image_url(prompt, model, negative_prompt)  # Thêm vào cache


def extract_model_from_command(text: str) -> Tuple[str, str]:
    """Trích xuất model và prompt từ lệnh"""
    parts = text.split(" ", 2)

    if len(parts) < 2:
        return IMAGE_CONFIG["default_model"], ""

    if len(parts) == 2:
        return IMAGE_CONFIG["default_model"], parts[1].strip()

    # Kiểm tra xem phần thứ hai có phải là model không
    potential_model = parts[1].lower().strip()
    if potential_model in IMAGE_CONFIG["models"]:
        return potential_model, parts[2].strip()
    else:
        # Nếu không phải model, ghép phần 1 và 2 làm prompt
        return IMAGE_CONFIG["default_model"], " ".join(parts[1:]).strip()


def is_english(text: str) -> bool:
    """Kiểm tra xem văn bản có phải tiếng Anh không (cải tiến)"""
    # Danh sách các từ thông dụng trong tiếng Anh
    common_english_words = {
        "the",
        "be",
        "to",
        "of",
        "and",
        "a",
        "in",
        "that",
        "have",
        "I",
        "it",
        "for",
        "not",
        "on",
        "with",
        "he",
        "as",
        "you",
        "do",
        "at",
    }

    # Tách văn bản thành các từ
    words = text.lower().split()

    # Nếu không có từ nào, không thể xác định
    if not words:
        return False

    # Đếm số từ tiếng Anh thông dụng
    english_word_count = sum(1 for word in words if word in common_english_words)

    # Nếu có ít nhất 20% từ tiếng Anh thông dụng, coi là tiếng Anh
    return english_word_count / len(words) >= 0.2


async def translate_to_english(text: str, user_id: int) -> str:
    """Dịch văn bản sang tiếng Anh sử dụng Gemini API"""
    try:
        logger.info(f"Translating prompt to English for user {user_id}: {text}")

        # Sử dụng hàm generate_response hiện có để dịch
        translation_prompt = f"Translate the following text to English, keep it concise and suitable for image generation. Only return the translation, nothing else: {text}"

        async with aiohttp.ClientSession() as session:
            translated_text = await generate_response(
                translation_prompt, user_id, session
            )

            # Làm sạch kết quả dịch
            translated_text = translated_text.replace("Translation:", "").strip()
            translated_text = translated_text.replace("*", "").strip()
            translated_text = translated_text.replace('"', "").strip()

            logger.info(f"Translated prompt: {translated_text}")
            return translated_text
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        # Nếu dịch thất bại, trả về văn bản gốc
        return text


async def optimize_image(image_data: bytes, quality: int = 90) -> bytes:
    """Tối ưu hóa ảnh để giảm kích thước mà vẫn giữ chất lượng"""
    try:
        # Sử dụng ThreadPoolExecutor để xử lý ảnh không chặn event loop
        loop = asyncio.get_event_loop()
        optimized_data = await loop.run_in_executor(
            executor, lambda: optimize_image_sync(image_data, quality)
        )
        return optimized_data
    except Exception as e:
        logger.error(f"Image optimization error: {str(e)}")
        # Nếu tối ưu thất bại, trả về ảnh gốc
        return image_data


def optimize_image_sync(image_data: bytes, quality: int = 90) -> bytes:
    """Phiên bản đồng bộ của optimize_image để chạy trong executor"""
    try:
        img = Image.open(io.BytesIO(image_data))
        output = io.BytesIO()

        # Giữ nguyên định dạng gốc nếu có thể
        if img.format == "PNG":
            img.save(output, format="PNG", optimize=True)
        else:
            # Mặc định chuyển sang JPEG với chất lượng tùy chỉnh
            img.convert("RGB").save(
                output, format="JPEG", quality=quality, optimize=True
            )

        return output.getvalue()
    except Exception as e:
        logger.error(f"Sync image optimization error: {str(e)}")
        return image_data


def get_progress_bar(percent: int, length: int = 10) -> str:
    """Tạo thanh tiến trình dạng text"""
    filled = int(percent * length / 100)
    bar = "█" * filled + "░" * (length - filled)
    return f"{bar} {percent}%"


def get_random_wait_messages() -> str:
    """Trả về thông báo chờ đợi ngẫu nhiên"""
    messages = [
        "🎨 Đang vẽ tác phẩm của bạn...",
        "✨ Đang tạo hình ảnh, chỉ một chút nữa thôi...",
        "🖌️ Đang phác họa chi tiết...",
        "🔮 Đang biến ý tưởng thành hình ảnh...",
        "🧙‍♂️ Đang thực hiện phép thuật AI...",
        "🎭 Đang sáng tạo nghệ thuật cho bạn...",
        "🌈 Đang thêm màu sắc vào tác phẩm...",
        "📸 Đang xử lý hình ảnh của bạn...",
        "⚡ Đang tạo hình ảnh với tốc độ tia chớp...",
        "🧠 AI đang suy nghĩ về tác phẩm của bạn...",
    ]
    return random.choice(messages)


# --- CORE IMAGE GENERATION FUNCTIONS ---
async def generate_image_with_g4f(
    prompt: str, model: str = "flux", negative_prompt: str = ""
) -> Optional[str]:
    """Tạo hình ảnh sử dụng g4f client với xử lý lỗi và retry"""
    model_config = IMAGE_CONFIG["models"].get(
        model, IMAGE_CONFIG["models"][IMAGE_CONFIG["default_model"]]
    )
    model_name = model_config["name"]
    max_retries = model_config["max_retries"]
    timeout = model_config["timeout"]
    cooldown = model_config["cooldown"]

    # Kiểm tra cache trước
    cached_url = get_cached_image_url(prompt, model, negative_prompt)
    if cached_url:
        logger.info(f"Cache hit for prompt: {prompt[:30]}...")
        return cached_url

    for attempt in range(max_retries):
        try:
            logger.info(
                f"Generating image with {model_name}, attempt {attempt+1}/{max_retries}"
            )

            # Sử dụng ThreadPoolExecutor để không chặn event loop
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                executor,
                lambda: g4f_client.images.generate(
                    model=model_name,
                    prompt=prompt,
                    negative_prompt=(
                        negative_prompt
                        if negative_prompt
                        else IMAGE_CONFIG["default_negative_prompt"]
                    ),
                    response_format="url",
                ),
            )

            if (
                response
                and hasattr(response, "data")
                and response.data
                and len(response.data) > 0
            ):
                image_url = response.data[0].url

                # Cập nhật cache
                update_image_cache(prompt, model, negative_prompt, image_url)

                logger.info(f"Successfully generated image with {model_name}")
                return image_url
            else:
                logger.warning(f"No image URL in the response from {model_name}")

                if attempt < max_retries - 1:
                    logger.info(f"Waiting {cooldown}s before retry...")
                    await asyncio.sleep(cooldown)
        except Exception as e:
            logger.error(
                f"Error in generate_image_with_g4f (attempt {attempt+1}): {str(e)}"
            )

            if attempt < max_retries - 1:
                # Tăng thời gian chờ theo cấp số nhân
                wait_time = cooldown * (attempt + 1)
                logger.info(f"Waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)

    # Nếu tất cả các lần thử đều thất bại
    logger.error(f"All {max_retries} attempts to generate image failed")
    return None


async def download_image(url: str, timeout: int = 30) -> Optional[bytes]:
    """Tải ảnh từ URL với timeout và xử lý lỗi"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=timeout) as response:
                if response.status == 200:
                    image_data = await response.read()

                    # Tối ưu hóa ảnh
                    optimized_data = await optimize_image(image_data)

                    return optimized_data
                else:
                    logger.error(f"Failed to download image: HTTP {response.status}")
                    return None
    except asyncio.TimeoutError:
        logger.error(f"Timeout while downloading image from {url}")
        return None
    except Exception as e:
        logger.error(f"Error downloading image from {url}: {str(e)}")
        return None


async def save_temp_image(image_data: bytes, user_id: int) -> str:
    """Lưu ảnh tạm thời và trả về đường dẫn"""
    # Tạo tên file duy nhất
    timestamp = int(time.time())
    filename = f"{IMAGE_CONFIG['temp_dir']}/img_{user_id}_{timestamp}.jpg"

    try:
        with open(filename, "wb") as f:
            f.write(image_data)
        return filename
    except Exception as e:
        logger.error(f"Error saving temporary image: {str(e)}")
        # Sử dụng tempfile nếu cách thông thường thất bại
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
                temp.write(image_data)
                return temp.name
        except Exception as e2:
            logger.error(f"Error using tempfile: {str(e2)}")
            return ""


# --- COMMAND HANDLERS ---
@bot.message_handler(commands=["image"])
async def handle_image_generation(message: Message):
    """Xử lý lệnh /image để tạo hình ảnh"""
    user_id = message.from_user.id
    chat_id = message.chat.id
    message_id = message.message_id

    # Trích xuất model và prompt
    model, prompt = extract_model_from_command(message.text)

    if not prompt:
        usage_message = (
            "⚠️ *Hướng dẫn sử dụng lệnh /image*\n\n"
            "Cú pháp: `/image [model] prompt`\n\n"
            "*Các model hỗ trợ:*\n"
        )

        for model_key, model_info in IMAGE_CONFIG["models"].items():
            usage_message += f"• `{model_key}` - {model_info['display_name']}\n"

        usage_message += "\n*Ví dụ:*\n"
        usage_message += "• `/image a white siamese cat`\n"
        usage_message += "• `/image flux a futuristic city at night`\n"
        usage_message += "• `/image dalle a portrait of a viking warrior`\n\n"
        usage_message += "Mặc định sẽ sử dụng model `flux` nếu không chỉ định."

        await bot.reply_to(message, usage_message, parse_mode="Markdown")
        return

    # Kiểm tra xem người dùng có đang có yêu cầu đang xử lý không
    if user_id in active_requests:
        await bot.reply_to(
            message,
            "⚠️ Bạn đã có một yêu cầu tạo hình ảnh đang được xử lý. Vui lòng đợi hoàn tất trước khi tạo ảnh mới.",
            parse_mode="Markdown",
        )
        return

    # Đánh dấu người dùng đang có yêu cầu đang xử lý
    active_requests[user_id] = {
        "prompt": prompt,
        "model": model,
        "start_time": time.time(),
    }

    # Gửi thông báo chờ đợi
    waiting_message = await bot.reply_to(
        message,
        f"🎨 *Đang chuẩn bị tạo hình ảnh*\n\n"
        f"• *Prompt:* {prompt}\n"
        f"• *Model:* {IMAGE_CONFIG['models'][model]['display_name']}\n\n"
        f"{get_progress_bar(5)} (Đang khởi tạo...)",
        parse_mode="Markdown",
    )

    # Cập nhật thông báo chờ đợi theo tiến trình
    update_task = asyncio.create_task(
        update_waiting_message(waiting_message, prompt, model)
    )

    try:
        # Kiểm tra và dịch prompt nếu không phải tiếng Anh
        if not is_english(prompt):
            # Cập nhật thông báo
            await bot.edit_message_text(
                f"🎨 *Đang chuẩn bị tạo hình ảnh*\n\n"
                f"• *Prompt:* {prompt}\n"
                f"• *Model:* {IMAGE_CONFIG['models'][model]['display_name']}\n\n"
                f"{get_progress_bar(10)} (Đang dịch prompt...)",
                chat_id=waiting_message.chat.id,
                message_id=waiting_message.message_id,
                parse_mode="Markdown",
            )

            # Dịch prompt sang tiếng Anh
            translated_prompt = await translate_to_english(prompt, user_id)
            logger.info(f"Translated prompt from '{prompt}' to '{translated_prompt}'")

            # Cập nhật thông báo với prompt đã dịch
            await bot.edit_message_text(
                f"🎨 *Đang chuẩn bị tạo hình ảnh*\n\n"
                f"• *Prompt gốc:* {prompt}\n"
                f"• *Prompt đã dịch:* {translated_prompt}\n"
                f"• *Model:* {IMAGE_CONFIG['models'][model]['display_name']}\n\n"
                f"{get_progress_bar(20)} (Đang tạo hình ảnh...)",
                chat_id=waiting_message.chat.id,
                message_id=waiting_message.message_id,
                parse_mode="Markdown",
            )

            # Sử dụng prompt đã dịch
            prompt = translated_prompt

        # Sử dụng semaphore để giới hạn số lượng yêu cầu đồng thời
        async with image_generation_semaphore:
            # Cập nhật thông báo
            await bot.edit_message_text(
                f"🎨 *Đang tạo hình ảnh*\n\n"
                f"• *Prompt:* {prompt}\n"
                f"• *Model:* {IMAGE_CONFIG['models'][model]['display_name']}\n\n"
                f"{get_progress_bar(30)} (Đang xử lý...)",
                chat_id=waiting_message.chat.id,
                message_id=waiting_message.message_id,
                parse_mode="Markdown",
            )

            # Tạo hình ảnh
            start_time = time.time()
            image_url = await generate_image_with_g4f(prompt, model)

            if not image_url:
                # Thử lại với model khác nếu model hiện tại thất bại
                fallback_models = [
                    m for m in IMAGE_CONFIG["models"].keys() if m != model
                ]

                if fallback_models:
                    fallback_model = fallback_models[0]
                    logger.info(
                        f"Trying fallback model {fallback_model} after {model} failed"
                    )

                    await bot.edit_message_text(
                        f"🎨 *Đang tạo hình ảnh*\n\n"
                        f"• *Prompt:* {prompt}\n"
                        f"• *Model:* {IMAGE_CONFIG['models'][model]['display_name']} (thất bại, đang thử với {IMAGE_CONFIG['models'][fallback_model]['display_name']})\n\n"
                        f"{get_progress_bar(40)} (Đang thử lại...)",
                        chat_id=waiting_message.chat.id,
                        message_id=waiting_message.message_id,
                        parse_mode="Markdown",
                    )

                    image_url = await generate_image_with_g4f(prompt, fallback_model)

            # Hủy task cập nhật thông báo chờ đợi
            update_task.cancel()

            if image_url:
                # Cập nhật thông báo
                await bot.edit_message_text(
                    f"🎨 *Đang tạo hình ảnh*\n\n"
                    f"• *Prompt:* {prompt}\n"
                    f"• *Model:* {IMAGE_CONFIG['models'][model]['display_name']}\n\n"
                    f"{get_progress_bar(70)} (Đang tải hình ảnh...)",
                    chat_id=waiting_message.chat.id,
                    message_id=waiting_message.message_id,
                    parse_mode="Markdown",
                )

                # Tải hình ảnh
                image_data = await download_image(image_url)

                if image_data:
                    # Lưu ảnh tạm thời
                    temp_image_path = await save_temp_image(image_data, user_id)

                    if temp_image_path:
                        # Tính thời gian xử lý
                        end_time = time.time()
                        processing_time = end_time - start_time

                        # Cập nhật thông báo
                        await bot.edit_message_text(
                            f"🎨 *Đang tạo hình ảnh*\n\n"
                            f"• *Prompt:* {prompt}\n"
                            f"• *Model:* {IMAGE_CONFIG['models'][model]['display_name']}\n\n"
                            f"{get_progress_bar(90)} (Đang gửi hình ảnh...)",
                            chat_id=waiting_message.chat.id,
                            message_id=waiting_message.message_id,
                            parse_mode="Markdown",
                        )

                        # Gửi hình ảnh
                        with open(temp_image_path, "rb") as photo:
                            caption = (
                                f'🖼️ *Hình ảnh từ prompt*: "{prompt}"\n'
                                f"🤖 *Model*: {IMAGE_CONFIG['models'][model]['display_name']}\n"
                                f"⏱️ *Thời gian xử lý*: {processing_time:.2f}s"
                            )

                            sent_message = await bot.send_photo(
                                chat_id,
                                photo,
                                caption=caption,
                                reply_to_message_id=message_id,
                                parse_mode="Markdown",
                            )

                            # Thêm các nút tương tác
                            markup = InlineKeyboardMarkup(row_width=2)
                            markup.add(
                                InlineKeyboardButton(
                                    "🔄 Tạo lại",
                                    callback_data=truncate_callback_data(
                                        "regenerate_image", prompt
                                    ),
                                ),
                                InlineKeyboardButton(
                                    "🔀 Biến thể",
                                    callback_data=truncate_callback_data(
                                        "variant_image", prompt
                                    ),
                                ),
                            )

                            # Thêm nút chọn model khác
                            model_buttons = []
                            for m_key, m_info in IMAGE_CONFIG["models"].items():
                                if m_key != model:  # Chỉ hiển thị các model khác
                                    model_buttons.append(
                                        InlineKeyboardButton(
                                            f"🔄 Thử với {m_info['display_name']}",
                                            callback_data=f"change_model:{m_key}:{truncate_callback_data('prompt', prompt, 30)}",
                                        )
                                    )

                            # Thêm các nút model theo từng hàng
                            for button in model_buttons:
                                markup.add(button)

                            await bot.edit_message_reply_markup(
                                chat_id=sent_message.chat.id,
                                message_id=sent_message.message_id,
                                reply_markup=markup,
                            )

                        # Xóa thông báo chờ đợi
                        await bot.delete_message(
                            chat_id=waiting_message.chat.id,
                            message_id=waiting_message.message_id,
                        )

                        # Xóa file tạm
                        try:
                            os.remove(temp_image_path)
                        except:
                            pass
                    else:
                        raise Exception("Không thể lưu hình ảnh tạm thời")
                else:
                    raise Exception("Không thể tải hình ảnh từ URL")
            else:
                raise Exception("Không thể tạo hình ảnh với tất cả các model")

    except asyncio.CancelledError:
        logger.warning(f"Image generation for user {user_id} was cancelled")
        try:
            await bot.edit_message_text(
                "❌ *Yêu cầu tạo hình ảnh đã bị hủy*",
                chat_id=waiting_message.chat.id,
                message_id=waiting_message.message_id,
                parse_mode="Markdown",
            )
        except:
            pass

    except Exception as e:
        logger.error(f"Error in handle_image_generation: {str(e)}")
        logger.error(traceback.format_exc())

        error_message = format_error_message(
            ErrorCodes.UNKNOWN_ERROR, f"Đã xảy ra lỗi khi tạo hình ảnh: {str(e)}"
        )

        try:
            await bot.edit_message_text(
                error_message,
                chat_id=waiting_message.chat.id,
                message_id=waiting_message.message_id,
                parse_mode="Markdown",
            )
        except:
            try:
                await bot.send_message(
                    chat_id,
                    error_message,
                    reply_to_message_id=message_id,
                    parse_mode="Markdown",
                )
            except:
                pass

    finally:
        # Xóa người dùng khỏi danh sách yêu cầu đang xử lý
        if user_id in active_requests:
            del active_requests[user_id]

        # Đảm bảo task cập nhật thông báo chờ đợi đã bị hủy
        if update_task and not update_task.done():
            update_task.cancel()


async def update_waiting_message(message: Message, prompt: str, model: str):
    """Cập nhật thông báo chờ đợi với thông tin tiến trình"""
    try:
        progress = 10
        while True:
            if progress >= 60:  # Giới hạn tiến trình giả ở 60%
                progress = 10  # Reset lại để tạo hiệu ứng chờ đợi

            wait_message = get_random_wait_messages()

            await bot.edit_message_text(
                f"🎨 *Đang tạo hình ảnh*\n\n"
                f"• *Prompt:* {prompt}\n"
                f"• *Model:* {IMAGE_CONFIG['models'][model]['display_name']}\n\n"
                f"{get_progress_bar(progress)} ({wait_message})",
                chat_id=message.chat.id,
                message_id=message.message_id,
                parse_mode="Markdown",
            )

            progress += 5
            await asyncio.sleep(3)  # Cập nhật mỗi 3 giây

    except asyncio.CancelledError:
        # Task bị hủy, không cần làm gì
        pass
    except Exception as e:
        logger.error(f"Error updating waiting message: {str(e)}")


# --- HELP COMMAND ---
@bot.message_handler(commands=["imagehelp"])
async def handle_image_help(message: Message):
    """Hiển thị trợ giúp về lệnh /image"""
    help_text = (
        "🖼️ *Hướng dẫn sử dụng lệnh /image*\n\n"
        "*Cú pháp cơ bản:*\n"
        "`/image [model] prompt`\n\n"
        "*Các model hỗ trợ:*\n"
    )

    for model_key, model_info in IMAGE_CONFIG["models"].items():
        help_text += f"• `{model_key}` - {model_info['display_name']}\n"

    help_text += (
        "\n*Ví dụ:*\n"
        "• `/image a white siamese cat`\n"
        "• `/image flux a futuristic city at night`\n"
        "• `/image dalle a portrait of a viking warrior`\n\n"
        "*Lưu ý:*\n"
        "• Nên sử dụng tiếng Anh để có kết quả tốt nhất\n"
        "• Nếu sử dụng tiếng Việt, bot sẽ tự động dịch sang tiếng Anh\n"
        "• Mỗi người dùng chỉ có thể tạo một hình ảnh tại một thời điểm\n"
        "• Sau khi tạo hình ảnh, bạn có thể:\n"
        "  - Tạo lại hình ảnh với cùng prompt\n"
        "  - Tạo biến thể của hình ảnh\n"
        "  - Thử với model khác\n\n"
        "*Mẹo tạo prompt hiệu quả:*\n"
        "• Mô tả chi tiết những gì bạn muốn thấy\n"
        "• Chỉ định phong cách nghệ thuật (ví dụ: oil painting, digital art, etc.)\n"
        "• Đề cập đến ánh sáng, màu sắc, góc nhìn nếu cần\n"
        "• Sử dụng các từ khóa như 'high quality', 'detailed', 'realistic' để cải thiện chất lượng\n"
    )

    await bot.reply_to(message, help_text, parse_mode="Markdown")


# --- ADMIN COMMANDS ---
@bot.message_handler(commands=["imagestats"])
async def handle_image_stats(message: Message):
    """Hiển thị thống kê về việc sử dụng lệnh /image (chỉ dành cho admin)"""
    # Kiểm tra xem người dùng có phải là admin không
    if message.from_user.id not in [6337636891]:  # Thay YOUR_ADMIN_ID bằng ID của bạn
        await bot.reply_to(message, "⚠️ Bạn không có quyền sử dụng lệnh này.")
        return

    # Tính toán thống kê
    stats = {
        "active_requests": len(active_requests),
        "cache_info": get_cached_image_url.cache_info(),
        "semaphore_value": image_generation_semaphore._value,  # Số slot còn trống
    }

    stats_text = (
        "📊 *Thống kê lệnh /image*\n\n"
        f"• *Yêu cầu đang xử lý:* {stats['active_requests']}/{IMAGE_CONFIG['max_concurrent_generations']}\n"
        f"• *Cache:* {stats['cache_info'].hits} hits, {stats['cache_info'].misses} misses\n"
        f"• *Tỷ lệ cache hit:* {stats['cache_info'].hits/(stats['cache_info'].hits+stats['cache_info'].misses)*100:.1f}% (nếu có)\n"
        f"• *Slot xử lý còn trống:* {stats['semaphore_value']}/{IMAGE_CONFIG['max_concurrent_generations']}\n"
    )

    if active_requests:
        stats_text += "\n*Yêu cầu đang xử lý:*\n"
        for user_id, request in active_requests.items():
            elapsed = time.time() - request["start_time"]
            stats_text += f"• User {user_id}: {request['model']} - '{request['prompt'][:20]}...' ({elapsed:.1f}s)\n"

    await bot.reply_to(message, stats_text, parse_mode="Markdown")


@bot.message_handler(commands=["imageclear"])
async def handle_image_clear(message: Message):
    """Xóa cache và reset trạng thái (chỉ dành cho admin)"""
    # Kiểm tra xem người dùng có phải là admin không
    if message.from_user.id not in [6337636891]:  # Thay YOUR_ADMIN_ID bằng ID của bạn
        await bot.reply_to(message, "⚠️ Bạn không có quyền sử dụng lệnh này.")
        return

    # Xóa cache
    get_cached_image_url.cache_clear()

    # Reset active_requests (cẩn thận với điều này)
    active_requests.clear()

    # Xóa các file tạm
    try:
        for file in os.listdir(IMAGE_CONFIG["temp_dir"]):
            file_path = os.path.join(IMAGE_CONFIG["temp_dir"], file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except Exception as e:
        logger.error(f"Error clearing temp files: {str(e)}")

    await bot.reply_to(
        message,
        "✅ Đã xóa cache và reset trạng thái thành công.",
        parse_mode="Markdown",
    )


async def main():
    logger.info(
        "Bot đang chạy với mô hình Gemini 2.0, bộ nhớ tạm thời và khả năng phân tích hình ảnh..."
    )
    while True:
        try:
            await bot.polling(
                non_stop=True, timeout=60, allowed_updates=["message", "callback_query"]
            )
        except Exception as e:
            logger.error(f"Lỗi polling: {str(e)}")
            await asyncio.sleep(10)


if __name__ == "__main__":
    asyncio.run(main())
