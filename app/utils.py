# ./app/utils.py

import logging
import nltk
from rapidfuzz import process, fuzz
from data.config import CONFIG

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Очистка фразы
def clear_phrase(phrase):
    if not phrase:
        return ""
    phrase = phrase.lower()
    alphabet = '1234567890qwertyuiopasdfghjklzxcvbnmабвгдеёжзийклмнопрстуфхцчшщъыьэюя- '
    return ''.join(symbol for symbol in phrase if symbol in alphabet).strip()

# Проверка на осмысленность текста
def is_meaningful_text(text):
    text = clear_phrase(text)
    words = text.split()
    return any(len(word) > 2 and all(c in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя' for c in word) for word in words)

# Извлечение возраста
def extract_age(replica):
    replica = clear_phrase(replica)
    logger.info(f"Extracting age from: '{replica}'")
    words = replica.split()
    for i, word in enumerate(words):
        if word.isdigit() and (i + 1 < len(words) and words[i + 1] in ['год', 'года', 'лет'] or 'для' in words[:i]):
            logger.info(f"Found age: {word}")
            return word
    logger.info("Age not found")
    return None

# Извлечение цены
def extract_price(replica):
    replica = clear_phrase(replica)
    logger.info(f"Extracting price from: '{replica}'")
    if not replica:
        return None
    words = replica.split()
    for i, word in enumerate(words):
        if word.isdigit() and (i + 1 < len(words) and words[i + 1] in ['рублей', 'руб'] or 'до' in words[:i] or 'дешевле' in words[:i]):
            logger.info(f"Found price: {word}")
            return int(word)
    logger.info("Price not found")
    return None

# Извлечение игрушки
def extract_toy_name(replica):
    replica = clear_phrase(replica)
    if not replica:
        return None
    for toy, data in CONFIG['toys'].items():
        if toy.lower() in replica or any(syn.lower() in replica for syn in data.get('synonyms', [])):
            return toy
        candidates = [toy] + data.get('synonyms', [])
        best_match = process.extractOne(replica, candidates, scorer=fuzz.partial_ratio)
        if best_match and best_match[1] > CONFIG['thresholds']['fuzzy_match_toy']:
            return toy
    return None

# Извлечение категории
def extract_toy_category(replica):
    replica = clear_phrase(replica)
    if not replica:
        return None
    for toy, data in CONFIG['toys'].items():
        for category in data.get('categories', []):
            if category.lower() in replica or any(syn.lower() in replica for syn in data.get('category_synonyms', {}).get(category, [])):
                return category
    return None

# Проверка возраста в диапазоне
def is_age_in_range(age, age_range):
    try:
        age = int(age)
        min_age = age_range['min_age']
        max_age = age_range['max_age']
        return min_age <= age <= (max_age if max_age is not None else float('inf'))
    except (ValueError, TypeError):
        return False

# Класс для управления статистикой
class Stats:
    def __init__(self, context):
        self.context = context
        if 'stats' not in context.user_data:
            context.user_data['stats'] = {'intent': 0, 'generate': 0, 'failure': 0}
        self.stats = context.user_data['stats']

    def add(self, type, replica, answer, context):
        """Обновляет статистику, сохраняет её в context и логирует."""
        if type in self.stats:
            self.stats[type] += 1
        else:
            self.stats[type] = 1
        self.context.user_data['stats'] = self.stats
        logger.info(f"Stats: {self.stats} | Вопрос: {replica} | Ответ: {answer}")
        