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
    alphabet = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя- '
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
        if word.isdigit():
            logger.info(f"Found age: {word}")
            return word
        elif word in ['год', 'года', 'лет']:
            if i > 0 and words[i - 1].isdigit():
                logger.info(f"Found age: {words[i - 1]}")
                return words[i - 1]
    logger.info("Age not found")
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
        if best_match and best_match[1] > 85:
            return toy
    return None

# Извлечение категории
def extract_toy_category(replica):
    replica = clear_phrase(replica)
    if not replica:
        return None
    for category in CONFIG['categories']:
        category_variants = [category, category + 'ы', category[:-1] + 'ая', category[:-1] + 'и']
        for variant in category_variants:
            if variant.lower() in replica:
                return category
    return None

# Извлечение цены
def extract_price(replica):
    replica = clear_phrase(replica)
    if not replica:
        return None
    words = replica.split()
    for word in words:
        if word.isdigit():
            return int(word)
    return None

# Проверка возраста в диапазоне
def is_age_in_range(age, age_range):
    try:
        age = int(age)
        if '-' in age_range:
            min_age, max_age = age_range.split('-')
            min_age = int(min_age)
            max_age = int(max_age) if max_age.isdigit() else 100
            return min_age <= age <= max_age
        elif '+' in age_range:
            min_age = int(age_range.replace('+', ''))
            return age >= min_age
        else:
            return age == int(age_range)
    except ValueError:
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