# ./app/bot.py

import random
import pickle
import os
import logging
import traceback
from enum import Enum
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from dotenv import load_dotenv
from data.config import CONFIG
from sklearn.metrics.pairwise import cosine_similarity
from utils import clear_phrase, is_meaningful_text, extract_age, extract_toy_name, extract_toy_category, extract_price, is_age_in_range, Stats, logger
from rapidfuzz import process, fuzz

# Загрузка токена
load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')

# Состояния бота
class BotState(Enum):
    NONE = "NONE"
    WAITING_FOR_TOY = "WAITING_FOR_TOY"
    WAITING_FOR_AGE = "WAITING_FOR_AGE"
    WAITING_FOR_INTENT = "WAITING_FOR_INTENT"

# Намерения
class Intent(Enum):
    HELLO = "hello"
    BYE = "bye"
    YES = "yes"
    NO = "no"
    TOY_TYPES = "toy_types"
    TOY_PRICE = "toy_price"
    TOY_AVAILABILITY = "toy_availability"
    TOY_RECOMMENDATION = "toy_recommendation"
    FILTER_TOYS = "filter_toys"
    TOY_INFO = "toy_info"
    ORDER_TOY = "order_toy"
    COMPARE_TOYS = "compare_toys"

# Типы ответов
class ResponseType(Enum):
    INTENT = "intent"
    GENERATE = "generate"
    FAILURE = "failure"

# Класс бота
class Bot:
    def __init__(self):
        """Инициализация моделей."""
        try:
            with open('models/intent_model.pkl', 'rb') as f:
                self.clf = pickle.load(f)
            with open('models/intent_vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open('models/dialogues_vectorizer.pkl', 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
            with open('models/dialogues_matrix.pkl', 'rb') as f:
                self.tfidf_matrix = pickle.load(f)
            with open('models/dialogues_answers.pkl', 'rb') as f:
                self.answers = pickle.load(f)
        except FileNotFoundError as e:
            logger.error(f"Не найдены файлы модели: {e}\n{traceback.format_exc()}")
            raise

    def _update_context(self, context, replica, answer, intent=None):
        """Обновляет контекст пользователя."""
        context.user_data.setdefault('state', BotState.NONE.value)
        context.user_data.setdefault('current_toy', None)
        context.user_data.setdefault('last_bot_response', None)
        context.user_data.setdefault('last_intent', None)
        context.user_data.setdefault('history', [])

        context.user_data['history'].append(replica)
        context.user_data['history'] = context.user_data['history'][-CONFIG['history_limit']:]
        context.user_data['last_bot_response'] = answer
        if intent:
            context.user_data['last_intent'] = intent

    def classify_intent(self, replica):
        """Классифицирует намерение пользователя."""
        replica = clear_phrase(replica)
        if not replica:
            return None
        vectorized = self.vectorizer.transform([replica])
        intent = self.clf.predict(vectorized)[0]
        best_score = 0
        best_intent = None
        for intent_key, data in CONFIG['intents'].items():
            examples = [clear_phrase(ex) for ex in data.get('examples', []) if clear_phrase(ex)]
            if not examples:
                continue
            match = process.extractOne(replica, examples, scorer=fuzz.ratio)
            if match and match[1] / 100 > best_score and match[1] / 100 >= CONFIG['thresholds']['intent_score']:
                best_score = match[1] / 100
                best_intent = intent_key
        logger.info(f"Classify intent: replica='{replica}', predicted='{intent}', best_intent='{best_intent}', score={best_score}")
        return best_intent or intent if best_score >= CONFIG['thresholds']['intent_score'] else None

    def _get_toy_response(self, intent, toy_name, replica, context):
        """Обрабатывает запросы, связанные с конкретной игрушкой."""
        if toy_name not in CONFIG['toys']:
            return "Извините, такой игрушки нет в каталоге."
        responses = CONFIG['intents'][intent]['responses']
        answer = random.choice(responses)
        toy_data = CONFIG['toys'][toy_name]
        answer = answer.replace('[toy_name]', toy_name)
        answer = answer.replace('[price]', str(toy_data['price']))
        answer = answer.replace('[age]', f"{toy_data['age']['min_age']}-{toy_data['age']['max_age'] or 'и старше'}")
        answer = answer.replace('[description]', toy_data.get('description', 'интересная игрушка'))
        return f"{answer} Что ещё интересует?"

    def _find_toy_by_context(self, replica, context):
        """Ищет игрушку на основе контекста или категории."""
        last_response = context.user_data.get('last_bot_response', '')
        last_intent = context.user_data.get('last_intent', '')
        toy_category = extract_toy_category(replica)

        if last_response and 'Кстати, у нас есть' in last_response:
            return extract_toy_name(last_response)
        elif toy_category:
            suitable_toys = [toy for toy, data in CONFIG['toys'].items() if toy_category in data.get('categories', [])]
            return random.choice(suitable_toys) if suitable_toys else None
        elif last_intent == Intent.TOY_TYPES.value:
            for hist in context.user_data.get('history', [])[::-1]:
                hist_toy = extract_toy_name(hist)
                if hist_toy:
                    return hist_toy
                hist_category = extract_toy_category(hist)
                if hist_category:
                    suitable_toys = [toy for toy, data in CONFIG['toys'].items() if hist_category in data.get('categories', [])]
                    if suitable_toys:
                        return random.choice(suitable_toys)
        return None

    def _handle_filter_toys(self, age, price, toy_category, context):
        """Обрабатывает фильтрацию игрушек по возрасту, цене и категории."""
        suitable_toys = [
            toy for toy, data in CONFIG['toys'].items()
            if (not age or is_age_in_range(age, data['age']))
            and (not price or data['price'] <= price)
            and (not toy_category or toy_category in data.get('categories', []))
        ]
        recent_toys = [extract_toy_name(h) for h in context.user_data.get('history', [])]
        suitable_toys = [t for t in suitable_toys if t not in recent_toys]

        if not suitable_toys:
            conditions = []
            if age:
                conditions.append(f"возраста {age} лет")
            if price:
                conditions.append(f"до {price} рублей")
            if toy_category:
                conditions.append(f"в категории {toy_category}")
            return f"Извините, нет игрушек для {', '.join(conditions)}."

        toys_list = ', '.join(suitable_toys)
        if age and not price and not toy_category:
            toy_name = random.choice(suitable_toys)
            context.user_data['current_toy'] = toy_name
            context.user_data['state'] = BotState.WAITING_FOR_INTENT.value
            return f"Для возраста {age} лет советую {toy_name}! Хотите узнать цену или описание?"
        return f"Вот что нашлось: {toys_list}."

    def get_answer_by_intent(self, intent, replica, context):
        """Генерирует ответ на основе намерения."""
        toy_name = context.user_data.get('current_toy')
        last_intent = context.user_data.get('last_intent', '')
        toy_category = extract_toy_category(replica)
        age = extract_age(replica)
        price = extract_price(replica)

        if intent not in CONFIG['intents']:
            return None
        responses = CONFIG['intents'][intent]['responses']
        if not responses:
            return None
        answer = random.choice(responses)

        if intent in [Intent.TOY_PRICE.value, Intent.TOY_AVAILABILITY.value, Intent.TOY_INFO.value, Intent.ORDER_TOY.value]:
            if not toy_name:
                toy_name = self._find_toy_by_context(replica, context)
                if toy_name:
                    context.user_data['current_toy'] = toy_name
                    context.user_data['state'] = BotState.WAITING_FOR_INTENT.value
                    return f"Из {toy_category or 'игрушек'} есть {toy_name}. Хотите узнать цену, описание или наличие?"
                context.user_data['state'] = BotState.WAITING_FOR_TOY.value
                return "Какую игрушку или категорию вы имеете в виду?"
            return self._get_toy_response(intent, toy_name, replica, context)

        elif intent == Intent.TOY_RECOMMENDATION.value:
            if age:
                answer = self._handle_filter_toys(age, None, toy_category, context)
            else:
                context.user_data['state'] = BotState.WAITING_FOR_AGE.value
                return "Для какого возраста нужна игрушка?"

        elif intent == Intent.FILTER_TOYS.value:
            if age or price or toy_category:
                answer = self._handle_filter_toys(age, price, toy_category, context)
            else:
                return "Укажите возраст, цену или категорию для фильтрации."

        elif intent == Intent.TOY_TYPES.value:
            categories = random.sample([cat for toy in CONFIG['toys'].values() for cat in toy.get('categories', [])], min(3, len(CONFIG['toys'])))
            toys = random.sample(list(CONFIG['toys'].keys()), min(2, len(CONFIG['toys'])))
            answer = f"У нас есть {', '.join(set(categories))} и игрушки вроде {', '.join(toys)}. Что интересно?"
            context.user_data['current_toy'] = None

        elif intent == Intent.COMPARE_TOYS.value:
            toy1 = random.choice(list(CONFIG['toys'].keys()))
            toy2 = random.choice([t for t in CONFIG['toys'].keys() if t != toy1])
            answer = answer.replace('[toy1]', toy1).replace('[toy2]', toy2)
            context.user_data['current_toy'] = toy1
            answer += f" Что интересует: {toy1} или {toy2}?"

        elif intent == Intent.YES.value:
            if last_intent == Intent.HELLO.value:
                categories = random.sample([cat for toy in CONFIG['toys'].values() for cat in toy.get('categories', [])], min(3, len(CONFIG['toys'])))
                answer = f"Отлично! У нас есть {', '.join(set(categories))}. Что хотите узнать?"
            elif last_intent in [Intent.TOY_PRICE.value, Intent.TOY_INFO.value, Intent.TOY_AVAILABILITY.value, Intent.ORDER_TOY.value]:
                if toy_name:
                    answer = f"Цена на {toy_name} — {CONFIG['toys'][toy_name]['price']} рублей. Что ещё интересует?"
                else:
                    answer = "Назови игрушку, чтобы я рассказал подробнее!"
            elif last_intent == Intent.TOY_TYPES.value:
                toys = random.sample(list(CONFIG['toys'].keys()), min(2, len(CONFIG['toys'])))
                answer = f"У нас есть {', '.join(toys)}. Назови одну, чтобы узнать больше!"
            elif last_intent == 'offtopic':
                answer = "Хорошо, давай продолжим! Хочешь узнать про игрушки?"
            else:
                answer = "Хорошо, что интересует? Игрушки, цены или что-то ещё?"

        elif intent == Intent.NO.value:
            context.user_data['current_toy'] = None
            context.user_data['state'] = BotState.NONE.value
            answer = "Хорошо, какую игрушку обсудим теперь?"

        if intent in [Intent.HELLO.value, Intent.TOY_TYPES.value] and random.random() < 0.2:
            ad_toy = random.choice([t for t in CONFIG['toys'].keys() if t != toy_name])
            answer += f" Кстати, у нас есть {ad_toy} — отличный выбор для детей {CONFIG['toys'][ad_toy]['age']['min_age']}-{CONFIG['toys'][ad_toy]['age']['max_age'] or 'и старше'}!"

        context.user_data['last_intent'] = intent
        return answer

    def generate_answer(self, replica, context):
        """Генерирует ответ на основе диалогов."""
        replica = clear_phrase(replica)
        if not replica or not self.answers:
            return None
        if not is_meaningful_text(replica):
            return None
        replica_vector = self.tfidf_vectorizer.transform([replica])
        similarities = cosine_similarity(replica_vector, self.tfidf_matrix).flatten()
        best_idx = similarities.argmax()
        if similarities[best_idx] > CONFIG['thresholds']['dialogues_similarity']:
            answer = self.answers[best_idx]
            logger.info(f"Found in dialogues.txt: replica='{replica}', answer='{answer}', similarity={similarities[best_idx]}")
            if random.random() < 0.3:
                ad_toy = random.choice(list(CONFIG['toys'].keys()))
                answer += f" Кстати, у нас есть {ad_toy} — отличный выбор для детей {CONFIG['toys'][ad_toy]['age']['min_age']}-{CONFIG['toys'][ad_toy]['age']['max_age'] or 'и старше'}!"
            context.user_data['last_intent'] = 'offtopic'
            return answer
        logger.info(f"No match in dialogues.txt for replica='{replica}'")
        return None

    def get_failure_phrase(self):
        """Возвращает фразу при неудачном запросе."""
        toy_name = random.choice(list(CONFIG['toys'].keys()))
        return random.choice(CONFIG['failure_phrases']).replace('[toy_name]', toy_name)

    def _process_none_state(self, replica, context):
        """Обрабатывает состояние NONE."""
        toy_name = extract_toy_name(replica)
        if toy_name:
            context.user_data['current_toy'] = toy_name
            context.user_data['state'] = BotState.WAITING_FOR_INTENT.value
            return f"Вы имеете в виду {toy_name}? Хотите узнать цену, описание или наличие?"

        toy_category = extract_toy_category(replica)
        if toy_category:
            suitable_toys = [toy for toy, data in CONFIG['toys'].items() if toy_category in data.get('categories', [])]
            if suitable_toys:
                toy_name = random.choice(suitable_toys)
                context.user_data['current_toy'] = toy_name
                context.user_data['state'] = BotState.WAITING_FOR_INTENT.value
                return f"Из {toy_category} есть {toy_name}. Хотите узнать цену, описание или наличие?"
            return f"У нас нет игрушек в категории {toy_category}. Попробуйте другую категорию!"

        intent = self.classify_intent(replica)
        if intent:
            return self.get_answer_by_intent(intent, replica, context)

        return self.generate_answer(replica, context) or self.get_failure_phrase()

    def _process_waiting_for_toy(self, replica, context):
        """Обрабатывает состояние WAITING_FOR_TOY."""
        toy_name = extract_toy_name(replica)
        if toy_name:
            context.user_data['current_toy'] = toy_name
            context.user_data['state'] = BotState.WAITING_FOR_INTENT.value
            return f"Вы имеете в виду {toy_name}? Хотите узнать цену, описание или наличие?"
        toy_category = extract_toy_category(replica)
        if toy_category:
            suitable_toys = [toy for toy, data in CONFIG['toys'].items() if toy_category in data.get('categories', [])]
            if suitable_toys:
                toy_name = random.choice(suitable_toys)
                context.user_data['current_toy'] = toy_name
                context.user_data['state'] = BotState.WAITING_FOR_INTENT.value
                return f"Из {toy_category} есть {toy_name}. Хотите узнать цену, описание или наличие?"
        return "Пожалуйста, уточните название игрушки или категорию."

    def _process_waiting_for_age(self, replica, context):
        """Обрабатывает состояние WAITING_FOR_AGE."""
        age = extract_age(replica)
        if age:
            context.user_data['state'] = BotState.NONE.value
            return self._handle_filter_toys(age, None, None, context)
        return "Укажите возраст, например, '5 лет'."

    def _process_waiting_for_intent(self, replica, context):
        """Обрабатывает состояние WAITING_FOR_INTENT."""
        intent = self.classify_intent(replica)
        toy_name = context.user_data.get('current_toy', 'игрушку')
        if intent in [Intent.TOY_PRICE.value, Intent.TOY_AVAILABILITY.value, Intent.TOY_INFO.value, Intent.ORDER_TOY.value]:
            context.user_data['state'] = BotState.NONE.value
            return self.get_answer_by_intent(intent, replica, context)
        if intent == Intent.YES.value:
            if toy_name:
                context.user_data['state'] = BotState.NONE.value
                return f"Цена на {toy_name} — {CONFIG['toys'][toy_name]['price']} рублей. Что ещё интересует?"
        if intent == Intent.NO.value:
            context.user_data['current_toy'] = None
            context.user_data['state'] = BotState.NONE.value
            return "Хорошо, какую игрушку обсудим теперь?"
        return f"Что хотите узнать про {toy_name}: цену, описание или наличие?"

    def process(self, replica, context):
        """Обрабатывает запрос пользователя."""
        stats = Stats(context)
        if not is_meaningful_text(replica):
            answer = self.get_failure_phrase()
            self._update_context(context, replica, answer)
            stats.add(ResponseType.FAILURE.value, replica, answer, context)
            return answer

        age = extract_age(replica)
        price = extract_price(replica)
        toy_category = extract_toy_category(replica)
        if age or price:
            answer = self._handle_filter_toys(age, price, toy_category, context)
            self._update_context(context, replica, answer, Intent.FILTER_TOYS.value)
            stats.add(ResponseType.INTENT.value, replica, answer, context)
            return answer

        state = context.user_data.get('state', BotState.NONE.value)
        logger.info(f"Processing: replica='{replica}', state='{state}', last_intent='{context.user_data.get('last_intent')}'")

        if state == BotState.WAITING_FOR_TOY.value:
            answer = self._process_waiting_for_toy(replica, context)
        elif state == BotState.WAITING_FOR_AGE.value:
            answer = self._process_waiting_for_age(replica, context)
        elif state == BotState.WAITING_FOR_INTENT.value:
            answer = self._process_waiting_for_intent(replica, context)
        else:
            answer = self._process_none_state(replica, context)

        self._update_context(context, replica, answer)
        stats.add(ResponseType.INTENT.value if self.classify_intent(replica) else ResponseType.GENERATE.value if 'dialogues.txt' in answer else ResponseType.FAILURE.value, replica, answer, context)
        return answer

# Голос в текст
def voice_to_text(voice_file):
    recognizer = sr.Recognizer()
    try:
        import signal
        def signal_handler(signum, frame):
            raise TimeoutError("Speech recognition timed out")
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(5)  # Таймаут 5 секунд
        audio = AudioSegment.from_ogg(voice_file)
        audio.export('voice.wav', format='wav')
        with sr.AudioFile('voice.wav') as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language='ru-RU')
        return text
    except (sr.UnknownValueError, sr.RequestError, TimeoutError, Exception) as e:
        logger.error(f"Ошибка распознавания голоса: {e}\n{traceback.format_exc()}")
        return None
    finally:
        signal.alarm(0)
        if os.path.exists('voice.wav'):
            os.remove('voice.wav')

# Текст в голос
def text_to_voice(text):
    if not text:
        return None
    try:
        tts = gTTS(text=text, lang='ru')
        voice_file = 'response.mp3'
        tts.save(voice_file)
        return voice_file
    except Exception as e:
        logger.error(f"Ошибка синтеза речи: {e}\n{traceback.format_exc()}")
        return None

# Telegram-обработчики
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    answer = CONFIG['start_message']
    context.user_data['last_bot_response'] = answer
    context.user_data['last_intent'] = Intent.HELLO.value
    await update.message.reply_text(answer)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    answer = CONFIG['help_message']
    context.user_data['last_bot_response'] = answer
    context.user_data['last_intent'] = 'help'
    await update.message.reply_text(answer)

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    stats = context.user_data.get('stats', {ResponseType.INTENT.value: 0, ResponseType.GENERATE.value: 0, ResponseType.FAILURE.value: 0})
    answer = (
        f"Статистика:\n"
        f"Обработано намерений: {stats[ResponseType.INTENT.value]}\n"
        f"Ответов из диалогов: {stats[ResponseType.GENERATE.value]}\n"
        f"Неудачных запросов: {stats[ResponseType.FAILURE.value]}"
    )
    await update.message.reply_text(answer)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    if not user_text:
        answer = "Пожалуйста, отправьте текст."
        context.user_data['last_bot_response'] = answer
        await update.message.reply_text(answer)
        return
    bot = context.bot_data.setdefault('bot', Bot())
    answer = bot.process(user_text, context)
    await update.message.reply_text(answer)

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    voice = update.message.voice
    bot = context.bot_data.setdefault('bot', Bot())
    try:
        voice_file = await context.bot.get_file(voice.file_id)
        await voice_file.download_to_drive('voice.ogg')
        text = voice_to_text('voice.ogg')
        if text:
            answer = bot.process(text, context)
            voice_response = text_to_voice(answer)
            if voice_response:
                with open(voice_response, 'rb') as audio:
                    await update.message.reply_voice(audio)
                os.remove(voice_response)
            else:
                await update.message.reply_text(answer)
        else:
            answer = "Не удалось распознать голос. Попробуйте ещё раз."
            context.user_data['last_bot_response'] = answer
            await update.message.reply_text(answer)
    except Exception as e:
        logger.error(f"Ошибка обработки голосового сообщения: {e}\n{traceback.format_exc()}")
        answer = "Произошла ошибка. Попробуйте снова."
        context.user_data['last_bot_response'] = answer
        await update.message.reply_text(answer)
    finally:
        if os.path.exists('voice.ogg'):
            os.remove('voice.ogg')

def run_bot():
    if not TOKEN:
        raise ValueError("TELEGRAM_TOKEN не найден")
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    logger.info("Бот запускается...")
    app.run_polling()

if __name__ == '__main__':
    run_bot()
    