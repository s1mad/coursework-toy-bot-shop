# ./app/bot.py

import random
import nltk
import pickle
import os
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from dotenv import load_dotenv
from data.config import CONFIG
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import clear_phrase, is_meaningful_text, extract_age, extract_toy_name, extract_toy_category, extract_price, is_age_in_range, Stats, logger

# Загрузка токена
load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')

# Загрузка модели для намерений
try:
    with open('models/intent_model.pkl', 'rb') as f:
        clf = pickle.load(f)
    with open('models/intent_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except FileNotFoundError as e:
    logger.error(f"Не найдены файлы модели для намерений: {e}")
    raise

# Загрузка модели для dialogues.txt
try:
    with open('models/dialogues_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    with open('models/dialogues_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)
    with open('models/dialogues_answers.pkl', 'rb') as f:
        answers = pickle.load(f)
except FileNotFoundError as e:
    logger.error(f"Не найдены файлы модели для dialogues.txt: {e}")
    raise

# Классификация намерения
def classify_intent(replica):
    replica = clear_phrase(replica)
    if not replica:
        return None
    vectorized = vectorizer.transform([replica])
    intent = clf.predict(vectorized)[0]
    best_score = 0
    best_intent = None
    for intent_key, data in CONFIG['intents'].items():
        for example in data.get('examples', []):
            example = clear_phrase(example)
            if not example:
                continue
            distance = nltk.edit_distance(replica, example)
            score = 1 - distance / max(len(example), 1)
            if score > best_score and score >= 0.65:
                best_score = score
                best_intent = intent_key
    logger.info(f"Classify intent: replica='{replica}', predicted='{intent}', best_intent='{best_intent}', score={best_score}")
    return best_intent or intent if best_score >= 0.65 else None

# Получение ответа
def get_answer_by_intent(intent, replica, context):
    toy_name = context.user_data.get('current_toy')
    last_response = context.user_data.get('last_bot_response', '')
    last_intent = context.user_data.get('last_intent', '')
    toy_category = extract_toy_category(replica)
    age = extract_age(replica)
    price = extract_price(replica)

    if intent in CONFIG['intents']:
        responses = CONFIG['intents'][intent]['responses']
        if not responses:
            return None
        answer = random.choice(responses)

        if intent in ['toy_price', 'toy_availability', 'toy_info', 'order_toy']:
            if not toy_name:
                if last_response and 'Кстати, у нас есть' in last_response:
                    toy_name = extract_toy_name(last_response)
                    context.user_data['current_toy'] = toy_name
                elif toy_category:
                    suitable_toys = [toy for toy, data in CONFIG['toys'].items() if toy_category in data.get('categories', [])]
                    if suitable_toys:
                        toy_name = random.choice(suitable_toys)
                        context.user_data['current_toy'] = toy_name
                        context.user_data['state'] = 'WAITING_FOR_INTENT'
                        return f"Из {toy_category} есть {toy_name}. Хотите узнать цену, описание или наличие?"
                elif last_intent == 'toy_types':
                    for hist in context.user_data.get('history', [])[::-1]:
                        hist_toy = extract_toy_name(hist)
                        if hist_toy:
                            toy_name = hist_toy
                            context.user_data['current_toy'] = toy_name
                            break
                        hist_category = extract_toy_category(hist)
                        if hist_category:
                            suitable_toys = [toy for toy, data in CONFIG['toys'].items() if hist_category in data.get('categories', [])]
                            if suitable_toys:
                                toy_name = random.choice(suitable_toys)
                                context.user_data['current_toy'] = toy_name
                                break
                if not toy_name:
                    context.user_data['state'] = 'WAITING_FOR_TOY'
                    return "Какую игрушку или категорию вы имеете в виду?"
            if toy_name in CONFIG['toys']:
                answer = answer.replace('[toy_name]', toy_name)
                answer = answer.replace('[price]', str(CONFIG['toys'][toy_name]['price']))
                answer = answer.replace('[age]', CONFIG['toys'][toy_name]['age'])
                answer = answer.replace('[description]', CONFIG['toys'][toy_name].get('description', 'интересная игрушка'))
                answer += f" Что ещё интересует?"
            else:
                return "Извините, такой игрушки нет в каталоге."

        elif intent == 'toy_recommendation':
            if age:
                suitable_toys = [toy for toy, data in CONFIG['toys'].items() if is_age_in_range(age, data['age'])]
                if suitable_toys:
                    toy_name = random.choice(suitable_toys)
                    context.user_data['current_toy'] = toy_name
                    answer = answer.replace('[toy_name]', toy_name).replace('[age]', age)
                    answer += f" Хотите узнать цену или описание {toy_name}?"
                else:
                    return f"Извините, у нас нет игрушек для возраста {age} лет."
            else:
                context.user_data['state'] = 'WAITING_FOR_AGE'
                return "Для какого возраста нужна игрушка?"

        elif intent == 'toy_types':
            categories = random.sample(CONFIG['categories'], min(3, len(CONFIG['categories'])))
            toys = random.sample(list(CONFIG['toys'].keys()), min(2, len(CONFIG['toys'])))
            answer = f"У нас есть {', '.join(categories)} и игрушки вроде {', '.join(toys)}. Что интересно?"
            context.user_data['current_toy'] = None

        elif intent == 'compare_toys':
            toy1 = random.choice(list(CONFIG['toys'].keys()))
            toy2 = random.choice([t for t in CONFIG['toys'].keys() if t != toy1])
            answer = answer.replace('[toy1]', toy1).replace('[toy2]', toy2)
            context.user_data['current_toy'] = toy1
            answer += f" Что интересует: {toy1} или {toy2}?"

        elif intent == 'yes':
            if last_intent == 'hello':
                categories = random.sample(CONFIG['categories'], min(3, len(CONFIG['categories'])))
                answer = f"Отлично! У нас есть {', '.join(categories)}. Что хотите узнать?"
            elif last_intent in ['toy_price', 'toy_info', 'toy_availability', 'order_toy']:
                if toy_name:
                    answer = f"Цена на {toy_name} — {CONFIG['toys'][toy_name]['price']} рублей. Что ещё интересует?"
                else:
                    answer = "Назови игрушку, чтобы я рассказал подробнее!"
            elif last_intent == 'toy_types':
                toys = random.sample(list(CONFIG['toys'].keys()), min(2, len(CONFIG['toys'])))
                answer = f"У нас есть {', '.join(toys)}. Назови одну, чтобы узнать больше!"
            elif last_intent == 'offtopic':
                answer = "Хорошо, давай продолжим! Хочешь узнать про игрушки?"
            else:
                answer = "Хорошо, что интересует? Игрушки, цены или что-то ещё?"

        elif intent == 'no':
            context.user_data['current_toy'] = None
            context.user_data['state'] = 'NONE'
            answer = "Хорошо, какую игрушку обсудим теперь?"

        elif intent == 'filter_toys':
            if price and age:
                suitable_toys = [toy for toy, data in CONFIG['toys'].items() if data['price'] <= price and is_age_in_range(age, data['age'])]
                if suitable_toys:
                    toys_list = ', '.join(suitable_toys)
                    answer = f"Для возраста {age} лет и до {price} рублей есть: {toys_list}."
                else:
                    answer = f"Извините, нет игрушек для возраста {age} лет и до {price} рублей."
            elif price:
                suitable_toys = [toy for toy, data in CONFIG['toys'].items() if data['price'] <= price]
                if suitable_toys:
                    toys_list = ', '.join(suitable_toys)
                    answer = f"До {price} рублей есть: {toys_list}."
                else:
                    answer = f"Извините, нет игрушек до {price} рублей."
            elif age:
                suitable_toys = [toy for toy, data in CONFIG['toys'].items() if is_age_in_range(age, data['age'])]
                if suitable_toys:
                    toys_list = ', '.join(suitable_toys)
                    answer = f"Для возраста {age} лет есть: {toys_list}."
                    context.user_data['current_toy'] = random.choice(suitable_toys)
                    context.user_data['state'] = 'WAITING_FOR_INTENT'
                else:
                    answer = f"Извините, нет игрушек для возраста {age} лет."
            else:
                answer = "Укажите возраст или цену для фильтрации."

        # Реклама
        if intent in ['hello', 'toy_types'] and random.random() < 0.2:
            ad_toy = random.choice([t for t in CONFIG['toys'].keys() if t != toy_name])
            answer += f" Кстати, у нас есть {ad_toy} — отличный выбор для детей {CONFIG['toys'][ad_toy]['age']}!"

        context.user_data['last_intent'] = intent
        return answer
    return None

# Ответ из dialogues.txt с TF-IDF
def generate_answer(replica, context):
    replica = clear_phrase(replica)
    if not replica or not answers:
        return None
    if not is_meaningful_text(replica):
        return None
    replica_vector = tfidf_vectorizer.transform([replica])
    similarities = cosine_similarity(replica_vector, tfidf_matrix).flatten()
    best_idx = similarities.argmax()
    if similarities[best_idx] > 0.5:
        answer = answers[best_idx]
        logger.info(f"Found in dialogues.txt: replica='{replica}', answer='{answer}', similarity={similarities[best_idx]}")
        if random.random() < 0.3:
            ad_toy = random.choice(list(CONFIG['toys'].keys()))
            answer += f" Кстати, у нас есть {ad_toy} — отличный выбор для детей {CONFIG['toys'][ad_toy]['age']}!"
        context.user_data['last_intent'] = 'offtopic'
        return answer
    logger.info(f"No match in dialogues.txt for replica='{replica}'")
    return None

# Заглушка
def get_failure_phrase():
    toy_name = random.choice(list(CONFIG['toys'].keys()))
    return random.choice(CONFIG['failure_phrases']).replace('[toy_name]', toy_name)

# Основная логика
def bot(replica, context):
    stats = Stats(context)
    if 'state' not in context.user_data:
        context.user_data['state'] = 'NONE'
    if 'current_toy' not in context.user_data:
        context.user_data['current_toy'] = None
    if 'last_bot_response' not in context.user_data:
        context.user_data['last_bot_response'] = None
    if 'last_intent' not in context.user_data:
        context.user_data['last_intent'] = None
    if 'history' not in context.user_data:
        context.user_data['history'] = []

    context.user_data['history'].append(replica)
    context.user_data['history'] = context.user_data['history'][-5:]

    state = context.user_data['state']
    logger.info(f"Processing: replica='{replica}', state='{state}', last_intent='{context.user_data.get('last_intent')}'")

    # Проверка на несуразный текст
    if not is_meaningful_text(replica):
        context.user_data['state'] = 'NONE'
        context.user_data['current_toy'] = None
        answer = get_failure_phrase()
        context.user_data['last_bot_response'] = answer
        stats.add('failure', replica, answer, context)
        return answer

    # Проверка возраста и цены
    age = extract_age(replica)
    price = extract_price(replica)
    if age or price:
        intent = 'filter_toys'
        answer = get_answer_by_intent(intent, replica, context)
        if answer:
            context.user_data['last_bot_response'] = answer
            stats.add('intent', replica, answer, context)
            return answer

    # Обработка состояния
    if state == 'WAITING_FOR_TOY':
        toy_name = extract_toy_name(replica)
        if toy_name:
            context.user_data['current_toy'] = toy_name
            context.user_data['state'] = 'WAITING_FOR_INTENT'
            answer = f"Вы имеете в виду {toy_name}? Хотите узнать цену, описание или наличие?"
            context.user_data['last_bot_response'] = answer
            stats.add('intent', replica, answer, context)
            return answer
        toy_category = extract_toy_category(replica)
        if toy_category:
            suitable_toys = [toy for toy, data in CONFIG['toys'].items() if toy_category in data.get('categories', [])]
            if suitable_toys:
                toy_name = random.choice(suitable_toys)
                context.user_data['current_toy'] = toy_name
                context.user_data['state'] = 'WAITING_FOR_INTENT'
                answer = f"Из {toy_category} есть {toy_name}. Хотите узнать цену, описание или наличие?"
                context.user_data['last_bot_response'] = answer
                stats.add('intent', replica, answer, context)
                return answer
        answer = "Пожалуйста, уточните название игрушки или категорию."
        context.user_data['last_bot_response'] = answer
        stats.add('failure', replica, answer, context)
        return answer

    if state == 'WAITING_FOR_AGE':
        if age:
            context.user_data['state'] = 'NONE'
            suitable_toys = [toy for toy, data in CONFIG['toys'].items() if is_age_in_range(age, data['age'])]
            if suitable_toys:
                toy_name = random.choice(suitable_toys)
                context.user_data['current_toy'] = toy_name
                answer = f"Для возраста {age} лет советую {toy_name}! Хотите узнать цену или описание?"
                context.user_data['last_bot_response'] = answer
                stats.add('intent', replica, answer, context)
                return answer
            answer = f"Извините, нет игрушек для возраста {age} лет. Попробуйте другой возраст."
            context.user_data['last_bot_response'] = answer
            stats.add('failure', replica, answer, context)
            return answer
        answer = "Укажите возраст, например, '5 лет'."
        context.user_data['last_bot_response'] = answer
        stats.add('failure', replica, answer, context)
        return answer

    if state == 'WAITING_FOR_INTENT':
        intent = classify_intent(replica)
        if intent in ['toy_price', 'toy_availability', 'toy_info', 'order_toy']:
            context.user_data['state'] = 'NONE'
            answer = get_answer_by_intent(intent, replica, context)
            if answer:
                context.user_data['last_bot_response'] = answer
                stats.add('intent', replica, answer, context)
                return answer
        if intent == 'yes':
            toy_name = context.user_data.get('current_toy')
            if toy_name:
                context.user_data['state'] = 'NONE'
                answer = f"Цена на {toy_name} — {CONFIG['toys'][toy_name]['price']} рублей. Что ещё интересует?"
                context.user_data['last_bot_response'] = answer
                stats.add('intent', replica, answer, context)
                return answer
        if intent == 'no':
            context.user_data['current_toy'] = None
            context.user_data['state'] = 'NONE'
            answer = "Хорошо, какую игрушку обсудим теперь?"
            context.user_data['last_bot_response'] = answer
            stats.add('intent', replica, answer, context)
            return answer
        toy_name = context.user_data.get('current_toy', 'игрушку')
        answer = f"Что хотите узнать про {toy_name}: цену, описание или наличие?"
        context.user_data['last_bot_response'] = answer
        stats.add('failure', replica, answer, context)
        return answer

    # Проверка игрушки
    toy_name = extract_toy_name(replica)
    if toy_name:
        context.user_data['current_toy'] = toy_name
        context.user_data['state'] = 'WAITING_FOR_INTENT'
        answer = f"Вы имеете в виду {toy_name}? Хотите узнать цену, описание или наличие?"
        context.user_data['last_bot_response'] = answer
        stats.add('intent', replica, answer, context)
        return answer

    # Проверка категории
    toy_category = extract_toy_category(replica)
    if toy_category:
        suitable_toys = [toy for toy, data in CONFIG['toys'].items() if toy_category in data.get('categories', [])]
        if suitable_toys:
            toy_name = random.choice(suitable_toys)
            context.user_data['current_toy'] = toy_name
            context.user_data['state'] = 'WAITING_FOR_INTENT'
            answer = f"Из {toy_category} есть {toy_name}. Хотите узнать цену, описание или наличие?"
            context.user_data['last_bot_response'] = answer
            stats.add('intent', replica, answer, context)
            return answer
        answer = f"У нас нет игрушек в категории {toy_category}. Попробуйте другую категорию!"
        context.user_data['last_bot_response'] = answer
        stats.add('failure', replica, answer, context)
        return answer

    # Классификация намерения
    intent = classify_intent(replica)
    if intent:
        answer = get_answer_by_intent(intent, replica, context)
        if answer:
            context.user_data['last_bot_response'] = answer
            stats.add('intent', replica, answer, context)
            return answer

    # dialogues.txt для отвлечённых тем
    answer = generate_answer(replica, context)
    if answer:
        context.user_data['last_bot_response'] = answer
        stats.add('generate', replica, answer, context)
        return answer

    # Заглушка как последний вариант
    answer = get_failure_phrase()
    context.user_data['last_bot_response'] = answer
    stats.add('failure', replica, answer, context)
    return answer

# Голос в текст
def voice_to_text(voice_file):
    recognizer = sr.Recognizer()
    try:
        audio = AudioSegment.from_ogg(voice_file)
        audio.export('voice.wav', format='wav')
        with sr.AudioFile('voice.wav') as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data, language='ru-RU')
    except (sr.UnknownValueError, sr.RequestError, Exception) as e:
        logger.error(f"Ошибка распознавания голоса: {e}")
        return None
    finally:
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
        logger.error(f"Ошибка синтеза речи: {e}")
        return None

# Telegram-обработчики
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    answer = CONFIG['start_message']
    context.user_data['last_bot_response'] = answer
    context.user_data['last_intent'] = 'hello'
    await update.message.reply_text(answer)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    answer = CONFIG['help_message']
    context.user_data['last_bot_response'] = answer
    context.user_data['last_intent'] = 'help'
    await update.message.reply_text(answer)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    if not user_text:
        answer = "Пожалуйста, отправьте текст."
        context.user_data['last_bot_response'] = answer
        await update.message.reply_text(answer)
        return
    answer = bot(user_text, context)
    await update.message.reply_text(answer)

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    voice = update.message.voice
    try:
        voice_file = await context.bot.get_file(voice.file_id)
        await voice_file.download_to_drive('voice.ogg')
        text = voice_to_text('voice.ogg')
        if text:
            answer = bot(text, context)
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
        logger.error(f"Ошибка обработки голосового сообщения: {e}")
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
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    logger.info("Бот запускается...")
    app.run_polling()

if __name__ == '__main__':
    run_bot()
    