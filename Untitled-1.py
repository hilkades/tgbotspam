
import telebot
from telebot import types
import re
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BOT_TOKEN = '7823075669:AAHm8tuESuo2HrxSZTBMb5jOF5HEvVMSJh8'
bot = telebot.TeleBot(BOT_TOKEN)

# --- Состояния пользователя ---
USER_STATES = {}
CHANNEL_STATES = {}

STATE_START = 0
STATE_CHOOSE_FEATURE = 1
STATE_CHOOSE_PLAN = 2
STATE_CHOOSE_PAYMENT = 3
STATE_INPUT_CHANNEL = 4
STATE_ASK_THEME = 5
STATE_ASK_RULES = 6
STATE_CHOOSE_STOP_WORD_ACTION = 7
STATE_CHECK_CONNECTION = 8
STATE_ACTIVE = 9
STATE_RECONFIGURE = 10

# --- Данные для хранения информации о выборе пользователя ---
USER_DATA = {}

# --- Состояние активации бота для каждого канала ---
BOT_ACTIVE = {}

# --- Статистика ---
CHAT_STATISTICS = {}
LAST_STATISTICS_REQUEST = {}
STATISTICS_COOLDOWN = 600 # 10 минут в секундах

WELCOME_MESSAGE = """
Приветствую нового участника! 👋

{theme_line}

{rules_line}

Приятного времяпрепровождения! 😉
"""
START_MESSAGE = """
Всем привет! 👋

Я бот, который поможет вам автоматизировать работу с этим каналом.

Сейчас я буду приветствовать всех новых участников!
"""

# Словарь для хранения информации о пользователях, ожидающих подтверждения
pending_users = {}
TIMEOUT_SECONDS = 300  # 5 минут в секундах

# --- Загрузка модели машинного обучения ---
MODEL_NAME = "GroNLP/bert-base-dutch-cased"  # Замените на имя вашей модели - Улучшено
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()  # Переводим модель в режим оценки (inference)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Переносим модель на GPU, если доступно - Улучшено
model.to(device)

# Пороговое значение уверенности для классификации спама - Улучшено
SPAM_THRESHOLD = 0.85

# --- Обучающие примеры для дообучения модели ---
TRAINING_EXAMPLES = [
    ("Набираю людей для удалённого сотрудничества с возможностью заработка от $500 в неделю. Гибкий график, неполная занятость, всё официально. Требования: от 18 лет и желание зарабатывать. Пиши в ЛС за подробностями! 🚀", 1),  # Спам
    ("Приветствую. Ищу людей для доп заработка на удалённой основе в свободное время, все подробности в личные сообщения", 1),  # Спам
    ("Привет, Требуется 5 человек, дocтoйный зapaбoтok, бeрём бeз oпытa! 🖥 Пишите + в ЛС", 1),  # Спам
    ("Доход от 500 долларов в неделю. Без опыта, свободный график. Пишите плюс в личные сообщения, расскажу подробности", 1),  # Спам
    ("Доход от 500 долларов в неделю. Гибкий график, онлайн. Пишите в личные сообщения", 1),  # Спам
    ("Требуются сотрудники для онлайн сотрудничества! 100-150$ в день, всего 1-2 часа твоего времемни Пишите + в личку!", 1), # Спам
    ("Интересная статья о новых технологиях в сфере AI.", 0), # Не спам
    ("Обсуждаем последние новости киноиндустрии.", 0) # Не спам
]


def finetune_model(model, tokenizer, examples, device, epochs=3):
    """Дообучает модель на предоставленных примерах."""
    model.train()  # Переводим модель в режим обучения
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)  # Используем AdamW
    model.to(device) # Ensure model is on the correct device

    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}/{epochs}")
        for text, label in examples:
            optimizer.zero_grad()  # Обнуляем градиенты
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device) # Tokenize and send to device
            labels = torch.tensor([label]).to(device)  # Send labels to device

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()  # Вычисляем градиенты
            optimizer.step()  # Делаем шаг оптимизации

    model.eval()  # Переводим обратно в режим оценки
    print("Модель дообучена.")

# Дообучаем модель - Улучшено
print("Начинаем дообучение модели...")
finetune_model(model, tokenizer, TRAINING_EXAMPLES, device)
print("Дообучение завершено.")

def is_telegram_link(url):
    """Проверяет, является ли строка ссылкой на телеграм-канал (включая @username)."""
    pattern = r"^(@[a-zA-Z0-9_]+)|(https?://(t\.me|telegram\.me)/[a-zA-Z0-9_]+)$"
    return bool(re.match(pattern, url))

def convert_to_telegram_link(username):
    """Преобразует @username в ссылку t.me."""
    return f"https://t.me/{username.lstrip('@')}"

@bot.message_handler(commands=['start'])
def start(message):
    """Обработчик команды /start."""
    user_id = message.from_user.id
    USER_STATES[user_id] = STATE_START
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    item_autopost = types.KeyboardButton("Автопостинг")
    item_autoapprove = types.KeyboardButton("Автоодобрение на вступление в группу")
    item_about = types.KeyboardButton("О нас")
    markup.add(item_autopost, item_autoapprove, item_about)
    bot.send_message(user_id, "Привет! Я бот, который поможет автоматизировать ваш телеграм канал. Что вы хотите?", reply_markup=markup)

@bot.message_handler(func=lambda message: message.text == "О нас")
def about_us(message):
    """Обработчик кнопки 'О нас'."""
    user_id = message.from_user.id
    bot.send_message(user_id, "Мы - команда разработчиков, стремящаяся упростить ведение телеграм каналов. С нашей помощью вы сможете автоматизировать публикацию контента и одобрение заявок на вступление в группу.  Нажмите /start, чтобы вернуться к выбору действий.")
    USER_STATES[user_id] = STATE_START

@bot.message_handler(func=lambda message: message.text in ["Автопостинг", "Автоодобрение на вступление в группу"])
def choose_feature(message):
    """Обработчик выбора 'Автопостинг' или 'Автоодобрение'."""
    user_id = message.from_user.id
    USER_STATES[user_id] = STATE_CHOOSE_FEATURE

    USER_DATA[user_id] = {}

    if message.text == "Автопостинг":
        USER_DATA[user_id]['feature'] = 'autoposting'
        bot.send_message(user_id, "Вы выбрали Автопостинг. Отлично!")
    elif message.text == "Автоодобрение на вступление в группу":
        USER_DATA[user_id]['feature'] = 'autoapprove'
        bot.send_message(user_id, "Вы выбрали Автоодобрение. Отлично!")

    choose_plan(message)


def choose_plan(message):
    """Функция выбора тарифного плана."""
    user_id = message.from_user.id
    USER_STATES[user_id] = STATE_CHOOSE_PLAN
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    item_1week = types.KeyboardButton("1 неделя")
    item_1month = types.KeyboardButton("1 месяц")
    item_1year = types.KeyboardButton("1 год")
    item_trial = types.KeyboardButton("Пробный период 7 дней")
    markup.add(item_1week, item_1month, item_1year, item_trial)
    bot.send_message(user_id, "Выберите тарифный план:", reply_markup=markup)

@bot.message_handler(func=lambda message: USER_STATES.get(message.from_user.id) == STATE_CHOOSE_PLAN)
def handle_any_text_choose_plan(message):
    user_id = message.from_user.id
    if message.text not in ["1 неделя", "1 месяц", "1 год", "Пробный период 7 дней"]:
        bot.send_message(user_id, "Пожалуйста, выберите тариф, нажав на одну из кнопок.")
    else:
        choose_payment(message)


@bot.message_handler(func=lambda message: message.text in ["1 неделя", "1 месяц", "1 год", "Пробный период 7 дней"])
def choose_payment(message):
    """Обработчик выбора тарифного плана."""
    user_id = message.from_user.id
    USER_STATES[user_id] = STATE_CHOOSE_PAYMENT

    if message.text == "1 неделя":
        USER_DATA[user_id]['plan'] = '1week'
    elif message.text == "1 месяц":
        USER_DATA[user_id]['plan'] = '1month'
    elif message.text == "1 год":
        USER_DATA[user_id]['plan'] = '1year'
    elif message.text == "Пробный период 7 дней":
        USER_DATA[user_id]['plan'] = 'trial'

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    item_sbp = types.KeyboardButton("СБП")
    item_card = types.KeyboardButton("Банковская карта")
    item_qiwi = types.KeyboardButton("Qiwi")
    markup.add(item_sbp, item_card, item_qiwi)
    bot.send_message(user_id, "Выберите способ оплаты:", reply_markup=markup)

@bot.message_handler(func=lambda message: USER_STATES.get(message.from_user.id) == STATE_CHOOSE_PAYMENT)
def handle_any_text_choose_payment(message):
    user_id = message.from_user.id
    if message.text not in ["СБП", "Банковская карта", "Qiwi"]:
        bot.send_message(user_id, "Пожалуйста, выберите способ оплаты, нажав на одну из кнопок.")
    else:
        show_cabinet(message)


@bot.message_handler(func=lambda message: message.text in ["СБП", "Банковская карта", "Qiwi"])
def show_cabinet(message):
    """Обработчик выбора способа оплаты и отображение личного кабинета."""
    user_id = message.from_user.id
    USER_STATES[user_id] = STATE_INPUT_CHANNEL

    # Здесь можно добавить логику обработки выбора способа оплаты (фиктивную)
    bot.send_message(user_id, f"Вы выбрали оплату через {message.text}.  Оплата пока не реализована.")

    # Запрашиваем ссылку на канал
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    bot.send_message(user_id, "Теперь укажите ссылку на ваш телеграм канал (например, https://t.me/mychannel или @mychannel):", reply_markup=markup)

@bot.message_handler(func=lambda message: USER_STATES.get(message.from_user.id) == STATE_INPUT_CHANNEL)
def process_channel_link(message):
    """Обработчик ссылки на канал."""
    user_id = message.from_user.id
    channel_link = message.text

    # Convert @username to a full t.me link if needed
    if channel_link.startswith('@'):
        channel_link = convert_to_telegram_link(channel_link)

    if is_telegram_link(channel_link):
        USER_DATA[user_id]['channel_link'] = channel_link

        # Запрашиваем тематику канала
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
        bot.send_message(user_id, "Укажите тематику вашего канала (например, 'Новости технологий'). Если тематики нет, напишите 'Нет'.", reply_markup=markup)
        USER_STATES[user_id] = STATE_ASK_THEME
    else:
        bot.send_message(user_id, "Пожалуйста, укажите корректную ссылку на телеграм-канал (например, https://t.me/mychannel или @mychannel).")

@bot.message_handler(func=lambda message: USER_STATES.get(message.from_user.id) == STATE_ASK_THEME)
def process_channel_theme(message):
    """Обработчик тематики канала."""
    user_id = message.from_user.id
    theme = message.text
    USER_DATA[user_id]['theme'] = theme

    #Запрашиваем ссылку на правила канала
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    bot.send_message(user_id, "Укажите ссылку на правила вашего канала (если есть). Если правил нет, напишите 'Нет'.", reply_markup=markup)
    USER_STATES[user_id] = STATE_ASK_RULES

@bot.message_handler(func=lambda message: USER_STATES.get(message.from_user.id) == STATE_ASK_RULES)
def process_channel_rules(message):
    """Обработчик ссылки на правила канала."""
    user_id = message.from_user.id
    rules_link = message.text
    USER_DATA[user_id]['rules_link'] = rules_link

    # Запрашиваем действие при обнаружении спама
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    item_ignore = types.KeyboardButton("Игнорировать")
    item_delete = types.KeyboardButton("Удалить сообщение")
    item_delete_and_ban = types.KeyboardButton("Удалить и заблокировать пользователя")
    markup.add(item_ignore, item_delete, item_delete_and_ban)
    bot.send_message(user_id, "Выберите действие при обнаружении спама:", reply_markup=markup) #Изменено "стоп-слова" на "спама"
    USER_STATES[user_id] = STATE_CHOOSE_STOP_WORD_ACTION

# Добавим обработчик для любых сообщений, когда пользователь находится в состоянии выбора действия при стоп-слове
@bot.message_handler(func=lambda message: USER_STATES.get(message.from_user.id) == STATE_CHOOSE_STOP_WORD_ACTION)
def handle_any_text_stop_word_action(message):
    user_id = message.from_user.id
    if message.text not in ["Игнорировать", "Удалить сообщение", "Удалить и заблокировать пользователя"]:
        # Сообщение не является одной из кнопок, сообщаем об этом пользователю
        bot.send_message(user_id, "Пожалуйста, выберите действие, нажав на одну из кнопок.")
    else:
        # Если пользователь нажал на кнопку
        process_stop_word_action(message)

@bot.message_handler(func=lambda message: message.text in ["Игнорировать", "Удалить сообщение", "Удалить и заблокировать пользователя"] and USER_STATES.get(message.from_user.id) == STATE_CHOOSE_STOP_WORD_ACTION)
def process_stop_word_action(message):
    """Обработчик выбора действия при обнаружении спама.""" #Изменено "стоп-слова" на "спама"
    user_id = message.from_user.id
    action = message.text

    USER_DATA[user_id]['stop_word_action'] = action

    # Выводим инструкцию
    instruction = f"""
Инструкция по подключению бота к каналу:
1. Добавьте этого бота в администраторы вашего канала: {bot.get_me().username}
2. Предоставьте боту права на публикацию сообщений и управление каналом.

После выполнения, нажмите кнопку 'Проверить подключение'."""

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    item_check = types.KeyboardButton("Проверить подключение")
    item_exit = types.KeyboardButton("Выйти")
    markup.add(item_check, item_exit)
    bot.send_message(user_id, instruction, reply_markup=markup)
    USER_STATES[user_id] = STATE_CHECK_CONNECTION

@bot.message_handler(func=lambda message: USER_STATES.get(message.from_user.id) == STATE_CHECK_CONNECTION)
def handle_any_text_check_connection(message):
    user_id = message.from_user.id
    if message.text not in ["Проверить подключение", "Выйти"]:
        bot.send_message(user_id, "Пожалуйста, нажмите на одну из кнопок.")
    else:
        check_connection(message)

@bot.message_handler(func=lambda message: message.text == "Проверить подключение" and USER_STATES.get(message.from_user.id) == STATE_CHECK_CONNECTION)
def check_connection(message):
    """Обработчик кнопки "Проверить подключение"."""
    user_id = message.from_user.id
    channel_link = USER_DATA[user_id].get('channel_link')
    theme = USER_DATA[user_id].get('theme')
    rules_link = USER_DATA[user_id].get('rules_link')
    stop_word_action = USER_DATA[user_id].get('stop_word_action')

    try:
        # Получаем ID канала из ссылки
        channel_id = channel_link.split('/')[-1]
        chat = bot.get_chat(f'@{channel_id}')
        chat_id = chat.id

        USER_DATA[user_id]['channel_id'] = chat_id

        # Get chat member information.
        member = bot.get_chat_member(chat_id=chat_id, user_id=bot.get_me().id)

        if member.status in ['administrator', 'creator']:
            # Бот является администратором или создателем канала
            bot.send_message(user_id, "Подключение успешно проверено! Бот является администратором канала и готов к работе!")

            # Initialize bot activation state for the channel
            BOT_ACTIVE[chat_id] = True
            CHANNEL_STATES[chat_id] = STATE_ACTIVE
            USER_STATES[user_id] = STATE_ACTIVE

            # Prepare the welcome message components
            theme_line = ""
            if theme.lower() != "нет":
                theme_line = f"Рад видеть тебя в нашем канале! Здесь ты найдешь много полезной информации о {theme}."
            else:
                theme_line = "Рад видеть тебя в нашем канале!"

            rules_line = ""
            if rules_link.lower() != "нет":
                rules_line = f"Пожалуйста, ознакомься с правилами канала: {rules_link}."

            # Format the welcome message
            formatted_welcome_message = WELCOME_MESSAGE.format(
                theme_line=theme_line,
                rules_line=rules_line
            )

            # Отправляем сообщение о начале работы в канал
            bot.send_message(chat_id, START_MESSAGE)
            # Добавляем кнопку "Перенастроить" и "Статистика"
            markup_active = types.ReplyKeyboardMarkup(resize_keyboard=True)  # remove one_time_keyboard=True
            item_reconfigure = types.KeyboardButton("⚙️ Перенастроить")
            item_statistics = types.KeyboardButton("📊 Статистика")
            markup_active.add(item_reconfigure, item_statistics)
            bot.send_message(user_id, "Бот активирован.", reply_markup=markup_active)

            # Initialize statistics for the channel
            if chat_id not in CHAT_STATISTICS:
                CHAT_STATISTICS[chat_id] = {
                    'new_members': 0,
                    'spam_deleted': 0
                }

        else:
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
            item_retry = types.KeyboardButton("Повторить проверку")
            item_exit = types.KeyboardButton("Выйти")
            markup.add(item_retry, item_exit)
            bot.send_message(user_id, "Бот не является администратором канала.  Пожалуйста, добавьте бота в администраторы с необходимыми правами и попробуйте снова.", reply_markup=markup)
            return  # Важно! Прекращаем выполнение функции, если проверка не прошла

    except telebot.apihelper.ApiTelegramException as e:
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
        item_retry = types.KeyboardButton("Повторить проверку")
        item_exit = types.KeyboardButton("Выйти")
        markup.add(item_retry, item_exit)
        if e.description == 'Bad Request: chat not found':
            bot.send_message(user_id, "Канал не найден.  Проверьте ссылку.", reply_markup=markup)
        elif e.description == "Bad Request: user not found":
            bot.send_message(user_id, "Пользователь не найден. Убедитесь, что бот добавлен в канал/группу.", reply_markup=markup)
        else:
            bot.send_message(user_id, f"Произошла ошибка: {e}", reply_markup=markup)
        return # Важно! Прекращаем выполнение функции, если проверка не прошла
    except Exception as e:
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
        item_retry = types.KeyboardButton("Повторить проверку")
        item_exit = types.KeyboardButton("Выйти")
        markup.add(item_retry, item_exit)
        bot.send_message(user_id, f"Произошла ошибка: {e}", reply_markup=markup)
        return  # Важно! Прекращаем выполнение функции, если проверка не прошла

# Добавляем обработчик кнопки "Повторить проверку"
@bot.message_handler(func=lambda message: message.text == "Повторить проверку" and USER_STATES.get(message.from_user.id) == STATE_CHECK_CONNECTION)
def retry_connection(message):
    """Обработчик кнопки "Повторить проверку"."""
    check_connection(message) # Просто вызываем функцию проверки снова

#Добавляем возможность перенастройки бота:
@bot.message_handler(func=lambda message: message.chat.type == 'private' and USER_STATES.get(message.from_user.id) == STATE_ACTIVE)
def handle_any_text_state_active(message):
    user_id = message.from_user.id
    if message.text not in ["⚙️ Перенастроить", "📊 Статистика"]:
        #Если это не кнопка, возможно, стоит проигнорировать сообщение
        #Но если вы хотите как-то реагировать, раскомментируйте следующую строку:
        #bot.send_message(user_id, "Пожалуйста, нажмите кнопку 'Перенастроить' или 'Статистика'.")
        return  # <--- Добавлено: просто выходим из функции, ничего не делая
    elif message.text == "⚙️ Перенастроить":
        reconfigure_channel(message)
    elif message.text == "📊 Статистика":
        send_statistics(message)

@bot.message_handler(func=lambda message: message.text == "⚙️ Перенастроить" and USER_STATES.get(message.from_user.id) == STATE_ACTIVE)
def reconfigure_channel(message):
    """Обработчик кнопки "Перенастроить"."""
    user_id = message.from_user.id
    USER_STATES[user_id] = STATE_RECONFIGURE
    chat_id = USER_DATA[user_id]['channel_id']

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True) # remove one_time_keyboard=True
    item_channel = types.KeyboardButton("Изменить канал")
    item_theme = types.KeyboardButton("Изменить тему")
    item_rules = types.KeyboardButton("Изменить правила")
    item_stop_word_action = types.KeyboardButton("Изменить действие при спаме") #Изменено "стоп-слове" на "спаме"
    # Add "Включить/Выключить бота" button
    if BOT_ACTIVE.get(chat_id, False):  # Get current state, default to False
        item_toggle_bot = types.KeyboardButton("❌ Выключить бота")
    else:
        item_toggle_bot = types.KeyboardButton("✅ Включить бота")

    item_exit = types.KeyboardButton("Выйти")
    markup.add(item_channel, item_theme, item_rules, item_stop_word_action, item_toggle_bot, item_exit)

    bot.send_message(user_id, "Что вы хотите изменить?", reply_markup=markup)

@bot.message_handler(func=lambda message: message.text == "Изменить канал" and USER_STATES.get(message.from_user.id) == STATE_RECONFIGURE)
def reconfigure_channel_link(message):
    """Обработчик изменения ссылки на канал."""
    user_id = message.from_user.id
    USER_STATES[user_id] = STATE_INPUT_CHANNEL  # Возвращаемся к шагу ввода канала
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    bot.send_message(user_id, "Укажите новую ссылку на ваш телеграм канал:", reply_markup=markup)

@bot.message_handler(func=lambda message: message.text == "Изменить тему" and USER_STATES.get(message.from_user.id) == STATE_RECONFIGURE)
def reconfigure_channel_theme(message):
    """Обработчик изменения темы канала."""
    user_id = message.from_user.id
    USER_STATES[user_id] = STATE_ASK_THEME  # Возвращаемся к шагу ввода темы
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    bot.send_message(user_id, "Укажите новую тему для вашего канала:", reply_markup=markup)

@bot.message_handler(func=lambda message: message.text == "Изменить правила" and USER_STATES.get(message.from_user.id) == STATE_RECONFIGURE)
def reconfigure_channel_rules(message):
    """Обработчик изменения правил канала."""
    user_id = message.from_user.id
    USER_STATES[user_id] = STATE_ASK_RULES  # Возвращаемся к шагу ввода правил
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    bot.send_message(user_id, "Укажите новую ссылку на правила вашего канала:", reply_markup=markup)

@bot.message_handler(func=lambda message: message.text == "Изменить действие при спаме" and USER_STATES.get(message.from_user.id) == STATE_RECONFIGURE) #Изменено "стоп-слове" на "спаме"
def reconfigure_stop_word_action(message):
    """Обработчик изменения действия при обнаружении спама.""" #Изменено "стоп-слова" на "спама"
    user_id = message.from_user.id
    USER_STATES[user_id] = STATE_CHOOSE_STOP_WORD_ACTION  # Возвращаемся к шагу выбора действия
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    item_ignore = types.KeyboardButton("Игнорировать")
    item_delete = types.KeyboardButton("Удалить сообщение")
    item_delete_and_ban = types.KeyboardButton("Удалить и заблокировать пользователя")
    markup.add(item_ignore, item_delete, item_delete_and_ban)
    bot.send_message(user_id, "Выберите новое действие при обнаружении спама:", reply_markup=markup) #Изменено "стоп-слова" на "спама"

@bot.message_handler(func=lambda message: message.text in ["✅ Включить бота", "❌ Выключить бота"] and USER_STATES.get(message.from_user.id) == STATE_RECONFIGURE)
def toggle_bot(message):
    """Обработчик кнопки "Включить бота" / "Выключить бота"."""
    user_id = message.from_user.id
    chat_id = USER_DATA[user_id]['channel_id']

    if message.text == "✅ Включить бота":
        BOT_ACTIVE[chat_id] = True
        bot.send_message(user_id, "Бот включен!")
    else:
        BOT_ACTIVE[chat_id] = False
        bot.send_message(user_id, "Бот выключен!")

    # Return to reconfiguration menu to update the button
    reconfigure_channel(message)


@bot.message_handler(func=lambda message: message.text == "Выйти" and USER_STATES.get(message.from_user.id) == STATE_RECONFIGURE)
def exit_reconfigure(message):
    """Обработчик кнопки "Выйти" из режима перенастройки."""
    user_id = message.from_user.id
    USER_STATES[user_id] = STATE_ACTIVE
    # Добавляем кнопку "Перенастроить" и "Статистика" при выходе из режима перенастройки
    markup_active = types.ReplyKeyboardMarkup(resize_keyboard=True) # remove one_time_keyboard=True
    item_reconfigure = types.KeyboardButton("⚙️ Перенастроить")
    item_statistics = types.KeyboardButton("📊 Статистика")
    markup_active.add(item_reconfigure, item_statistics)
    bot.send_message(user_id, "Вы вышли из режима перенастройки.", reply_markup=markup_active)

@bot.message_handler(func=lambda message: message.text == "📊 Статистика" and USER_STATES.get(message.from_user.id) == STATE_ACTIVE)
def send_statistics(message):
    """Отправляет статистику по каналу."""
    user_id = message.from_user.id
    chat_id = USER_DATA[user_id]['channel_id']
    now = time.time()

    if user_id in LAST_STATISTICS_REQUEST and (now - LAST_STATISTICS_REQUEST[user_id]) < STATISTICS_COOLDOWN:
        remaining_time = int(STATISTICS_COOLDOWN - (now - LAST_STATISTICS_REQUEST[user_id]))
        bot.send_message(user_id, f"Пожалуйста, подождите {remaining_time} секунд перед запросом статистики снова.")
        return

    if chat_id in CHAT_STATISTICS:
        stats = CHAT_STATISTICS[chat_id]
        message_text = f"""
📊 Статистика канала:
Новых участников: {stats['new_members']}
Удалено спам-сообщений: {stats['spam_deleted']}
        """
        # Add the buttons back
        markup_active = types.ReplyKeyboardMarkup(resize_keyboard=True) # remove one_time_keyboard=True
        item_reconfigure = types.KeyboardButton("⚙️ Перенастроить")
        item_statistics = types.KeyboardButton("📊 Статистика")
        markup_active.add(item_reconfigure, item_statistics)
        
        bot.send_message(user_id, message_text, reply_markup=markup_active) # Send the statistics with the buttons
        LAST_STATISTICS_REQUEST[user_id] = now
    else:
         # Add the buttons back even if there's no stats
        markup_active = types.ReplyKeyboardMarkup(resize_keyboard=True) # remove one_time_keyboard=True
        item_reconfigure = types.KeyboardButton("⚙️ Перенастроить")
        item_statistics = types.KeyboardButton("📊 Статистика")
        markup_active.add(item_reconfigure, item_statistics)
        bot.send_message(user_id, "Статистика пока не собрана.",  reply_markup=markup_active)

# --- Функции спам фильтра---
def is_spam(text, threshold=SPAM_THRESHOLD):
    """Использует модель машинного обучения для определения, является ли текст спамом."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device) # Send inputs to device

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        spam_probability = probabilities[0, 1].item()  # Вероятность класса "спам"

    return spam_probability > threshold


@bot.message_handler(func=lambda message: message.chat.type in ['group', 'supergroup'], content_types=['text', 'photo', 'video', 'audio', 'document'])
def filter_messages(message):
    """Фильтрует сообщения, используя модель машинного обучения."""
    chat_id = message.chat.id
    # user_id = message.from_user.id # This user id is of the person sending message to group not admin
    # Check if the bot is active for this channel
    if chat_id in BOT_ACTIVE and BOT_ACTIVE[chat_id]:
        if chat_id in CHANNEL_STATES and CHANNEL_STATES[chat_id] == STATE_ACTIVE:
            text = ""
            # Извлекаем текст из разных типов сообщений
            if message.text:
                text = message.text
            elif message.caption:
                text = message.caption

            # Проверяем, является ли сообщение спамом
            if text and is_spam(text):
                print(f"Обнаружен спам в сообщении ID {message.message_id}.")
                
                # We need to get the user_id of the OWNER not the message sender
                # This can be done by retrieving the owner's user_id from USER_DATA dict using the channel_id
                owner_user_id = None
                for user_id, data in USER_DATA.items():
                    if data.get('channel_id') == chat_id:
                        owner_user_id = user_id
                        break

                # Get the configured action for stop words
                # action = USER_DATA.get(user_id, {}).get('stop_word_action', 'Удалить сообщение')  # Default to "Удалить сообщение" if not set
                action = USER_DATA.get(owner_user_id, {}).get('stop_word_action', 'Удалить сообщение')  # Default to "Удалить сообщение" if not set
                print(f"Выбранное действие: {action}")

                try:
                    if action == "Удалить сообщение":
                        bot.delete_message(chat_id, message.message_id)
                        print(f"Удалено сообщение ID {message.message_id}.")
                        # Increment spam counter
                        if chat_id in CHAT_STATISTICS:
                            CHAT_STATISTICS[chat_id]['spam_deleted'] += 1

                    elif action == "Удалить и заблокировать пользователя":
                        bot.delete_message(chat_id, message.message_id)
                        bot.kick_chat_member(chat_id, message.from_user.id)
                        print(f"Удалено сообщение ID {message.message_id} и заблокирован пользователь {message.from_user.id}.")
                        # Increment spam counter
                        if chat_id in CHAT_STATISTICS:
                            CHAT_STATISTICS[chat_id]['spam_deleted'] += 1
                    elif action == "Игнорировать":
                        print(f"Обнаружен спам, но действие - игнорировать.")
                    return  # Exit after processing the spam

                except telebot.apihelper.ApiTelegramException as e:
                    print(f"Ошибка при выполнении действия: {e}")
                return
    else:
        print(f"Бот выключен для канала {chat_id}. Фильтрация сообщений не выполняется.")

# # Функция для отправки сообщения пользователю с кнопкой "Я не бот" - Убрано
# def send_verification_message(chat_id, user_id):
#     markup = types.InlineKeyboardMarkup()
#     item_verify = types.InlineKeyboardButton("Я не бот", callback_data=f"verify:{user_id}")
#     markup.add(item_verify)

#     try:
#         bot.send_message(user_id, "Пожалуйста, подтвердите, что вы не бот, нажав на кнопку ниже в течение 5 минут.", reply_markup=markup)
#         # Запоминаем пользователя и время отправки сообщения
#         pending_users[user_id] = {
#             'chat_id': chat_id,
#             'timestamp': time.time()
#         }
#     except telebot.apihelper.ApiTelegramException as e:
#         print(f"Ошибка отправки сообщения пользователю {user_id}: {e}")
#         # В случае ошибки (например, пользователь заблокировал бота), удаляем его из чата.
#         try:
#             bot.kick_chat_member(chat_id, user_id)
#         except Exception as ex:
#             print(f"Не удалось кикнуть пользователя {user_id} из чата {chat_id}: {ex}")

# Обработчик добавления новых участников в чат - Убрано
# @bot.message_handler(content_types=['new_chat_members'])
# def handle_new_member(message):
#     chat_id = message.chat.id
#     for new_member in message.new_chat_members:
#         # Игнорируем добавление самого бота
#         if new_member.id == bot.get_me().id:
#             continue
#         if not new_member.is_bot:            # Increment new members counter
#             if chat_id in CHAT_STATISTICS:
#                 CHAT_STATISTICS[chat_id]['new_members'] += 1

# --- Обработчик всех остальных сообщений ---
@bot.message_handler(func=lambda message: message.chat.type == 'private') #Убрал условие True
def echo_all(message):
    """Обработчик всех остальных сообщений."""
    user_id = message.from_user.id
    if USER_STATES.get(user_id) is None:
        bot.reply_to(message, "Пожалуйста, начните с команды /start")
    else:
        if USER_STATES.get(user_id) == STATE_ACTIVE:
            # Если бот активирован, то переадресуем в обработчик кнопок
            handle_any_text_state_active(message)
        else:
             bot.reply_to(message, "Не понимаю. Пожалуйста, используйте кнопки или команду /start.") #Вставил else

# --- Запуск бота ---
if __name__ == '__main__':
    print("Бот запущен...")
    bot.infinity_polling()