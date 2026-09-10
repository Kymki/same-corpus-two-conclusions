import telethon.sync
from telethon import TelegramClient
import csv
import datetime
import time
import asyncio
import os
import sys
from dotenv import load_dotenv

# Add root directory to sys.path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import DATA_RAW

load_dotenv()

# --- CONFIGURATION ---
# Telegram API Credentials
API_ID = int(os.getenv("TELEGRAM_API_ID", 0))
API_HASH = os.getenv("TELEGRAM_API_HASH", "")

# Name for the Telethon session file
SESSION_NAME = "thesis_session"

# Target entities (Channels or Groups)
# Channels actually present in telegram_messaggi_raccolti.csv, verified against
# the chat_title field of the collected data (message counts in that corpus):
#   @WarTranslated 728, @rybar_in_english 711, @Slavyangrad 580,
#   @ClashReport 459, @militaresemplice 308
TARGET_ENTITIES = ["militaresemplice", "ClashReport", "rybar_in_english", "WarTranslated", "Slavyangrad"]

# Collection start date (February 2022)
START_DATE = datetime.datetime(2025, 5, 25, tzinfo=datetime.timezone.utc)

# Message limit per entity (set to None for unlimited)
MESSAGE_LIMIT_PER_ENTITY = 1000

OUTPUT_FILE_TELEGRAM = os.path.join(DATA_RAW, "telegram_messaggi_raccolti.csv")

# English Keywords
KEYWORDS_EN = [
    "ukraine", "ukrainian", "russia", "russian", "war", "conflict", "attack",
    "military", "troops", "soldiers", "putin", "zelensky", "nato", "drone",
    "invasion", "forces", "defense", "weapon", "sanction", "peace",
    "refugee", "casualty", "territory", "frontline", "kyiv", "moscow", "kremlin",
    "donbas", "crimea", "kharkiv", "kherson", "mariupol", "bakhmut", "shelling",
    "artillery", "wagner", "himars"
]

# Italian Keywords
KEYWORDS_IT = [
    "ucraina", "ucraino", "russia", "russo", "guerra", "conflitto", "attacco",
    "militare", "truppe", "soldati", "putin", "zelensky", "nato", "drone",
    "invasione", "forze", "difesa", "arma", "sanzione", "pace",
    "profugo", "vittima", "territorio", "fronte", "kiev", "mosca", "cremlino",
    "donbas", "crimea", "kharkiv", "kherson", "mariupol", "bakhmut",
    "bombardamento", "artiglieria", "wagner"
]

KEYWORDS_FILTER = KEYWORDS_EN + KEYWORDS_IT

# --- DATA COLLECTION FUNCTION ---

async def collect_telegram_data():
    """Main async function to collect messages from Telegram channels."""
    
    client = TelegramClient(SESSION_NAME, API_ID, API_HASH)

    print("Attempting to connect to Telegram...")
    try:
        await client.connect()
    except Exception as e:
        print(f"ERROR: Could not connect. Details: {e}")
        return

    if not await client.is_user_authorized():
        print("Authentication required.")
        phone_number_input = input("Enter your phone number (international format, e.g., +39...): ")
        
        try:
            sent_code_info = await client.send_code_request(phone_number_input)
            code_input = input("Enter the code received via Telegram: ")
            await client.sign_in(
                phone=phone_number_input,
                code=code_input,
                phone_code_hash=sent_code_info.phone_code_hash
            )
        except Exception as e_auth:
            print(f"Authentication failed: {e_auth}")
            if client.is_connected(): await client.disconnect()
            return
            
        print("Authentication successful!")
    else:
        print("Authentication already valid.")

    try:
        with open(OUTPUT_FILE_TELEGRAM, 'w', newline='', encoding='utf-8') as file_csv:
            writer = csv.writer(file_csv)
            writer.writerow([
                "message_id", "chat_id", "chat_title", "sender_id", 
                "text", "timestamp_utc", "reply_to_message_id", "views"
            ])

            for entity_identifier in TARGET_ENTITIES:
                print(f"\n--- Scraping entity: {entity_identifier} ---")
                try:
                    entity = await client.get_entity(entity_identifier)
                    chat_title = getattr(entity, 'title', str(getattr(entity, 'username', 'Unknown')))
                    
                    print(f"  Channel/Group found: '{chat_title}' (ID: {entity.id})")

                    message_count = 0
                    async for message in client.iter_messages(entity, limit=MESSAGE_LIMIT_PER_ENTITY):
                        if message.date < START_DATE:
                            print(f"  Reached start date ({START_DATE.date()}) for '{chat_title}'.")
                            break
                        
                        if message.text and message.text.strip():
                            text_content = message.text.strip()
                            
                            # Apply keyword filter
                            if KEYWORDS_FILTER:
                                if not any(kw.lower() in text_content.lower() for kw in KEYWORDS_FILTER):
                                    continue
                            
                            sender_id_val = getattr(message, 'sender_id', None)
                            reply_to_val = getattr(message.reply_to, 'reply_to_msg_id', None) if hasattr(message, 'reply_to') else None
                            views_val = getattr(message, 'views', None)

                            writer.writerow([
                                message.id, entity.id, chat_title, sender_id_val, 
                                text_content, message.date.timestamp(), 
                                reply_to_val, views_val
                            ])
                            message_count += 1
                            if message_count % 100 == 0: 
                                print(f"    ...collected {message_count} messages from '{chat_title}'")
                    
                    print(f"  Finished: {message_count} messages collected from '{chat_title}'.")

                except Exception as e_msg:
                    print(f"  Error processing '{entity_identifier}': {e_msg}")
                
                print("Waiting 5 seconds before next entity...")
                time.sleep(5) 

    except Exception as e_outer:
        print(f"Global error: {e_outer}")
    finally:
        if client.is_connected():
            await client.disconnect()
            print("Disconnected from Telegram.")

if __name__ == "__main__":
    if not API_ID or not API_HASH:
        print("ERROR: Please provide TELEGRAM_API_ID and TELEGRAM_API_HASH in the .env file.")
    else:
        asyncio.run(collect_telegram_data())
