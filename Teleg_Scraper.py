import os
import time
import logging
import asyncio
import aiohttp
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json

# Machine Learning & AI imports
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline, AutoTokenizer, AutoModel
import torch

# Telegram specific imports
from telethon import TelegramClient, events
from telethon.tl.types import Message, User, Channel, Chat
from telethon.tl.functions.messages import GetHistoryRequest

# External API imports
import requests
from shodan import Shodan

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("telegram_osint.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TelegramOSINTScraper:
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the Telegram OSINT Scraper with configuration
        """
        self.load_config(config_path)
        self.setup_database()
        self.setup_ml_models()
        self.setup_apis()
        
        # Initialize Telegram client
        self.client = TelegramClient(
            self.config['telegram']['session_name'],
            self.config['telegram']['api_id'],
            self.config['telegram']['api_hash']
        )
        
        # Statistics and monitoring
        self.stats = {
            'messages_processed': 0,
            'users_analyzed': 0,
            'last_activity': datetime.now(),
            'errors': 0
        }
        
    def load_config(self, config_path: str):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    def setup_database(self):
        """Setup SQLite database for storage"""
        try:
            self.db_path = self.config['database'].get('path', 'telegram_osint.db')
            self.conn = sqlite3.connect(self.db_path)
            self.create_tables()
            logger.info("Database setup completed")
        except Exception as e:
            logger.error(f"Error setting up database: {e}")
            raise
    
    def create_tables(self):
        """Create necessary database tables"""
        cursor = self.conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT,
                first_name TEXT,
                last_name TEXT,
                phone TEXT,
                is_bot INTEGER,
                is_premium INTEGER,
                is_verified INTEGER,
                status TEXT,
                last_online TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                chat_id INTEGER,
                message_text TEXT,
                date TIMESTAMP,
                views INTEGER,
                forwards INTEGER,
                replies INTEGER,
                media_type TEXT,
                sentiment_score REAL,
                toxicity_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Activity patterns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS activity_patterns (
                user_id INTEGER PRIMARY KEY,
                active_hours TEXT,
                avg_messages_per_day REAL,
                most_active_days TEXT,
                response_time_avg REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Similarity analysis table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS similarities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user1_id INTEGER,
                user2_id INTEGER,
                similarity_score REAL,
                similarity_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user1_id) REFERENCES users (id),
                FOREIGN KEY (user2_id) REFERENCES users (id)
            )
        ''')
        
        # External data table (Shodan, etc.)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS external_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                data_source TEXT,
                data_type TEXT,
                data JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        self.conn.commit()
    
    def setup_ml_models(self):
        """Initialize ML models for analysis"""
        try:
            # Sentiment analysis
            nltk.download('vader_lexicon', quiet=True)
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            
            # Toxicity classification
            self.toxicity_analyzer = pipeline(
                "text-classification",
                model="unitary/multilingual-toxic-xlm-roberta",
                tokenizer="unitary/multilingual-toxic-xlm-roberta"
            )
            
            # Text embedding model for similarity analysis
            self.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
            self.embedding_model = AutoModel.from_pretrained(self.embedding_model_name)
            
            # Anomaly detection model
            self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
            
            logger.info("ML models initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ML models: {e}")
            raise
    
    def setup_apis(self):
        """Setup external APIs"""
        try:
            # Shodan API
            self.shodan_client = Shodan(self.config['apis'].get('shodan', '')) if self.config['apis'].get('shodan') else None
            
            # Other APIs can be added here
            self.netlet_api_key = self.config['apis'].get('netlet', '')
            self.ipscraper_api_key = self.config['apis'].get('ipscraper', '')
            self.malboxer_api_key = self.config['apis'].get('malboxer', '')
            
            logger.info("API clients initialized")
        except Exception as e:
            logger.error(f"Error setting up APIs: {e}")
    
    async def start_scraping(self):
        """Start the scraping process"""
        try:
            await self.client.start()
            logger.info("Telegram client started successfully")
            
            # Get target channels/groups from config
            targets = self.config['scraping']['targets']
            
            for target in targets:
                logger.info(f"Starting to scrape target: {target}")
                await self.scrape_target(target)
                
            # Schedule periodic analysis
            await self.schedule_periodic_analysis()
            
        except Exception as e:
            logger.error(f"Error starting scraping: {e}")
    
    async def scrape_target(self, target: str):
        """Scrape a specific target (channel/group)"""
        try:
            entity = await self.client.get_entity(target)
            
            # Get historical messages
            await self.get_historical_messages(entity)
            
            # Set up handler for new messages
            @self.client.on(events.NewMessage(chats=entity))
            async def new_message_handler(event):
                await self.process_message(event.message, entity)
            
            logger.info(f"Scraping setup complete for {target}")
            
        except Exception as e:
            logger.error(f"Error scraping target {target}: {e}")
    
    async def get_historical_messages(self, entity, limit=10000):
        """Get historical messages from a channel/group"""
        try:
            messages = await self.client(GetHistoryRequest(
                peer=entity,
                limit=limit,
                offset_date=None,
                offset_id=0,
                max_id=0,
                min_id=0,
                add_offset=0,
                hash=0
            ))
            
            for message in messages.messages:
                await self.process_message(message, entity)
                
            logger.info(f"Processed {len(messages.messages)} historical messages from {entity.title if hasattr(entity, 'title') else entity.id}")
            
        except Exception as e:
            logger.error(f"Error getting historical messages: {e}")
    
    async def process_message(self, message: Message, entity: Any):
        """Process a single message"""
        try:
            # Extract message data
            message_data = {
                'id': message.id,
                'user_id': message.sender_id,
                'chat_id': entity.id,
                'text': message.text,
                'date': message.date,
                'views': message.views if hasattr(message, 'views') else 0,
                'forwards': message.forwards if hasattr(message, 'forwards') else 0,
                'replies': message.replies.replies if hasattr(message, 'replies') and message.replies else 0,
                'media_type': type(message.media).__name__ if message.media else 'text'
            }
            
            # Analyze message content
            analysis = await self.analyze_message(message_data['text'])
            message_data.update(analysis)
            
            # Store message in database
            self.store_message(message_data)
            
            # Update user information
            if message.sender_id:
                await self.update_user_info(message.sender_id)
            
            # Update statistics
            self.stats['messages_processed'] += 1
            self.stats['last_activity'] = datetime.now()
            
            # Periodic analysis every 100 messages
            if self.stats['messages_processed'] % 100 == 0:
                await self.run_periodic_analysis()
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.stats['errors'] += 1
    
    async def analyze_message(self, text: str) -> Dict[str, Any]:
        """Analyze message text with various AI models"""
        if not text:
            return {
                'sentiment_score': 0,
                'toxicity_score': 0,
                'embedding': None
            }
        
        try:
            # Sentiment analysis
            sentiment = self.sentiment_analyzer.polarity_scores(text)
            
            # Toxicity analysis (with error handling for long texts)
            try:
                toxicity = self.toxicity_analyzer(text[:512])[0]  # Limit text length
                toxicity_score = toxicity['score'] if toxicity['label'] == 'toxic' else 0
            except:
                toxicity_score = 0
            
            # Text embedding for similarity analysis
            embedding = await self.get_text_embedding(text)
            
            return {
                'sentiment_score': sentiment['compound'],
                'toxicity_score': toxicity_score,
                'embedding': json.dumps(embedding.tolist()) if embedding is not None else None
            }
            
        except Exception as e:
            logger.error(f"Error analyzing message: {e}")
            return {
                'sentiment_score': 0,
                'toxicity_score': 0,
                'embedding': None
            }
    
    async def get_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get text embedding for similarity analysis"""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            return embedding
        except Exception as e:
            logger.error(f"Error generating text embedding: {e}")
            return None
    
    def store_message(self, message_data: Dict[str, Any]):
        """Store message in database"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO messages 
                (id, user_id, chat_id, message_text, date, views, forwards, replies, media_type, sentiment_score, toxicity_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                message_data['id'],
                message_data['user_id'],
                message_data['chat_id'],
                message_data['text'],
                message_data['date'],
                message_data['views'],
                message_data['forwards'],
                message_data['replies'],
                message_data['media_type'],
                message_data['sentiment_score'],
                message_data['toxicity_score']
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error storing message: {e}")
    
    async def update_user_info(self, user_id: int):
        """Update user information in database"""
        try:
            user = await self.client.get_entity(user_id)
            
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO users 
                (id, username, first_name, last_name, phone, is_bot, is_premium, is_verified, status, last_online)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user.id,
                user.username,
                user.first_name,
                user.last_name,
                user.phone,
                1 if user.bot else 0,
                1 if user.premium else 0,
                1 if user.verified else 0,
                str(user.status) if user.status else None,
                user.status.was_online if hasattr(user.status, 'was_online') else None
            ))
            self.conn.commit()
            
            self.stats['users_analyzed'] += 1
            
        except Exception as e:
            logger.error(f"Error updating user info: {e}")
    
    async def schedule_periodic_analysis(self):
        """Schedule periodic analysis tasks"""
        try:
            while True:
                # Run analysis every hour
                await asyncio.sleep(3600)
                await self.run_periodic_analysis()
        except Exception as e:
            logger.error(f"Error in periodic analysis: {e}")
    
    async def run_periodic_analysis(self):
        """Run periodic analysis on collected data"""
        logger.info("Running periodic analysis...")
        
        try:
            # Analyze activity patterns
            await self.analyze_activity_patterns()
            
            # Find similar users
            await self.find_similar_users()
            
            # Detect anomalies
            await self.detect_anomalies()
            
            # Enrich with external data
            await self.enrich_with_external_data()
            
            logger.info("Periodic analysis completed")
            
        except Exception as e:
            logger.error(f"Error in periodic analysis: {e}")
    
    async def analyze_activity_patterns(self):
        """Analyze user activity patterns"""
        try:
            cursor = self.conn.cursor()
            
            # Get message data for activity analysis
            cursor.execute('''
                SELECT user_id, datetime(date) as message_time, COUNT(*) as message_count
                FROM messages 
                WHERE date > datetime('now', '-7 days')
                GROUP BY user_id, strftime('%H', date), strftime('%w', date)
            ''')
            
            activity_data = cursor.fetchall()
            
            # Process activity patterns (simplified)
            user_activity = {}
            for user_id, message_time, message_count in activity_data:
                if user_id not in user_activity:
                    user_activity[user_id] = {
                        'hours': [0] * 24,
                        'days': [0] * 7,
                        'total_messages': 0
                    }
                
                hour = datetime.strptime(message_time, '%Y-%m-%d %H:%M:%S').hour
                user_activity[user_id]['hours'][hour] += message_count
                user_activity[user_id]['total_messages'] += message_count
            
            # Store activity patterns
            for user_id, activity in user_activity.items():
                cursor.execute('''
                    INSERT OR REPLACE INTO activity_patterns 
                    (user_id, active_hours, avg_messages_per_day, most_active_days)
                    VALUES (?, ?, ?, ?)
                ''', (
                    user_id,
                    json.dumps(activity['hours']),
                    activity['total_messages'] / 7,  # Average per day over 7 days
                    json.dumps(activity['days'])
                ))
            
            self.conn.commit()
            logger.info("Activity patterns analyzed and stored")
            
        except Exception as e:
            logger.error(f"Error analyzing activity patterns: {e}")
    
    async def find_similar_users(self):
        """Find similar users based on various criteria"""
        try:
            # This would typically use the text embeddings we stored
            # For simplicity, we'll use a basic approach here
            
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT user_id, AVG(sentiment_score) as avg_sentiment, 
                       AVG(toxicity_score) as avg_toxicity,
                       COUNT(*) as message_count
                FROM messages 
                GROUP BY user_id
                HAVING message_count > 10
            ''')
            
            user_data = cursor.fetchall()
            
            # Simple similarity based on sentiment and toxicity
            for i, (user1_id, sent1, tox1, count1) in enumerate(user_data):
                for j, (user2_id, sent2, tox2, count2) in enumerate(user_data):
                    if i >= j:  # Avoid duplicate comparisons
                        continue
                    
                    # Calculate similarity score (simplified)
                    sentiment_diff = abs(sent1 - sent2)
                    toxicity_diff = abs(tox1 - tox2)
                    similarity_score = 1 - (sentiment_diff + toxicity_diff) / 2
                    
                    if similarity_score > 0.7:  Threshold for similarity
                        cursor.execute('''
                            INSERT INTO similarities 
                            (user1_id, user2_id, similarity_score, similarity_type)
                            VALUES (?, ?, ?, ?)
                        ''', (user1_id, user2_id, similarity_score, 'behavioral'))
            
            self.conn.commit()
            logger.info("Similar user analysis completed")
            
        except Exception as e:
            logger.error(f"Error finding similar users: {e}")
    
    async def detect_anomalies(self):
        """Detect anomalous behavior using ML"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT user_id, AVG(sentiment_score), AVG(toxicity_score), 
                       COUNT(*) as message_count, 
                       COUNT(DISTINCT DATE(date)) as active_days
                FROM messages 
                GROUP BY user_id
            ''')
            
            user_stats = cursor.fetchall()
            
            # Prepare data for anomaly detection
            X = []
            user_ids = []
            for user_id, avg_sent, avg_tox, msg_count, active_days in user_stats:
                X.append([avg_sent, avg_tox, msg_count, active_days])
                user_ids.append(user_id)
            
            if len(X) > 10:  # Only run if we have enough data
                X = np.array(X)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Detect anomalies
                anomalies = self.anomaly_detector.fit_predict(X_scaled)
                
                # Log anomalous users
                for i, is_anomaly in enumerate(anomalies):
                    if is_anomaly == -1:
                        logger.warning(f"Anomalous user detected: {user_ids[i]}")
            
            logger.info("Anomaly detection completed")
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
    
    async def enrich_with_external_data(self):
        """Enrich user data with external APIs"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('SELECT id, username FROM users WHERE username IS NOT NULL')
            users = cursor.fetchall()
            
            for user_id, username in users:
                # Shodan enrichment (if API available)
                if self.shodan_client:
                    try:
                        results = self.shodan_client.search(f"telegram:{username}")
                        if results['total'] > 0:
                            cursor.execute('''
                                INSERT INTO external_data 
                                (user_id, data_source, data_type, data)
                                VALUES (?, ?, ?, ?)
                            ''', (user_id, 'shodan', 'ip_info', json.dumps(results)))
                    except Exception as e:
                        logger.debug(f"No Shodan results for {username} or API error: {e}")
                
                # Add other API enrichments here (Netlet, IPScraper, Malboxer, etc.)
            
            self.conn.commit()
            logger.info("External data enrichment completed")
            
        except Exception as e:
            logger.error(f"Error in external data enrichment: {e}")
    
    def generate_report(self):
        """Generate a report of findings"""
        try:
            cursor = self.conn.cursor()
            
            # Get summary statistics
            cursor.execute('SELECT COUNT(*) FROM users')
            user_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM messages')
            message_count = cursor.fetchone()[0]
            
            cursor.execute('''
                SELECT COUNT(DISTINCT user1_id) FROM similarities 
                WHERE similarity_score > 0.7
            ''')
            similar_users_count = cursor.fetchone()[0]
            
            # Generate report
            report = {
                'generated_at': datetime.now().isoformat(),
                'user_count': user_count,
                'message_count': message_count,
                'similar_users_count': similar_users_count,
                'active_last_24h': self.get_active_users(24),
                'active_last_7d': self.get_active_users(24*7),
                'top_toxic_users': self.get_toxic_users(10),
                'most_active_users': self.get_most_active_users(10),
                'user_clusters': self.get_user_clusters()
            }
            
            # Save report
            with open('osint_report.json', 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Report generated: osint_report.json")
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return None
    
    def get_active_users(self, hours: int):
        """Get users active in the last specified hours"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT COUNT(DISTINCT user_id) 
            FROM messages 
            WHERE date > datetime('now', ?)
        ''', (f'-{hours} hours',))
        return cursor.fetchone()[0]
    
    def get_toxic_users(self, limit: int = 10):
        """Get most toxic users"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT user_id, AVG(toxicity_score) as avg_toxicity
            FROM messages 
            GROUP BY user_id 
            HAVING COUNT(*) > 5
            ORDER BY avg_toxicity DESC 
            LIMIT ?
        ''', (limit,))
        return cursor.fetchall()
    
    def get_most_active_users(self, limit: int = 10):
        """Get most active users"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT user_id, COUNT(*) as message_count
            FROM messages 
            GROUP BY user_id 
            ORDER BY message_count DESC 
            LIMIT ?
        ''', (limit,))
        return cursor.fetchall()
    
    def get_user_clusters(self):
        """Get user clusters based on behavior"""
        # This would typically use clustering algorithms
        # Simplified implementation
        return {"clusters": []}
    
    async def close(self):
        """Cleanup and close connections"""
        try:
            await self.client.disconnect()
            self.conn.close()
            logger.info("Connections closed successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

async def main():
    """Main function"""
    # Check if configuration exists
    if not os.path.exists("config.json"):
        print("Creating default configuration file...")
        create_default_config()
        print("Please edit config.json with your API keys and settings")
        return
    
    # Initialize and run the scraper
    scraper = TelegramOSINTScraper()
    
    try:
        await scraper.start_scraping()
        
        # Keep the script running
        while True:
            await asyncio.sleep(60)
            
            # Generate report every 6 hours
            if datetime.now().hour % 6 == 0:
                scraper.generate_report()
                
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        await scraper.close()

def create_default_config():
    """Create a default configuration file"""
    config = {
        "telegram": {
            "api_id": "YOUR_API_ID",
            "api_hash": "YOUR_API_HASH",
            "session_name": "telegram_osint"
        },
        "scraping": {
            "targets": ["@examplechannel", "@anothergroup"]
        },
        "apis": {
            "shodan": "YOUR_SHODAN_API_KEY",
            "netlet": "YOUR_NETLET_API_KEY",
            "ipscraper": "YOUR_IPSCRAPER_API_KEY",
            "malboxer": "YOUR_MALBOXER_API_KEY"
        },
        "database": {
            "path": "telegram_osint.db"
        },
        "analysis": {
            "similarity_threshold": 0.7,
            "anomaly_contamination": 0.1
        }
    }
    
    with open("config.json", "w") as f:
        json.dump(config, f, indent=2)

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())
