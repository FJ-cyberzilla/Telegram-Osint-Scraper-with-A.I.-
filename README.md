# Telegram-Osint-Scraper-with-A.I.-
Comprehensive Telegram OSINT scraper with AI capabilities that works on Windows/WSL. This solution will include data gathering, analysis, and machine learning components.


Key Features

1. Multi-source Data Collection: Scrapes Telegram messages, user profiles, and metadata
2. AI-Powered Analysis:
   · Sentiment analysis using NLTK's VADER
   · Toxicity detection with multilingual XLM-RoBERTa
   · Text embedding for similarity analysis
   · Anomaly detection with Isolation Forest
3. Pattern Recognition:
   · User activity patterns (active hours, days)
   · Behavioral similarities between users
   · Anomalous behavior detection
4. External API Integration:
   · Shodan for external intelligence
   · Support for Netlet, IPScraper, and Malboxer APIs
5. Comprehensive Storage:
   · SQLite database with structured tables
   · Efficient data retrieval for analysis
6. Reporting:
   · Automated report generation
   · Statistics on users, messages, and patterns

Installation Requirements

```bash
pip install telethon aiohttp sqlite3 pandas numpy scikit-learn nltk transformers torch requests shodan
```

Usage

1. Obtain API keys from Telegram, Shodan, and other services
2. Edit the config.json file with your credentials
3. Run the script: python telegram_osint.py
4. The system will start scraping and analyzing data
5. Reports will be generated periodically in osint_report.json
