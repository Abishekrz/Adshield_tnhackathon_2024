# AdShield - TN Hackathon 2024

## Overview
AdShield is an AI-powered cyber scam detection system that analyzes social media content (text, images, and videos) to identify fraudulent advertisements and scams. The project aims to protect users by detecting scam patterns and profiling vulnerable users based on their exposure to fraudulent content.

## Features
- **Cyber Scam Detection**: Uses machine learning models to detect scam ads.
- **Multimodal Analysis**: Processes text, images, and videos.
- **User Profiling**: Identifies users susceptible to scams using NLP and clustering algorithms.
- **Real-time Alerts**: Notifies users of potential scams via WebSockets or Firebase Cloud Messaging (FCM).
- **Scam Data Collection**: Uses APIs and web scraping for data collection.

## Technologies Used
- **Programming Language**: Python
- **Data Collection**: Facebook Graph API, Twitter API, Selenium, BeautifulSoup
- **Machine Learning**: Random Forest, Gradient Boosting, Neural Networks
- **Deep Learning**: CNNs, Transformers (BERT, RoBERTa)
- **Database**: MongoDB, PostgreSQL
- **Deployment**: AWS/GCP
- **Messaging**: WebSockets, Kafka, RabbitMQ

## Installation
```bash
# Clone the repository
git clone https://github.com/Abishekrz/Adshield_tnhackathon_2024.git
cd Adshield_tnhackathon_2024

# Install dependencies
pip install -r requirements.txt

# Run the project
python "filename".py

```
# Contribution

Feel free to open issues and submit pull requests to improve AdShield.

# License

This project is open-source and available under the MIT License.
