<<<<<<< HEAD
# Adshield
## Abstract:
Social media platforms like Facebook, Instagram, and Twitter are essential hubs for advertisements, influencing billions
of users daily. However, this ecosystem is also exploited by cyber scammers, resulting in financial and psychological
harm to users. This project proposes a robust, automated system to collect, analyze, and flag potentially harmful
advertisements in real-time, leveraging cutting-edge technologies in data collection, machine learning, and natural
language processing (NLP).

## Objectives:
1. Ad Collection: Use APIs, web scraping, and browser extensions to gather advertisements and associated
metadata such as demographics, engagement metrics, and targeting criteria.
3. Data Preprocessing: Employ tools like MongoDB and PostgreSQL for structured storage and remove
duplicates or irrelevant data to ensure clean datasets.
4. Scam Detection: Develop machine learning models using Random Forest, Gradient Boosting, and neural
networks to detect patterns indicative of scams in ads.
5. User Profiling: Apply clustering algorithms and NLP (e.g., BERT, RoBERTa) to understand user behavior,
flag vulnerabilities, and predict susceptibility to scams.
6. Real-Time Alerting: Implement real-time notifications using WebSockets, Kafka, or RabbitMQ to warn users
about flagged ads and notify platforms for moderation.
=======
# AdShield - Social Media Scam Detection & User Profiling

**AdShield** is a cyber scam detection system designed for social media platforms (Twitter, Instagram, and Facebook). The project focuses on analyzing text, images, and videos to detect scam advertisements and profile users based on their vulnerability to scams.

---

## 🔍 Features
- **Social Media Scraping**: Collects user data from Instagram and Facebook.
- **Scam Detection**: Uses BERT-based NLP models for text analysis.
- **User Profiling**: Clusters users based on scam exposure.
- **Machine Learning Models**: Includes Random Forest, Gradient Boosting, and deep learning models.
- **Evaluation & Testing**: Model performance analysis with `model_eval.py`.

---

## 📁 Project Structure
- **facebook_userprofiling.py** – Profiles users on Facebook based on scam exposure.
- **scam_detection_text_bert.py**– Detects scams using a BERT model on textual data.
- **scrap_instgram_userdata.py**– Scrapes Instagram user data.
- **scrappy.py**:– Web scraping module for scam data collection.
- **userprofiling.py**:– Generalized user profiling based on scam vulnerability.
- **model_training.py**:– Trains the scam detection models.
- **model_eval.py**:– Evaluates model accuracy and performance.
- **test_data.json**:– Sample dataset for testing.
- **requirements.txt**: – Lists required dependencies.

  
---

## 🚀 Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Abishekrz/Adshield_tnhackathon_2024.git
   cd Adshield_tnhackathon_2024
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the scam detection model:**
   ```bash
   python scam_detection_text_bert.py
   ```

## Contribution

Feel free to open issues and submit pull requests to improve AdShield.

## License

This project is open-source and available under the MIT License.

>>>>>>> 69cca4e2820016741a50b80cada5e57e43adc0e9
