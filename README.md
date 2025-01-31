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

