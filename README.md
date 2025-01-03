
# FinanceBot: News Research Tool

FinanceBot is a user-friendly Streamlit-based application designed to help you analyze and summarize news articles. Simply provide URLs of news articles, and the bot will process the content, store it in a FAISS index, and allow you to ask questions about the articles with answers generated by OpenAI's language model.


## Features

* Process News Articles: Input up to three article URLs to extract and analyze content.

* Text Splitting: Automatically splits long articles into manageable chunks for efficient processing.

* Embeddings and FAISS Index: Creates vector embeddings of the text and stores them in a FAISS index for quick retrieval.

* Question Answering: Ask questions about the content of the articles and get accurate answers with source citations.


## Installation

* Clone repository:

```bash
https://github.com/SahilChukka19/FinanceBot.git
cd financebot
```
* Install required dependencies:
```bash
pip install -r requirments.txt
```
* Add your OpenAI API key to the .env file:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```


## Usage

* Run the Streamlit application:
```bash
streamlit run app.py
```
* Open your web browser and navigate to the provided local URL (e.g., http://localhost:8501).

* Enter up to three URLs of news articles in the sidebar.

* Click on the Process URLs button to analyze the content.

* Ask questions about the articles in the main input box and view answers along with sources

## Important Notes

* Ensure your .env file is in the same directory as app.py.

* Keep your OpenAI API key secure and do not share it publicly.

## 🔗 Links
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sahil-chukka)

[![gmail](https://img.shields.io/badge/gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:sahil.chukka@gmail.com)
