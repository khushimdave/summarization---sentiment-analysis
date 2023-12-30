# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
import numpy as np
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelWithLMHead
from collections import OrderedDict
import streamlit_scrollable_textbox as stx
import torch
import scipy
from scipy import special
import asyncio

# Set page title and favicon
st.set_page_config(
    page_title="Summarization & Sentiment Analysis",
    page_icon="📚",
)

# Main content
st.write("""
# SUMMARIZATION & SENTIMENT ANALYSIS MODEL
You can either input a URL or some text to generate its *summary* and its *sentiment*
""")

@st.cache_resource
def load_models():
    summer_model_name = "T5_Base"
    summer_tokenizer = AutoTokenizer.from_pretrained(summer_model_name)
    summer_model = AutoModelWithLMHead.from_pretrained(summer_model_name, return_dict=True)
    
    # Load FINBERT model and tokenizer
    senti_model_name = "epoch_1"
    senti_tokenizer = AutoTokenizer.from_pretrained(senti_model_name)
    finbert = AutoModelForSequenceClassification.from_pretrained(senti_model_name)
    
    return summer_tokenizer, summer_model, senti_tokenizer, finbert


# Function for summarization and sentiment analysis
# @st.cache_data()
def summerize_and_sentiment(text):
    words = np.char.count(text, ' ') + 1        
    summary = ""  # Move the definition outside the if-else block

    if words <= 50:
        summary = "No need to summarize as it is already small."
        senti_inputs = senti_tokenizer(text, return_tensors="pt")

        
    else:
        inputs = summer_tokenizer.encode("summarize: " + text, return_tensors='pt', max_length=512, truncation=True)
        output = summer_model.generate(inputs, min_length=100, max_length=650)
        summary = summer_tokenizer.decode(output[0])
        summary = summary.replace('<pad> ', '').replace('</s>', '').replace('<pad>', '').replace('<extra_id_0>', '').replace(': ', '').replace('<unk>', '')
        sentences = summary.split(". ")
        unique_sentences = list(OrderedDict.fromkeys(sentences))
        capitalized_sentences = [sentence.capitalize() for sentence in unique_sentences]
        summary = ". ".join(capitalized_sentences)
        if not summary.endswith('.'):
            summary += '.'
            
        senti_inputs = senti_tokenizer(summary, return_tensors="pt")

    with torch.no_grad():
        logits = finbert(**senti_inputs).logits

    # Sentiment from the summary
    sentiment_scores_summary = {k: v for k, v in zip(finbert.config.id2label.values(), scipy.special.softmax(logits.numpy().squeeze()))}
    # Sentiment from the original text
    with torch.no_grad():
        logits_text = finbert(**senti_tokenizer(text, return_tensors="pt", max_length=512)).logits

    sentiment_scores_text = {k: v for k, v in zip(finbert.config.id2label.values(), scipy.special.softmax(logits_text.numpy().squeeze()))}

    # Choose the sentiment with the maximum score
    overall_sentiment = max(sentiment_scores_text, key=sentiment_scores_text.get)
    if sentiment_scores_text[overall_sentiment] < sentiment_scores_summary[overall_sentiment]:
        overall_sentiment = max(sentiment_scores_summary, key=sentiment_scores_summary.get)

    return summary, overall_sentiment, sentiment_scores_text, sentiment_scores_summary

# Radio button for input choice
choise = st.radio(
    'Choose Input Type: URL or Text?',
    ["URL", "Text"],
#     captions = ["Directly enter a URL only.", "Enter your choice of text."]
)


summer_tokenizer, summer_model, senti_tokenizer, finbert = load_models()

# Input handling based on user choice
if choise == "URL":
    url = st.text_input("Enter the URL")
    if st.button("Enter", key="url_button"):
        try:
            response = requests.get(url)
            content = BeautifulSoup(response.text, "html.parser")

            text = ""
            article_content = content.find("div", class_="content_wrapper")
            if article_content:
                paragraphs = article_content.find_all("p")
                for p in paragraphs:
                    if "Follow our live blog" in p.get_text() or "Disclaimer:" in p.get_text():
                        continue

                    cleaned_text = p.get_text().replace("\xa0", " ")  # Replacing \xa0 with a space
                    text += cleaned_text + " "

            text.strip()

            try:
                summary, overall_sentiment, sentiment_scores_text, sentiment_scores_summary = summerize_and_sentiment(text)
                st.success("SUCCESS")
                
            except Exception as e:
                st.error(f"Error: {e}")

            # Displaying results
            st.info("SCRAPPED TEXT")
            stx.scrollableTextbox(text, height=300)

            st.info("SUMMARY")
            stx.scrollableTextbox(summary, height=150)

            st.info("SENTIMENT SCORES")
            overall_sentiment_text = max(sentiment_scores_text, key=sentiment_scores_text.get)
            for label, score in sentiment_scores_text.items():
                percentage = score * 100
                st.write(f"{label.capitalize()}: {score:.4f}  ( {percentage:.2f}%)")
                
            st.info(f"OVERALL SENTIMENT")
            st.write(overall_sentiment_text.capitalize())

        except Exception as e:
            st.error(f"Error: {e}")

else:
    text = st.text_area("Enter the text")
    if st.button("Enter", key="text_button"):
        try:
            try:
                summary, overall_sentiment, sentiment_scores_text, sentiment_scores_summary = summerize_and_sentiment(text)
                st.success("SUCCESS")
                
            except Exception as e:
                st.error(f"Error: {e}")

            # Displaying results
            st.info("ENTERED TEXT")
            stx.scrollableTextbox(text, height=300)

            st.info("SUMMARY")
            stx.scrollableTextbox(summary, height=150)

            st.info("SENTIMENT SCORES")
            overall_sentiment_text = max(sentiment_scores_text, key=sentiment_scores_text.get)
            for label, score in sentiment_scores_text.items():
                percentage = score * 100
                st.write(f"{label.capitalize()}: {score:.4f}  ( {percentage:.2f}%)")
                
            st.info(f"OVERALL SENTIMENT")
            st.write(overall_sentiment_text.capitalize())

        except Exception as e:
            st.error(f"Error: {e}")
