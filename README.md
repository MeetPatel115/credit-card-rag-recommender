# 💳 RAG-Based Credit Card Recommendation System

An AI-powered credit card advisor that recommends the best credit cards
based on a user's spending habits. The system compares recommendations
from a standard LLM prompt and a Retrieval-Augmented Generation (RAG)
pipeline to demonstrate how RAG improves recommendation accuracy.

------------------------------------------------------------------------

## 🚀 Project Overview

Choosing the right credit card can significantly impact cashback, travel
rewards, and yearly savings. However, users often struggle to compare
multiple cards with complex reward structures.

This project solves the problem by:

1.  Collecting credit card data from bank websites
2.  Structuring reward and fee information
3.  Creating embeddings for card descriptions
4.  Storing them in a vector database
5.  Using RAG to retrieve the most relevant cards
6.  Generating recommendations using an LLM

The system then compares:

-   Standard LLM recommendation
-   RAG-based recommendation

------------------------------------------------------------------------

## 🧠 Architecture

User Query → Embedding Model → Vector Database → Relevant Card Retrieval
→ LLM → Top Card Recommendations

------------------------------------------------------------------------

## 📊 Key Features

-   Web scraping of credit card information
-   Structured dataset of credit card rewards
-   Embedding generation for semantic search
-   Vector database for similarity retrieval
-   RAG pipeline for intelligent recommendations
-   Comparison between LLM vs RAG performance
-   Test framework to evaluate recommendation quality

------------------------------------------------------------------------

## 🗂 Project Structure

credit-card-rag/

data/ - raw/ - processed/

scraper/ - scrape_amex.py - scrape_cibc.py - scrape_scotia.py

rag/ - create_embeddings.py - vector_store.py - rag_pipeline.py

llm/ - generate_response.py

evaluation/ - test_queries.json - compare_results.py

app.py\
requirements.txt\
README.md

------------------------------------------------------------------------

## 📥 Data Collection

Credit card information is scraped from major Canadian banks including:

-   American Express
-   CIBC
-   Scotiabank

Collected fields include:

-   Card name
-   Annual fee
-   Welcome bonus
-   Cashback categories
-   Travel rewards
-   Insurance benefits
-   Interest rates
-   Eligibility requirements

------------------------------------------------------------------------

## 🔍 Retrieval-Augmented Generation (RAG)

Instead of asking the LLM directly, the system:

1.  Converts card descriptions into vector embeddings
2.  Stores them in a vector database
3.  Retrieves the most relevant cards based on user spending
4.  Sends retrieved cards to the LLM
5.  Generates recommendations based on factual card data

This reduces hallucination and improves recommendation accuracy.

------------------------------------------------------------------------

## 🤖 LLM Model

The project uses Ollama to run local LLM models such as:

-   Llama3
-   Mistral
-   Gemma

Example setup:

ollama pull llama3

------------------------------------------------------------------------

## 🧪 Evaluation

Test queries simulate real user spending patterns.

Example:

{ "monthly_spending": { "groceries": 600, "travel": 300, "gas": 200,
"restaurants": 250 } }

The system generates:

-   Top 3 cards from Standard LLM
-   Top 3 cards from RAG

Then potential savings are calculated.

------------------------------------------------------------------------

## ⚙️ Installation

Install dependencies:

pip install -r requirements.txt

Install Ollama: https://ollama.com

Pull model:

ollama pull llama3

------------------------------------------------------------------------

## ▶️ Run the Project

Create embeddings:

python create_embeddings.py

Run recommendation system:

python app.py

------------------------------------------------------------------------

## 🧪 Run Evaluation

python compare_results.py

------------------------------------------------------------------------

## 🧩 Technologies Used

-   Python
-   Selenium / BeautifulSoup
-   Sentence Transformers
-   FAISS / Chroma Vector DB
-   Ollama LLM
-   Retrieval-Augmented Generation (RAG)

------------------------------------------------------------------------

## 👨‍💻 Author

Meet Patel\
AI / Machine Learning Engineer\
Toronto, Canada\
Email: meet115patel@gmail.com
