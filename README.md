# 📝 AI Blog Writing Agent

An AI-powered **multi-agent blog generation system** built using **LangGraph**, **LangChain**, **Groq LLMs**, **Google Gemini**, **Tavily Search**, and **Streamlit**.

The application automatically plans blog content, performs web research when required, generates well-structured Markdown articles, creates AI-generated images, and provides an interactive interface to preview and download blogs.

---

## 🚀 Features

- 🤖 Multi-agent workflow using LangGraph
- 📝 Automatic blog planning and outlining
- 🌐 Web research using Tavily Search
- ✍️ AI-generated blog content with Groq LLMs
- 🖼️ AI image generation with Google Gemini
- 📄 Markdown blog generation
- 👀 Live blog preview using Streamlit
- 📥 Download blogs as Markdown or ZIP bundles
- 📚 View previously generated blogs

---

## 🏗️ System Architecture

```
                 User Topic
                      │
                      ▼
            ┌─────────────────┐
            │  Planner Agent  │
            └─────────────────┘
                      │
                      ▼
            ┌─────────────────┐
            │ Research Agent  │
            │   (Tavily API)  │
            └─────────────────┘
                      │
                      ▼
            ┌─────────────────┐
            │ Writing Agent   │
            │  (Groq LLM)     │
            └─────────────────┘
                      │
                      ▼
            ┌─────────────────┐
            │ Image Generator │
            │ (Gemini API)    │
            └─────────────────┘
                      │
                      ▼
            ┌─────────────────┐
            │ Markdown Output │
            └─────────────────┘
                      │
                      ▼
            ┌─────────────────┐
            │ Streamlit UI    │
            └─────────────────┘
```

---

## 🛠️ Tech Stack

- Python
- LangGraph
- LangChain
- Groq API
- Google Gemini API
- Tavily Search API
- Streamlit
- Pandas
- Pydantic

---

## 📂 Project Structure

```text
AI_Blog_Writing_Agent/
│
├── bwa_backend.py              # LangGraph backend
├── bwa_frontend.py             # Streamlit frontend
├── generated_blogs/            # Generated Markdown blogs
├── notebooks/
│   └── bwa_backend.ipynb
├── images/                     # Generated images
├── .env.example
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/Harseerat1001/AI-Blog-Writing-Agent.git

cd AI-Blog-Writing-Agent
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Create a `.env` file

```env
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### Run the application

```bash
streamlit run bwa_frontend.py
```

---

## 💻 Usage

1. Launch the Streamlit application.
2. Enter the blog topic.
3. Click **Generate Blog**.
4. The system automatically:
   - Plans the article
   - Performs web research (if needed)
   - Generates the blog
   - Creates AI-generated images
   - Produces a downloadable Markdown file
5. Preview and download the generated blog.

---

## 📁 Example Generated Blogs

- Self Attention in Transformer Architecture
- State of Multimodal LLMs in 2026
- Tech Jobs Evolution since 2023
- Understanding Self Attention in Transformer Architecture

---

## 🎯 Future Improvements

- Export blogs as PDF and DOCX
- Multiple LLM support
- Citation generation
- User authentication
- Blog editing interface
- Docker deployment
- Cloud storage integration
- Persistent database for blog history

---

## 👨‍💻 Author

**Harseerat Kaur**

B.Tech Computer Science Engineering  
Thapar Institute of Engineering and Technology

GitHub: https://github.com/Harseerat1001

---

## ⭐ If you found this project useful, consider giving it a star!
