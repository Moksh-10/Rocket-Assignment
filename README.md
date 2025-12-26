1. Clone the repository

git clone https://github.com/Moksh-10/Rocket-Assignment.git

cd Rocket-Assignment

2. Install uv

pip install uv

3. Create & activate virtual environment

macOS / Linux

uv venv .venv

source .venv/bin/activate

Windows (PowerShell)

uv venv .venv

.venv\Scripts\activate

4. Install dependencies

uv pip install -r requirements.txt

5. Set environment variables

Create a .env file in the project root:


TAVILY_API_KEY=your_tavily_api_key

GROQ_API_KEY=your_groq_api_key

TOKENIZERS_PARALLELISM=false

6. Initialize the database (first run only)

python database.py

7. Run the research pipeline (CLI)

python pipeline.py

8. Run the Streamlit UI

streamlit run app.py
