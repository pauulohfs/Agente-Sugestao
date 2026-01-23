import os
import requests
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuração de logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# --- Configurações ---
MOODLE_URL = "https://mdev.titasdarobotica.com/webservice/rest/server.php"
MOODLE_TOKEN = "c994b2d14206cd86ac34953365b24aa6"
INDEX_PATH = "faiss_index_leds"
UPDATE_INTERVAL_SECONDS = 86400

vector_db: Optional[FAISS] = None

class QueryRequest(BaseModel):
    prompt: str

# --- Base de Conhecimento Fixa (FAQ / Regras de Negócio) ---
FAQ_LEDS_ACADEMY = """
SOPRE A LEDS ACADEMY:
- O que é: Plataforma de aprendizagem por trilhas e cursos.
- Como começar: (1) entender objetivo, (2) sugerir trilha, (3) indicar sequência de cursos.
- Diferença entre curso e trilha: Curso é unidade de aprendizagem; Trilha organiza vários cursos em sequência para um objetivo.
- Tempo de conclusão: Depende do ritmo e horas semanais. Estimar rotas realistas (ex: 3-6 semanas).
- Mais de um curso: Pode fazer simultaneamente. Recomendado combinar um principal com um de apoio (ex: Backend + Git).
- Recomendação por perfil: Considera objetivos, nível atual, preferências e lacunas.
- Acesso: Cursos ficam na página inicial e nas trilhas.

TRILHAS SUGERIDAS:
- Backend: Git e GitHub -> Desenvolvimento Backend (APIs, DB) -> DevOps.
- Iniciante: Começar por Git e GitHub.
- IA: Requer base de programação. Rota: Git -> Desenvolvimento -> IA.
- Frontend: Foco em UI/UX e interface.
"""

# --- Funções de Ingestão ---

def fetch_only_course_names() -> str:
    logger.info("Coletando nomes dos cursos do Moodle...")
    course_names = []
    try:
        params = {
            'wstoken': MOODLE_TOKEN,
            'wsfunction': 'core_course_get_courses',
            'moodlewsrestformat': 'json'
        }
        res = requests.get(MOODLE_URL, params=params, timeout=20).json()
        if not isinstance(res, list): return ""
        for course in res:
            if course.get('id') == 1: continue 
            name = course.get('fullname')
            if name: course_names.append(f"CURSO DISPONÍVEL: {name}")
        return "\n".join(course_names)
    except Exception as e:
        logger.error(f"Erro na coleta: {e}")
        return ""

def sync_vector_store():
    global vector_db
    raw_data = fetch_only_course_names()
    if not raw_data: return
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(raw_data)
    embeddings = OpenAIEmbeddings()
    new_db = FAISS.from_texts(chunks, embeddings)
    new_db.save_local(INDEX_PATH)
    vector_db = new_db
    logger.info("Base vetorial sincronizada!")

async def scheduled_sync_task():
    while True:
        try: await asyncio.to_thread(sync_vector_store)
        except Exception as e: logger.error(f"Erro: {e}")
        await asyncio.sleep(UPDATE_INTERVAL_SECONDS)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_db
    if os.path.exists(INDEX_PATH):
        try:
            embeddings = OpenAIEmbeddings()
            vector_db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        except Exception as e: logger.error(f"Erro ao carregar: {e}")
    sync_task = asyncio.create_task(scheduled_sync_task())
    yield
    sync_task.cancel()

app = FastAPI(title="Leds Academy API", lifespan=lifespan)

@app.post("/api/generate")
async def ask_rag(data: QueryRequest):
    if not vector_db:
        raise HTTPException(status_code=503, detail="Base não carregada.")

    # Busca cursos dinâmicos
    docs = vector_db.similarity_search(data.prompt, k=5)
    context_courses = "\n".join([d.page_content for d in docs])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2) # Temp levemente maior para fluidez
    
    messages = [
        SystemMessage(content=(
            "Você é o assistente da Leds Academy. Responda de forma prestativa, clara e direta.\n\n"
            "REGRAS DE RESPOSTA:\n"
            "1. Se a pergunta for sobre o funcionamento da plataforma (tempo, como começar, fazer 2 cursos, etc.), use a 'BASE DE CONHECIMENTO FIXA'.\n"
            "2. Se a pergunta for sobre quais cursos existem ou sugestão de nomes, use a 'LISTA DE CURSOS DO MOODLE'.\n"
            "3. Se o usuário perguntar algo totalmente fora do contexto educacional da Leds Academy, diga que só pode ajudar com temas da plataforma.\n"
            "4. Mantenha o tom de voz das respostas exemplo fornecidas.\n\n"
            f"BASE DE CONHECIMENTO FIXA (FUNCIONAMENTO):\n{FAQ_LEDS_ACADEMY}\n\n"
            f"LISTA DE CURSOS DO MOODLE (DADOS EM TEMPO REAL):\n{context_courses}"
        )),
        HumanMessage(content=data.prompt)
    ]

    response = llm.invoke(messages)
    return {"message": response.content}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("sugestaorag:app", host="0.0.0.0", port=8000)