from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool , initialize_agent, AgentType
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, Request
from pydantic import BaseModel
from uvicorn import Config, Server
import nest_asyncio
import os
from dotenv import load_dotenv
load_dotenv()

print("Chave de API")
print(os.environ.get("OPENAI_API_KEY"))
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7,
    api_key=os.environ.get("OPENAI_API_KEY")
)

nest_asyncio.apply()

app = FastAPI()
################################ FUNÇOES QUE SERÃO TRANSFORMADAS EM TOOLS ######################################

def func_listar_cursos_old(*args, **kwargs):
    cursos = []
    url_mooc = "https://leds.academy/?redirect=0"
    r = requests.get(url_mooc)
    soup = BeautifulSoup(r.text, "html.parser")
    for a in soup.find_all("a", class_="aalink", href=True):
        nome = a.text.strip()
        link = a["href"]
        cursos.append({"nome": nome, "link": link})
        print(f"[DEBUG] Curso encontrado: {nome} -> {link}")
    return cursos


def func_listar_cursos(*args, **kwargs):
    cursos = []
    url_mooc = "https://mdev.titasdarobotica.com/?redirect=0"
    r = requests.get(url_mooc)
    soup = BeautifulSoup(r.text, "html.parser")
    for a in soup.find_all("a", class_="aalink", href=True):
        nome = a.text.strip()
        link = a["href"]
        cursos.append(f"{nome}: {link}")
    return "Cursos disponíveis:\n" + "\n".join(cursos)



def func_filtrar_cursos_relevantes(*args, **kwargs):
    cursos = kwargs.get("cursos", [])
    criterio = kwargs.get("criterio", "")
    cursos_relevantes = []
    for curso in cursos:
        url = curso["link"]
        r = requests.get(url)
        soup = BeautifulSoup(r.text, "html.parser")
        conteudo = soup.get_text().lower()
        if criterio.lower() in conteudo or criterio.lower() in curso["nome"].lower():
            cursos_relevantes.append(curso)
    return cursos_relevantes


####################################################DEFINIÇÃO DAS TOOLS##################################

listagem_cursos_tool = Tool(
    name="Listagem de Cursos",
    func=func_listar_cursos,
    description="Lista os cursos disponíveis na plataforma de cursos online dada a URL principal."
)
filtragem_cursos_tool = Tool(
    name="Filtro de Cursos Relevantes com base no Prompt",
    func=func_filtrar_cursos_relevantes,
    description=(
        "Use esta ferramenta apenas quando o usuário pedir para filtrar cursos por tema, palavra-chave ou assunto específico. "
        "Não use para listar todos os cursos."
    )
)

############################# PROMPT PERSONALIZADO ########################################################

prompt_tutor = ChatPromptTemplate.from_messages([
    ("system", 
        "Você é especialista tutor de tecnologia. "
        "Sua resposta deve estar somente em português do Brasil (pt-br). "
        "- Não use JSON nem outro formato de código."
    ),
    ("user", "pergunta: {pergunta}")
])


#################### INICIALIZAÇÃO DOS AGENTES #############################################################

agente_tutor = initialize_agent(
    tools=[listagem_cursos_tool, filtragem_cursos_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,   ##STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION   ZERO_SHOT_REACT_DESCRIPTION  
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3,
    prompt = prompt_tutor,
    early_stopping_method="generate" ,
)

######################## ORQUESTRADOR ####################################################
def orquestrador(pergunta):
    resposta = agente_tutor.run(pergunta)
    return resposta

# modelo do JSON esperado
from fastapi.middleware.cors import CORSMiddleware # Novo Import

origins = [
    "*", # Permite acesso de qualquer origem (ideal para desenvolvimento local/ngrok)
    # Se você quiser ser mais restrito, use a URL do seu Moodle: "http://moodle.local",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Permite todos os métodos (GET, POST, OPTIONS, etc.)
    allow_headers=["*"], # Permite todos os cabeçalhos
)
# --- Fim da Configuração do CORS ---
class LLMRequest(BaseModel):
    model: str
    prompt: str
    stream: bool = False


@app.post("/api/generate")
async def chamar_llm(data: LLMRequest):
    try:
        resposta = orquestrador(data.prompt)
        return {
            "model": data.model,
            "prompt": data.prompt,
            "stream": data.stream,
            "output": resposta
        }
    except Exception as e:
        return {"error": str(e)}


#config = Config(app=app, host="0.0.0.0", port=8000, log_level="info")
#server = Server(config)
#await server.serve()

#uvicorn sugestao:app --reload --host 0.0.0.0 --port 8000


