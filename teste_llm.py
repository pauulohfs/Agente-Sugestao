import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv
import uvicorn

# Carrega a chave da OpenAI do seu arquivo .env
load_dotenv()

app = FastAPI()

# Inicializa a LLM (GPT-3.5)
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", 
    api_key=os.getenv("OPENAI_API_KEY")
)

# Modelo de dados igual ao que o seu Moodle envia agora
class LLMRequest(BaseModel):
    model: str
    prompt: str
    course_id: int  # Capturamos o ID que você configurou no externallib.php

@app.post("/api/generate")
async def test_api(data: LLMRequest):
    print(f"--- NOVA REQUISIÇÃO ---")
    print(f"Pergunta do Moodle: {data.prompt}")
    print(f"ID do Curso recebido: {data.course_id}")

    try:
        # Pergunta simples para a LLM
        mensagem = [
            HumanMessage(content=f"O aluno do curso {data.course_id} perguntou: {data.prompt}")
        ]
        resposta = llm.invoke(mensagem)

        return {
            "model": data.model,
            "output": resposta.content,
            "status": "Sucesso! O Moodle e a IA estão conversando."
        }
    except Exception as e:
        print(f"ERRO: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)