import os
import csv
import PyPDF2
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from groq import Groq

# ================================================
# Carregar variáveis de ambiente
# ================================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ================================================
# Ler CSV de usuários (para autenticação)
# ================================================
users_file = "users.csv"
users_data = []
with open(users_file, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        users_data.append(row)

def authenticate_user(username: str, password: str) -> bool:
    for user in users_data:
        if user["username"] == username and user["password"] == password:
            return True
    return False

# ================================================
# Mapear cada contrato a um arquivo CSV específico
# ================================================
contract_csv_map = {
    "1": "1_manutencao.csv",
    "5": "5_consumo_prestacaoservico.csv",
    "10": "10_trabalho.csv",
    # etc. (adicione os outros contratos conforme você criar os arquivos)
}

def load_contract_requirements(contract_id: str):
    """
    Carrega o arquivo CSV específico para o contrato.
    Ex: se contract_id = '5', abre '5_consumo_prestacaoservico.csv'.
    Retorna uma lista de dicionários (each row).
    """
    if contract_id not in contract_csv_map:
        # Retorna lista vazia ou levanta exceção caso não haja mapeamento
        return []

    data_file = contract_csv_map[contract_id]
    
    rows = []
    with open(data_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=",")
        for r in reader:
            rows.append(r)
    return rows

# ================================================
# Inicializar LLMs
# ================================================
def initialize_embeddings(provider="openai"):
    if provider == "openai":
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
        return llm
    elif provider == "groq":
        groq_client = Groq(api_key=GROQ_API_KEY)
        return groq_client
    else:
        raise ValueError("Provedor inválido. Use 'openai' ou 'groq'.")

# ================================================
# Geração de resposta 
# ================================================
def generate_response(pdf_text: str, selected_contract: str, llm_or_groq, analysis_mode: str) -> str:
    """
    analysis_mode: "Apenas Requisitos" ou "Completo".
    """
    # 1) Extrair ID do contrato
    contract_id = selected_contract.split("-")[0].strip()  # e.g. '5'
    
    # 2) Carregar requisitos do CSV específico
    rows = load_contract_requirements(contract_id)
    if not rows:
        return f"Não encontrei arquivo de requisitos para o contrato de ID {contract_id}."

    # 3) Montar texto com base nos requisitos
    # Por exemplo, iremos concatenar todos os requisitos em um texto para a IA
    requirements_text = ""
    for row in rows:
        requirements_text += (
            f"- ({row.get('id')}) {row.get('tema')}: {row.get('requisito')} "
            f"[Fundamento: {row.get('fundamento_legal')}] "
            f"(Prioridade: {row.get('prioridade')})\n"
        )

    if analysis_mode == "Apenas Requisitos":
        prompt = f"""
        Você é um assistente virtual especializado em análise de contratos.

        TEXTO DO CONTRATO:
        {pdf_text}

        REQUISITOS (DO CSV):
        {requirements_text}

        INSTRUÇÕES:
        1. Para cada requisito, procure termos iguais ou equivalentes (sinônimos) no contrato.
        2. Se encontrar, use o ícone ✅. Cite o trecho exato do contrato (ou parte dele) como evidência.
        3. Se não encontrar nada relevante, use o ícone ❌.
        4. Não agrupe requisitos; analise cada ID separadamente e retorne na resposta o ID e o tema.
        5. Conclua com sugestões de melhoria (💡).
        """

    else:
        # "Completo": cláusula a cláusula
        prompt = f"""
        Você é um assistente virtual especializado em análise de contratos.

        TEXTO DO CONTRATO:
        {pdf_text}

        Estes são os requisitos pertinentes a esse tipo de contrato (ID {contract_id}):
        {requirements_text}

        Por favor, faça uma ANÁLISE COMPLETA das cláusulas:
        - Identifique cada cláusula no texto do contrato e resuma.
        - Identifique possíveis ilicitudes ou incongruências.
        - Depois aponte onde os requisitos estão atendidos (ou não).
        - Ao final, inclua sugestões de melhoria com o ícone 💡.
        """

    # 4) Chamar LLM
    if hasattr(llm_or_groq, "predict"):
        # Caso seja um ChatOpenAI (Langchain)
        response = llm_or_groq.predict(prompt)
        return response
    elif hasattr(llm_or_groq, "__call__"):
        # Versão de ChatOpenAI que funciona como call
        response = llm_or_groq(prompt)
        if hasattr(response, "content"):
            return response.content
        return str(response)
    elif hasattr(llm_or_groq, "chat"):
        # GROQ
        messages = [
            {"role": "system", "content": "Você é um assistente jurídico especializado."},
            {"role": "user", "content": prompt}
        ]
        chat_completion = llm_or_groq.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
        )
        if hasattr(chat_completion, "choices") and len(chat_completion.choices) > 0:
            return chat_completion.choices[0].message.content
        else:
            return "Erro ao processar a resposta com a API GROQ."
    else:
        return "Provedor de IA inválido ou não suportado."

# ================================================
# Processar PDF
# ================================================
def process_pdf(file) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        if not text.strip():
            raise ValueError("Nenhum texto encontrado no PDF.")
        return text
    except Exception as e:
        st.error(f"Erro ao processar o PDF: {e}")
        return ""

# ================================================
# MAIN
# ================================================
def main():
    st.set_page_config(page_title="Texto Manager - Tahech", page_icon=":bird:")
    st.title("Analisador de Cláusulas Contratuais - Tahech Advogados")

    # Autenticação
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        st.subheader("Login")
        username = st.text_input("Usuário:")
        password = st.text_input("Senha:", type="password")
        if st.button("Entrar"):
            if authenticate_user(username, password):
                st.session_state["logged_in"] = True
                st.success(f"Bem-vindo, {username}!")
            else:
                st.error("Usuário ou senha inválidos.")
        return

    # Se logado:
    if "user_text" not in st.session_state:
        st.session_state["user_text"] = None

    provider = st.radio("Escolha o provedor de API:", ("openai", "groq"))
    llm_or_groq = initialize_embeddings(provider)

    analysis_mode = st.radio(
        "Escolha o modo de análise:",
        ("Apenas Requisitos", "Completo")
    )

    input_mode = st.radio("Modo de entrada do contrato:", ("Carregar PDF", "Inserir Manualmente"))
    if input_mode == "Carregar PDF":
        uploaded_file = st.file_uploader("Carregue um arquivo PDF", type="pdf")
        if uploaded_file is not None:
            text = process_pdf(uploaded_file)
            if text:
                st.session_state["user_text"] = text
                st.success("Texto processado!")
    else:
        user_input = st.text_area("Digite o texto do contrato:")
        if user_input:
            st.session_state["user_text"] = user_input
            st.success("Texto processado!")

    # Exemplo: ID 5 se refere ao CSV '5_consumo_prestacaoservico.csv'
    # Mas na interface, você pode ter combos com todos os contratos
    contract_types = [
        "1 - Contrato de manutenção de serviço",
        "5 - Contrato de Consumo ou prestação de serviços",
        "10 - Contrato de trabalho"
        # etc...
    ]
    selected_contract = st.selectbox("Selecione a característica contratual (ID - Nome):", contract_types)

    if st.session_state["user_text"] and st.button("Analisar Informação"):
        st.write("Gerando análise...")
        result = generate_response(
            pdf_text=st.session_state["user_text"],
            selected_contract=selected_contract,
            llm_or_groq=llm_or_groq,
            analysis_mode=analysis_mode
        )
        st.subheader("Resposta Gerada")
        st.text_area("Resultado da Análise", value=result, height=300, disabled=True)

if __name__ == '__main__':
    main()
