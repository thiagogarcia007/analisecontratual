import os
import PyPDF2
from groq import Groq
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from streamlit_authenticator import Authenticate
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials

# Carregar variáveis de ambiente
load_dotenv()

# Chaves de API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI")

# Inicializar cliente GROQ
groq_client = Groq(api_key=GROQ_API_KEY)

# Carregar o CSV
data_file = "qa_with_id_first_column.csv"
loader = CSVLoader(file_path=data_file)
documents = loader.load()

# Configurar embeddings e FAISS
def initialize_embeddings(provider):
    if provider == "openai":
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
        db = FAISS.from_documents(documents, embeddings)
        return db, llm
    elif provider == "groq":
        db = [{"content": doc.page_content} for doc in documents]
        return db, groq_client
    else:
        raise ValueError("Provedor de API inválido. Use 'openai' ou 'groq'.")

# Função para buscar informações similares
def retrieve_info(query, db, provider):
    if provider == "openai":
        similar_response = db.similarity_search(query, k=1)
        return [doc.page_content for doc in similar_response]
    elif provider == "groq":
        messages = [
            {"role": "system", "content": "Você é um assistente jurídico."},
            {"role": "user", "content": f"Encontre informações relacionadas a: {query}"}
        ]
        response = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile"
        )
        if hasattr(response, 'choices') and len(response.choices) > 0:
            return [response.choices[0].message.content]
        else:
            return ["Erro: Nenhuma resposta válida recebida da API GROQ."]

# Função para gerar respostas
def generate_response(message, contract_type, db, llm, provider):
    best_practice = retrieve_info(message, db, provider)
    if not best_practice:
        return "Desculpe, não encontrei informações suficientes no banco de dados para responder a esta pergunta."

    prompt = f"""
    Você é um assistente virtual especializado em análise de contratos. Sua função é analisar contratos recebidos com base nos requisitos fornecidos no banco de dados e na característica contratual selecionada pelo usuário.

    Aqui está a mensagem recebida do solicitante:
    {message}

    O usuário identificou que esta informação está relacionada a seguinte característica do contrato:
    {contract_type}

    Aqui estão os requisitos relacionados encontrados no banco de dados:
    {best_practice}

    Sua tarefa é identificar as partes do contrato e verificar os requisitos listados no banco de dados, fornecendo na resposta quando identificado o requisito, um pequeno trecho do conteúdo identificado do requisito e identificação da clausula. Identifique os requisitos não atendidos. Ao final, faça uma breve sugestão para que se possa melhorar a qualidade do contrato.
    
    //
    Seja restrito aos requisitos informados considerando apenas a opção informada pelo usuário em {message}, {contract_type} e o banco de dados  {best_practice} e formate o texto no formato esperado abaixo:
        PARTES DO CONTRATO


        Contratante: [NOME DO CONTRATANTE]

        Contratada: [NOME DO CONTRATADO]



        REQUISITOS DO CONTRATO

        [REQUISITO 1]  [CLAUSULA X, PARÁGRAFO Y];

        [REQUISITO 2]  [CLAUSULA X, PARÁGRAFO Y];

        [REQUISITO 3]  [CLAUSULA X, PARÁGRAFO Y]; 

        ...


        Requisitos Não atendidos: 

        [REQUISITO 1]  ;

        [REQUISITO 2]  ;

        [REQUISITO 3]  ; 



        [ANÁLISE FINAL DA IA] ...;
...

    """

    prompt2 = f"""
    Você é um assistente virtual especializado em análise de contratos. Sua função é analisar contratos recebidos com base nos requisitos fornecidos no banco de dados e na característica contratual selecionada pelo usuário.

    Aqui está a mensagem recebida do solicitante:
    {message}

    O usuário identificou que esta informação está relacionada a seguinte característica do contrato:
    {contract_type}

    Aqui estão os requisitos relacionados encontrados no banco de dados:
    {best_practice}

    Sua tarefa é identificar as partes do contrato e verificar os requisitos listados no banco de dados, fornecendo na resposta quando identificado o requisito, um pequeno trecho do conteúdo identificado do requisito e identificação da clausula. Identifique os requisitos não atendidos. Ao final, faça uma breve sugestão para que se possa melhorar a qualidade do contrato.
    
    //
    Seja restrito aos requisitos informados considerando apenas a opção informada pelo usuário em {message}, {contract_type} e o banco de dados  {best_practice}.

    """

    if provider == "openai":
        response = llm(prompt)
        return response.content if hasattr(response, 'content') else "Erro ao processar a resposta."
    elif provider == "groq":
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Você é um assistente jurídico e deve se ater com a solicitação e os dados que possui."},
                {"role": "user", "content": prompt2}
            ],
            model="llama-3.3-70b-versatile",
        )
        if hasattr(chat_completion, 'choices') and len(chat_completion.choices) > 0:
            return chat_completion.choices[0].message.content
        else:
            return "Erro ao processar a resposta com a API GROQ."

# Função para processar PDF
def process_pdf(file):
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
        return None

# Função de autenticação com Google
def authenticate_user():
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": GOOGLE_CLIENT_ID,
                "project_id": "project-id",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uris": [REDIRECT_URI]
            }
        },
        scopes=["https://www.googleapis.com/auth/userinfo.email"]
    )

    authorization_url, _ = flow.authorization_url(prompt="consent")
    st.markdown(f"[Faça login com Google]({authorization_url})")
    code = st.text_input("Cole o código de autorização:")

    if code:
        flow.fetch_token(code=code)
        credentials = flow.credentials
        user_info_url = "https://www.googleapis.com/oauth2/v1/userinfo"
        response = requests.get(user_info_url, headers={"Authorization": f"Bearer {credentials.token}"})
        if response.status_code == 200:
            user_info = response.json()
            return user_info.get("email")
        else:
            st.error("Erro ao obter informações do usuário.")
            return None

# Interface Streamlit
def main():
    st.set_page_config(page_title="Texto Manager - Tahech", page_icon=":bird:")
    st.header("Analisador de Cláusulas Contratuais - Tahech Advogados")

    # Autenticação do usuário
    #user_email = authenticate_user()
    #if not user_email:
    #    st.stop()

    #st.success(f"Bem-vindo, {user_email}!")

    if "user_text" not in st.session_state:
        st.session_state["user_text"] = None

    # Botão seletor para o provedor de API
    provider = st.radio("Escolha o provedor de API:", ("openai", "groq"))
    db, llm = initialize_embeddings(provider)

    input_mode = st.radio("Escolha como deseja fornecer o texto:", ("Carregar PDF", "Inserir Manualmente"))

    if input_mode == "Carregar PDF":
        uploaded_file = st.file_uploader("Carregue um arquivo PDF", type="pdf")
        if uploaded_file:
            st.write("Processando o arquivo PDF...")
            text = process_pdf(uploaded_file)
            if text:
                st.session_state["user_text"] = text
                st.success("Texto processado com sucesso!")
            else:
                st.warning("Não foi possível processar o arquivo. Por favor, carregue outro arquivo ou insira o texto manualmente.")
    elif input_mode == "Inserir Manualmente":
        user_input = st.text_area("Digite o texto do cliente:")
        if user_input:
            st.session_state["user_text"] = user_input
            st.success("Texto processado com sucesso!")

    if st.session_state["user_text"]:
        contract_types = {
            "Contrato social": "Definição do capital social. Direitos e deveres dos sócios.",
            "Contrato de compra e venda": "Descrição do bem objeto da compra. Valor, forma e prazo de pagamento.",
            "5 - Contrato de Consumo": "Contrato pautado no Código de Defesa do Consumidor.",
        }
        selected_contract = st.selectbox("Selecione a característica contratual:", list(contract_types.keys()))

        if st.button("Analisar Informação"):
            st.write("Gerando uma análise com base na seleção do contrato...")
            result = generate_response(st.session_state["user_text"], selected_contract, db, llm, provider)
            st.markdown("### Resposta Gerada")
            st.write(result)

if __name__ == '__main__':
    main()
