import os
import PyPDF2
from groq import Groq
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Carregar variáveis de ambiente
load_dotenv()

# Chaves de API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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
Você é um assistente virtual especializado em análise de contratos. Sua função é analisar contratos recebidos RESTRITIVAMENTE base nos requisitos fornecidos no banco de dados e na característica contratual selecionada pelo usuário.

Aqui está a mensagem recebida do solicitante:
{message}

O usuário identificou que esta informação está relacionada ao seguinte tipo de contrato:
{contract_type}

Aqui estão os requisitos relacionados encontrados no banco de dados:
{best_practice}

Sua tarefa é:
1. Identificar as partes do contrato (contratante e contratada) e apresentá-las no seguinte formato:
   - a) CONTRATANTE: [Identifique no texto do contrato]
   - b) CONTRATADA: [Identifique no texto do contrato]

2. Verificar os requisitos listados no banco de dados:
   - Para cada requisito localizado, informe o nome do requisito, a cláusula e o parágrafo correspondentes. Apresente no seguinte formato:
     - a) [Nome do Requisito] (Cláusula X, Parágrafo Y)
   - Para cada requisito não localizado, informe explicitamente no seguinte formato:
     - a) [Nome do Requisito Não Encontrado]

Exemplo de resposta formatada:

1. Partes do contrato:
   - a) CONTRATANTE: [Exemplo]
   - b) CONTRATADA: [Exemplo]

2. Requisitos localizados:
   - a) [Requisito 1] (Cláusula X, Parágrafo Y)
   - b) [Requisito 2] (Cláusula X, Parágrafo Y)

3. Requisitos não localizados:
   - a) [Requisito Não Localizado 1]
   - b) [Requisito Não Localizado 2]

Certifique-se de seguir o formato apresentado e de separar claramente os itens por linhas distintas para manter a organização e a clareza da resposta. Não combine múltiplos itens em uma única linha. sEJA RESTRITO A BASE DE DADOS FORNECIDAS, SEM ACRESCENTAR ITENS.
"""

    if provider == "openai":
        response = llm(prompt)
        return response.content if hasattr(response, 'content') else "Erro ao processar a resposta."
    elif provider == "groq":
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Você é um assistente jurídico."},
                {"role": "user", "content": prompt}
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

# Interface Streamlit
def main():
    st.set_page_config(page_title="Texto Manager - Tahech", page_icon=":bird:")
    st.header("Analisador de Cláusulas Contratuais - Tahech Advogados")
    st.write("Bem-vindo ao sistema de análise de cláusulas contratuais.")

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
            "Contrato social": "Objeto ou finalidade do contrato. Definição do capital social. Direitos e deveres dos sócios. Conformidade com a legislação societária.",
            "Contrato de compra e venda": "Objeto ou finalidade do contrato. Identificação clara das partes. Descrição do bem objeto da compra. Valor, forma e prazo de pagamento. Garantias e condições de rescisão.",
            "Contratos administrativos": "Objeto ou finalidade do contrato. Conformidade com a legislação administrativa. Definição clara das obrigações das partes. Garantia de execução. Respeito aos princípios da legalidade e moralidade.",
            "Contrato de consumo": "Objeto ou finalidade do contrato. Regras para rescisão contratual. Conformidade com os termos do Código de Defesa do Consumidor.",
            "Contrato de sociedade (Acordo de Quotistas)": "Objeto ou finalidade do contrato. Estabelecimento de direitos e deveres dos sócios. Regras para distribuição de lucros e administração. Mecanismos para resolução de conflitos.",
            "Contratos mercantis": "Objeto ou finalidade do contrato. Definição de direitos e obrigações das partes. Conformidade com a legislação comercial. Regras de pagamento e rescisão.",
            "Contrato de prestação de serviços": "Objeto ou finalidade do contrato. Descrição clara dos serviços prestados. Regras de pagamento e prazo. Garantias de qualidade. Mecanismos para resolução de disputas.",
            "Contrato eletrônico": "Objeto ou finalidade do contrato. Conformidade com a legislação digital, incluindo a LGPD. Clareza nas condições de oferta. Garantias legais de consumo. Proteção de dados e privacidade.",
            "Contrato de trabalho": "Objeto ou finalidade do contrato. Definição das funções e responsabilidades do trabalhador. Condições de remuneração e jornada de trabalho. Conformidade com a legislação trabalhista. Previsão de rescisão."
        }
        selected_contract = st.selectbox("Selecione a característica contratual:", list(contract_types.keys()))

        if st.button("Analisar Informação"):
            st.write("Gerando uma análise com base na seleção do contrato...")
            result = generate_response(st.session_state["user_text"], selected_contract, db, llm, provider)
            st.markdown("### Resposta Gerada")
            st.write(result)

if __name__ == '__main__':
    main()
