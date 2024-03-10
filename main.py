from crewai import Agent,Task,Crew,Process
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI,ChatOllama
import os

load_dotenv()
#api = os.getenv('OPENAI_API_KEY')
#openllm = ChatOpenAI(model_name='gpt-3.5-turbo-16k-0613',api_key=api)
ollm = ChatOllama(model='llama2')

agente_pesquisa_eventos = Agent(
    role='Pesquisador de Eventos Culturais',
    goal='Identificar os eventos culturais e sazonais mais relevantes com base nos interesses do usuário.',
    backstory="Você é um especialista em cultura e eventos, com amplo conhecimento sobre festivais, exposições artísticas e celebrações sazonais. Sua missão é descobrir eventos que ofereçam experiências autênticas e enriquecedoras.",
    verbose=True,
    llm=ollm
)

# Agente de Planejamento de Itinerários
agente_planejamento_itinerarios = Agent(
    role='Planejador de Itinerários',
    goal='Criar itinerários personalizados que integrem os eventos identificados, otimizando a experiência do usuário.',
    backstory="Com sua habilidade em logística e planejamento de viagens, você transforma a pesquisa de eventos em um itinerário detalhado, considerando localização, datas e preferências do usuário para garantir uma experiência inesquecível.",
    verbose=True,
    llm=ollm
)

# Tarefa para o Agente de Pesquisa de Eventos
tarefa_pesquisa_eventos = Task(
    expected_output='ola',
    description='''
    Identifique eventos culturais e sazonais que correspondam aos interesses e à disponibilidade do usuário delimitado entre as tags <evento></evento>. Sua resposta final deve ser uma lista de eventos recomendados, com detalhes sobre cada um, incluindo datas, localizações e uma breve descrição.
    
    <evento>
    - Disponibilidade: 16 a 18 de fevereiro de 2024
    - Eventos de interesse: Concerto de música clássica
    </evento>
    ''',
    agent=agente_pesquisa_eventos
)

# Tarefa para o Agente de Planejamento de Itinerários
tarefa_planejamento_itinerarios = Task(
    expected_output='ola',
    description='Com base nos eventos identificados, crie um itinerário detalhado que otimize a viagem do usuário. Inclua recomendações de transporte, acomodações e dicas locais. A resposta final deve ser um plano de viagem completo, com um cronograma diário.',
    agent=agente_planejamento_itinerarios
)

# Criando a equipe com processo sequencial
equipe = Crew(
    agents=[agente_pesquisa_eventos, agente_planejamento_itinerarios],
    tasks=[tarefa_pesquisa_eventos, tarefa_planejamento_itinerarios],
    verbose=True
)

# Iniciar o trabalho da equipe
resultado = equipe.kickoff()