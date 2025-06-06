!pip install elasticsearch scikit-learn numpy pandas faker
from elasticsearch import Elasticsearch

# CUIDADO: Substitua estes valores pelos seus próprios
ES_HOST = "SEU_ENDPOINT_ELASTICSEARCH" # Ex: 'https://seunome.es.us-central1.gcp.cloud.es.io:9243'
ES_USERNAME = "elastic"
ES_PASSWORD = "SUA_SENHA_ELASTICSEARCH"

try:
    es = Elasticsearch(
        ES_HOST,
        http_auth=(ES_USERNAME, ES_PASSWORD)
    )
    if es.ping():
        print("Conectado ao Elasticsearch!")
    else:
        print("Não foi possível conectar ao Elasticsearch.")
except Exception as e:
    print(f"Erro ao conectar ao Elasticsearch: {e}")
    import datetime
import time
import random
from faker import Faker

fake = Faker('pt_BR')

def generate_financial_data(num_records=1):
    data = []
    for _ in range(num_records):
        timestamp = datetime.datetime.now() - datetime.timedelta(minutes=random.randint(0, 60)) # Dados recentes
        category = random.choice(['Alimentação', 'Transporte', 'Moradia', 'Lazer', 'Educação', 'Saúde', 'Outros'])
        amount = round(random.uniform(10.0, 500.0), 2)
        description = fake.sentence(nb_words=6)
        
        # Simular uma anomalia de vez em quando
        if random.random() < 0.05: # 5% de chance de ser uma anomalia (valor muito alto)
            amount = round(random.uniform(1000.0, 5000.0), 2)
            description = "GASTO ANORMAL: " + description

        data.append({
            "timestamp": timestamp.isoformat(),
            "category": category,
            "amount": amount,
            "description": description
        })
    return data

# Testar a geração de dados
# print(generate_financial_data(3))
def ingest_data_to_elasticsearch(data, index_name="gastos_financeiros"):
    for record in data:
        try:
            res = es.index(index=index_name, document=record)
            # print(f"Documento indexado: {res['_id']}")
        except Exception as e:
            print(f"Erro ao indexar documento: {e}")

# Exemplo de ingestão
# sample_data = generate_financial_data(5)
# ingest_data_to_elasticsearch(sample_data)
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np

# Função para buscar dados do Elasticsearch para treinamento
def fetch_data_for_training(index_name="gastos_financeiros", size=1000):
    try:
        res = es.search(index=index_name, body={"size": size, "sort": [{"timestamp": {"order": "desc"}}]},
                        track_total_hits=True)
        hits = res['hits']['hits']
        
        if not hits:
            print("Nenhum dado encontrado para treinamento.")
            return pd.DataFrame()

        records = [hit['_source'] for hit in hits]
        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        print(f"Erro ao buscar dados para treinamento: {e}")
        return pd.DataFrame()

# Treinar o modelo de detecção de anomalias
model = None

def train_anomaly_detector():
    global model
    df_train = fetch_data_for_training()
    
    if df_train.empty or len(df_train) < 10: # Mínimo de dados para treinar
        print("Dados insuficientes para treinar o modelo.")
        model = None
        return

    # Usaremos apenas a coluna 'amount' para a detecção de anomalias para simplificar
    X = df_train[['amount']]
    
    # IsolationForest é bom para dados numéricos em alta dimensão, mas funciona bem aqui.
    # contamination: a proporção esperada de anomalias nos dados. Ajuste conforme necessário.
    model = IsolationForest(contamination=0.05, random_state=42) 
    model.fit(X)
    print("Modelo de detecção de anomalias treinado com sucesso!")

# Treinar o modelo inicialmente
train_anomaly_detector()

# Função para detectar anomalias em novos dados
def detect_anomalies(new_data):
    if model is None:
        print("Modelo não treinado. Treine o modelo primeiro.")
        return []

    # Transformar os novos dados em um DataFrame
    df_new = pd.DataFrame(new_data)
    df_new['timestamp'] = pd.to_datetime(df_new['timestamp'])

    if df_new.empty:
        return []

    # Prever anomalias
    # O método predict retorna -1 para anomalias e 1 para observações normais
    predictions = model.predict(df_new[['amount']])
    
    anomalies = []
    for i, pred in enumerate(predictions):
        if pred == -1:
            anomalies.append(new_data[i])
    return anomalies
    def send_alert(anomaly_data):
    print("\n!!! ALERTA DE ANOMALIA DETECTADA !!!")
    print(f"Gasto fora do padrão: R$ {anomaly_data['amount']:.2f}")
    print(f"Categoria: {anomaly_data['category']}")
    print(f"Descrição: {anomaly_data['description']}")
    print(f"Timestamp: {anomaly_data['timestamp']}")
    print("--------------------------------------")
    import time

def start_monitoring_agent(interval_seconds=10, records_per_ingestion=2):
    print(f"\nIniciando Agente de Monitoramento e Alerta... (Intervalo: {interval_seconds}s)")
    while True:
        try:
            # 1. Gerar novos dados
            new_financial_data = generate_financial_data(records_per_ingestion)
            
            # 2. Ingerir dados no Elasticsearch
            ingest_data_to_elasticsearch(new_financial_data)
            
            # 3. Detectar anomalias
            anomalies = detect_anomalies(new_financial_data)
            
            # 4. Enviar alertas para anomalias detectadas
            if anomalies:
                for anomaly in anomalies:
                    send_alert(anomaly)
            else:
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Nenhuma anomalia detectada nos últimos dados.")
            
            # Re-treinar o modelo periodicamente (por exemplo, a cada 10 ciclos)
            if random.randint(1, 10) == 1:
                train_anomaly_detector()

            time.sleep(interval_seconds)

        except KeyboardInterrupt:
            print("\nAgente de monitoramento interrompido.")
            break
        except Exception as e:
            print(f"Ocorreu um erro no loop principal: {e}")
            time.sleep(interval_seconds) # Esperar um pouco antes de tentar novamente

# Para iniciar o agente, execute a linha abaixo:
# start_monitoring_agent(interval_seconds=5, records_per_ingestion=3)
start_monitoring_agent(interval_seconds=5, records_per_ingestion=3)
