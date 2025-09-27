# 📋 Guia Completo de Comandos - ShellHacks Agents

Este documento contém todos os comandos disponíveis para usar o sistema de agents com treinamento real de modelos.

## 🚀 Comandos de Inicialização

### 1. Iniciar o Servidor Principal
```bash
cd agents
python3 app.py
```
**O que faz:** Inicia o servidor Flask na porta 8080 com todos os endpoints da API

### 2. Iniciar o Serviço de String Comparison
```bash
cd string-comparison
python3 backend.py
```
**O que faz:** Inicia o serviço de comparação semântica na porta 8000

## 📡 Comandos de API (cURL)

### Comandos Básicos

#### Health Check
```bash
curl http://localhost:8080/health
```
**Resposta esperada:**
```json
{
  "status": "healthy",
  "service": "shellhacks-adk-agents",
  "version": "2.0.0",
  "training_type": "REAL_MODELS"
}
```

#### Página Inicial (Documentação da API)
```bash
curl http://localhost:8080/
```

#### Listar Todos os Agentes
```bash
curl http://localhost:8080/api/v1/agents
```

### Comandos de Criação de Agentes

#### Criar Agente Básico
```bash
curl -X POST http://localhost:8080/api/v1/agents \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "python_expert",
    "training_data": "Sou especialista em Python, Flask, Django e desenvolvimento web. Posso ajudar com dúvidas sobre programação e boas práticas.",
    "base_model": "distilbert-base-uncased"
  }'
```

#### Criar Agente Avançado (JSON Dataset + QLoRA + String Comparison)
```bash
curl -X POST http://localhost:8080/api/v1/agents/advanced \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "advanced_expert",
    "json_dataset": [
      {"prompt": "Como criar uma API REST?", "answer": "Use Flask com @app.route()"},
      {"prompt": "O que é machine learning?", "answer": "ML é IA que aprende de dados"},
      {"prompt": "Como usar Docker?", "answer": "Docker cria containers portáteis"}
    ],
    "base_model": "distilbert-base-uncased"
  }'
```

### Comandos de Monitoramento

#### Verificar Status do Agente
```bash
curl http://localhost:8080/api/v1/agents/python_expert/status
```

#### Verificar Pipeline Detalhado
```bash
curl http://localhost:8080/api/v1/agents/python_expert/pipeline
```

### Comandos de Inferência

#### Fazer Pergunta ao Modelo Treinado
```bash
curl -X POST http://localhost:8080/api/v1/agents/python_expert/inference \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Como criar uma API REST com Flask?"
  }'
```

#### Avaliar Modelo com String Comparison
```bash
curl -X POST http://localhost:8080/api/v1/agents/python_expert/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "test_prompt": "O que é Flask?",
    "expected_answer": "Flask é um framework web para Python"
  }'
```

### Comandos de Gerenciamento

#### Deletar Agente
```bash
curl -X DELETE http://localhost:8080/api/v1/agents/python_expert
```

#### Obter Configuração Atual
```bash
curl http://localhost:8080/api/v1/config
```

## 🐍 Scripts Python Disponíveis

### Scripts de Teste Automático

#### Teste do Sistema Básico
```bash
python3 test_agent_api.py
```
**O que faz:**
- Cria 3 agentes especializados
- Testa inferência com cada um
- Modo interativo disponível

#### Teste do Sistema Avançado
```bash
python3 test_advanced_system.py
```
**O que faz:**
- Cria agente avançado com JSON dataset
- Testa QLoRA + CUDA + String Comparison
- Mostra métricas detalhadas

#### Listar Agentes Existentes
```bash
python3 listar_agentes.py
```
**O que faz:**
- Lista todos os agentes ativos
- Mostra status e informações

### Scripts Utilitários

#### Script de Setup Inicial
```bash
./setup.sh
```
**O que faz:**
- Instala dependências
- Configura ambiente virtual
- Prepara diretórios

#### Script de Deploy
```bash
./deploy.sh cloud-run
```
**O que faz:**
- Deploy para Google Cloud Run
- Configura Vertex AI
- Setup de monitoramento

## 🔧 Comandos de Desenvolvimento

### Instalação de Dependências
```bash
pip3 install -r requirements.txt
```

### Dependências Específicas
```bash
# Machine Learning
pip3 install torch transformers peft accelerate

# String Comparison Service
pip3 install fastapi sentence-transformers spacy nltk

# Google Cloud
pip3 install google-cloud-aiplatform google-adk
```

### Comandos de Debug

#### Verificar Logs em Tempo Real
```bash
tail -f logs/user_logs/*/training.log
```

#### Verificar Modelos Treinados
```bash
ls -la models/user_models/
```

#### Verificar Dados Processados
```bash
ls -la data/user_data/
```

## 🎯 Exemplos de Uso Completo

### Exemplo 1: Agente Especialista em Python

```bash
# 1. Criar agente
curl -X POST http://localhost:8080/api/v1/agents \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "python_guru",
    "training_data": "Sou especialista em Python com 10 anos de experiência. Domino Flask, Django, FastAPI, pandas, numpy, scikit-learn. Posso ajudar com desenvolvimento web, análise de dados, machine learning e automação.",
    "base_model": "distilbert-base-uncased"
  }'

# 2. Aguardar treinamento (verificar status)
curl http://localhost:8080/api/v1/agents/python_guru/status

# 3. Fazer perguntas
curl -X POST http://localhost:8080/api/v1/agents/python_guru/inference \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Como otimizar performance de uma aplicação Flask?"}'

# 4. Avaliar resposta
curl -X POST http://localhost:8080/api/v1/agents/python_guru/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "test_prompt": "Como otimizar Flask?",
    "expected_answer": "Use cache, otimize queries, configure gunicorn"
  }'
```

### Exemplo 2: Agente Avançado com Dataset JSON

```bash
# 1. Criar agente avançado
curl -X POST http://localhost:8080/api/v1/agents/advanced \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "devops_expert",
    "json_dataset": [
      {
        "prompt": "Como configurar CI/CD?",
        "answer": "Configure GitHub Actions ou GitLab CI com stages de build, test e deploy. Use containers Docker e ambientes separados."
      },
      {
        "prompt": "O que é Kubernetes?",
        "answer": "Kubernetes é um orquestrador de containers que automatiza deploy, scaling e gerenciamento de aplicações containerizadas."
      },
      {
        "prompt": "Como monitorar aplicações?",
        "answer": "Use Prometheus para métricas, Grafana para dashboards, ELK Stack para logs e alertas via PagerDuty ou Slack."
      }
    ]
  }'

# 2. Acompanhar treinamento com logs detalhados
curl http://localhost:8080/api/v1/agents/devops_expert/status

# 3. Testar inferência
curl -X POST http://localhost:8080/api/v1/agents/devops_expert/inference \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Como fazer deploy seguro?"}'
```

## 🔍 Comandos de Monitoramento e Debug

### Verificar Status dos Serviços
```bash
# Verificar se agents está rodando
curl http://localhost:8080/health

# Verificar se string-comparison está rodando
curl http://localhost:8000/health

# Listar processos Python ativos
ps aux | grep python
```

### Logs e Debugging
```bash
# Ver logs do servidor
tail -f logs/app.log

# Ver logs de treinamento específico
tail -f logs/user_logs/python_expert/training.log

# Ver performance logs
cat logs/user_logs/python_expert/latest_performance.json
```

### Limpeza e Reset
```bash
# Parar todos os serviços
killall Python

# Limpar dados de treino
rm -rf data/user_data/*
rm -rf models/user_models/*
rm -rf logs/user_logs/*

# Reiniciar servidor
python3 app.py
```

## 🌐 Comandos de Deploy e Produção

### Deploy Local com Docker
```bash
# Build da imagem
docker build -t shellhacks-agents .

# Executar container
docker run -p 8080:8080 shellhacks-agents
```

### Deploy para Google Cloud
```bash
# Configurar projeto
export GOOGLE_CLOUD_PROJECT="arctic-keyword-473423-g6"
gcloud config set project $GOOGLE_CLOUD_PROJECT

# Deploy para Cloud Run
./deploy.sh cloud-run

# Verificar deploy
gcloud run services list
```

### Comandos de Vertex AI
```bash
# Listar jobs de treinamento
gcloud ai custom-jobs list

# Listar endpoints
gcloud ai endpoints list

# Verificar modelos
gcloud ai models list
```

## 📊 Comandos de Métricas e Análise

### Análise de Performance
```bash
# Ver estatísticas de todos os agentes
python3 -c "
import requests
r = requests.get('http://localhost:8080/api/v1/agents')
for user_id, info in r.json()['users'].items():
    print(f'{user_id}: {info.get(\"status\", \"unknown\")}')
"
```

### Comparação de String Comparison
```bash
# Teste direto do serviço
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{
    "sentence1": "Como criar uma API REST?",
    "sentence2": "Use Flask com decoradores @app.route"
  }'
```

## 🆘 Comandos de Troubleshooting

### Problemas Comuns

#### Servidor não inicia
```bash
# Verificar porta ocupada
lsof -i :8080

# Matar processo na porta
kill -9 $(lsof -ti :8080)

# Reinstalar dependências
pip3 install -r requirements.txt --force-reinstall
```

#### Erro de permissões Google Cloud
```bash
# Verificar autenticação
gcloud auth list

# Renovar credenciais
gcloud auth login

# Verificar service account
gcloud iam service-accounts list
```

#### String comparison não funciona
```bash
# Verificar se está rodando
curl http://localhost:8000/health

# Iniciar se necessário
cd string-comparison
python3 backend.py
```

## 📚 Comandos de Documentação

### Gerar Documentação da API
```bash
# Ver documentação interativa
curl http://localhost:8080/ | jq .

# Exportar schema da API
curl http://localhost:8080/openapi.json > api_schema.json
```

### Ver Logs de Performance
```bash
# Logs formatados
python3 -c "
import json
with open('logs/user_logs/python_expert/latest_performance.json') as f:
    print(json.dumps(json.load(f), indent=2))
"
```

---

## 🎉 Resumo dos Comandos Principais

| Ação | Comando |
|------|---------|
| **Iniciar servidor** | `python3 app.py` |
| **Criar agente básico** | `curl -X POST localhost:8080/api/v1/agents -d '{...}'` |
| **Criar agente avançado** | `curl -X POST localhost:8080/api/v1/agents/advanced -d '{...}'` |
| **Verificar status** | `curl localhost:8080/api/v1/agents/USER_ID/status` |
| **Fazer pergunta** | `curl -X POST localhost:8080/api/v1/agents/USER_ID/inference -d '{...}'` |
| **Avaliar modelo** | `curl -X POST localhost:8080/api/v1/agents/USER_ID/evaluate -d '{...}'` |
| **Listar agentes** | `curl localhost:8080/api/v1/agents` |
| **Teste automático** | `python3 test_advanced_system.py` |
| **String comparison** | `curl -X POST localhost:8000/compare -d '{...}'` |

Este sistema oferece treinamento real de modelos com QLoRA, CUDA, avaliação por string comparison e logging detalhado de performance! 🚀
