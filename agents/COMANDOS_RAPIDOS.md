# ⚡ Comandos Rápidos - ShellHacks Agents

## 🚀 Início Rápido

```bash
# 1. Iniciar serviços
cd agents && python3 app.py &
cd string-comparison && python3 backend.py &

# 2. Teste rápido
python3 test_advanced_system.py
```

## 📋 Comandos Essenciais

### Criar Agente Básico
```bash
curl -X POST localhost:8080/api/v1/agents \
  -H "Content-Type: application/json" \
  -d '{"user_id": "expert", "training_data": "Sou especialista em..."}'
```

### Criar Agente Avançado (JSON + String Comparison)
```bash
curl -X POST localhost:8080/api/v1/agents/advanced \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "advanced_expert",
    "json_dataset": [
      {"prompt": "Como fazer X?", "answer": "Para fazer X, você..."},
      {"prompt": "O que é Y?", "answer": "Y é um conceito que..."}
    ]
  }'
```

### Verificar Status
```bash
curl localhost:8080/api/v1/agents/expert/status
```

### Fazer Pergunta
```bash
curl -X POST localhost:8080/api/v1/agents/expert/inference \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Como resolver este problema?"}'
```

### Listar Agentes
```bash
curl localhost:8080/api/v1/agents
```

## 🔍 Health Checks

```bash
# Serviço principal
curl localhost:8080/health

# String comparison
curl localhost:8000/health

# Listar agentes
curl localhost:8080/api/v1/agents | jq '.users | keys'
```

## 🧪 Testes Automáticos

```bash
# Teste completo do sistema
python3 test_advanced_system.py

# Teste básico
python3 test_agent_api.py

# Listar agentes existentes
python3 listar_agentes.py
```

## 🛠️ Debug e Limpeza

```bash
# Parar serviços
killall Python

# Ver processos
ps aux | grep python

# Limpar dados
rm -rf data/user_data/* models/user_models/* logs/user_logs/*
```

## 📊 Exemplos de Dataset JSON

### Python Expert
```json
{
  "user_id": "python_expert",
  "json_dataset": [
    {"prompt": "Como criar API REST?", "answer": "Use Flask com @app.route()"},
    {"prompt": "O que é Django?", "answer": "Django é um framework web Python"},
    {"prompt": "Como usar pandas?", "answer": "Pandas é para análise de dados"}
  ]
}
```

### DevOps Expert
```json
{
  "user_id": "devops_expert", 
  "json_dataset": [
    {"prompt": "Como configurar CI/CD?", "answer": "Use GitHub Actions ou GitLab CI"},
    {"prompt": "O que é Docker?", "answer": "Docker cria containers portáteis"},
    {"prompt": "Como usar Kubernetes?", "answer": "K8s orquestra containers"}
  ]
}
```

## 🎯 Comandos por Funcionalidade

### Treinamento
| Ação | Comando |
|------|---------|
| Criar agente básico | `POST /api/v1/agents` |
| Criar agente avançado | `POST /api/v1/agents/advanced` |
| Ver status | `GET /api/v1/agents/{id}/status` |

### Inferência
| Ação | Comando |
|------|---------|
| Fazer pergunta | `POST /api/v1/agents/{id}/inference` |
| Avaliar resposta | `POST /api/v1/agents/{id}/evaluate` |
| Ver pipeline | `GET /api/v1/agents/{id}/pipeline` |

### Gerenciamento
| Ação | Comando |
|------|---------|
| Listar todos | `GET /api/v1/agents` |
| Deletar agente | `DELETE /api/v1/agents/{id}` |
| Ver config | `GET /api/v1/config` |

## 🔥 Performance Logging

O sistema automaticamente loga no terminal:
- ✅ Percentual de cada teste de string comparison
- 📊 Métricas detalhadas de similaridade
- 🎯 Score geral de performance
- 🟢🟡🔴 Classificação por qualidade

Exemplo de saída:
```
🎯🎯🎯🎯 AGENT PERFORMANCE REPORT - ADVANCED_EXPERT 🎯🎯🎯🎯
📅 Completed at: 2024-01-15 14:30:22
🤖 Agent Type: advanced_qlora

🔍 STRING COMPARISON EVALUATION
   Total Tests: 8
   Successful: 8 (100.0%)
   Failed: 0 (0.0%)

📈 SIMILARITY METRICS
   Average String Similarity: 85.32%
   Average Model Confidence: 91.45%
   Best Similarity: 95.67%
   Worst Similarity: 72.18%

🎯 PERFORMANCE DISTRIBUTION
   🟢 High Quality (>80%): 6 (75.0%)
   🟡 Medium Quality (50-80%): 2 (25.0%)
   🔴 Low Quality (<50%): 0 (0.0%)

🏆 OVERALL PERFORMANCE SCORE: 87.2%
🏆 EXCELLENT PERFORMANCE!
```
