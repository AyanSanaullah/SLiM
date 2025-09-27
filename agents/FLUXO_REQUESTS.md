# Fluxo de Requests para Receber Respostas dos Modelos

## 🔄 Diagrama do Fluxo

```
┌─────────────────┐
│   Cliente       │
│   (Você)        │
└─────────┬───────┘
          │
          │ 1. POST /api/v1/agents
          │    {"user_id": "user123", "training_data": "..."}
          ▼
┌─────────────────┐
│   Flask API     │
│   (app.py)      │
└─────────┬───────┘
          │
          │ 2. Cria agent pipeline
          ▼
┌─────────────────┐
│ UserAgentManager│
│ (user_agent_    │
│  manager.py)    │
└─────────┬───────┘
          │
          │ 3. Executa pipeline
          ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ DataProcessor   │───▶│ ModelTrainer    │───▶│ ModelEvaluator  │
│ Agent           │    │ Agent           │    │ Agent           │
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │
          │ 4. Status: ready
          ▼
┌─────────────────┐
│   Cliente       │
│   (Você)        │
└─────────┬───────┘
          │
          │ 5. POST /api/v1/agents/user123/inference
          │    {"prompt": "Sua pergunta aqui"}
          ▼
┌─────────────────┐
│   Flask API     │
│   (app.py)      │
└─────────┬───────┘
          │
          │ 6. make_inference()
          ▼
┌─────────────────┐
│ Gemini API      │
│ (Modelo base)   │
└─────────┬───────┘
          │
          │ 7. Resposta personalizada
          ▼
┌─────────────────┐
│   Cliente       │
│   (Você)        │
└─────────────────┘
```

## 📋 Passo a Passo Detalhado

### Fase 1: Criação do Agent
```
Cliente → Flask API → UserAgentManager → Pipeline de Agents
```

1. **Cliente faz POST** para `/api/v1/agents`
2. **Flask API** recebe a requisição
3. **UserAgentManager** cria o pipeline de agents
4. **Pipeline** é executado em thread separada

### Fase 2: Treinamento (Automático)
```
DataProcessor → ModelTrainer → ModelEvaluator → ModelDeployer
```

1. **DataProcessor**: Processa os dados de treinamento
2. **ModelTrainer**: Treina o modelo personalizado
3. **ModelEvaluator**: Avalia a performance
4. **ModelDeployer**: Prepara para deploy

### Fase 3: Inferência (Sua Pergunta)
```
Cliente → Flask API → Gemini API → Resposta Personalizada
```

1. **Cliente faz POST** para `/api/v1/agents/{user_id}/inference`
2. **Flask API** valida a requisição
3. **Gemini API** gera resposta usando contexto personalizado
4. **Cliente recebe** resposta personalizada

## 🔍 Estados do Sistema

### Estados do Agent
```
initializing → processing → ready
     ↓             ↓          ↓
   Criando    Treinando   Pronto para
   pipeline   modelo      responder
```

### Estados de Erro
```
initializing → error
processing → error
```

## 📊 Exemplo de Sequência de Requests

### Request 1: Criar Agent
```bash
POST /api/v1/agents
{
  "user_id": "python_expert",
  "training_data": "Sou especialista em Python..."
}

Response:
{
  "message": "Agent pipeline created...",
  "status": "created"
}
```

### Request 2: Verificar Status
```bash
GET /api/v1/agents/python_expert/status

Response:
{
  "status": "processing"  // ou "ready" ou "error"
}
```

### Request 3: Fazer Pergunta (quando ready)
```bash
POST /api/v1/agents/python_expert/inference
{
  "prompt": "Como criar uma API Flask?"
}

Response:
{
  "response": "Para criar uma API Flask...",
  "timestamp": "2024-01-15T10:40:00Z"
}
```

## ⏱️ Timeline Típica

```
T+0s:   Cliente cria agent
T+5s:   Status: initializing
T+10s:  Status: processing
T+30s:  Status: ready
T+35s:  Cliente faz primeira pergunta
T+36s:  Cliente recebe resposta
```

## 🔧 Componentes Envolvidos

### 1. Flask API (app.py)
- Recebe requests HTTP
- Valida dados
- Chama UserAgentManager
- Retorna respostas JSON

### 2. UserAgentManager
- Gerencia agents por usuário
- Cria pipelines de treinamento
- Controla estados
- Faz inferência

### 3. Pipeline de Agents
- **DataProcessorAgent**: Processa dados
- **ModelTrainerAgent**: Treina modelo
- **ModelEvaluatorAgent**: Avalia performance
- **ModelDeployerAgent**: Deploy

### 4. Gemini API
- Modelo base para inferência
- Usa contexto personalizado
- Gera respostas

## 🎯 Pontos Chave

1. **Um Request para Criar**: POST `/api/v1/agents`
2. **Múltiplos Requests para Status**: GET `/api/v1/agents/{user_id}/status`
3. **Múltiplos Requests para Perguntas**: POST `/api/v1/agents/{user_id}/inference`
4. **Aguardar Status "ready"** antes de fazer perguntas
5. **Contexto Personalizado** é usado em cada resposta

## 🚨 Fluxo de Erro

```
Request → Validação → Erro → Response com error
```

Exemplos de erros:
- Agent não encontrado
- Modelo não está pronto
- Dados inválidos
- Erro de conexão

## 📈 Monitoramento

### Endpoints de Monitoramento
- `GET /health`: Status do serviço
- `GET /api/v1/agents`: Lista todos os agents
- `GET /api/v1/config`: Configuração atual

### Logs Importantes
- Criação de agents
- Status de treinamento
- Requests de inferência
- Erros e exceções

---

**Resumo**: O fluxo é simples: criar agent → aguardar ready → fazer perguntas. Cada pergunta gera uma resposta personalizada baseada nos dados de treinamento do usuário.
