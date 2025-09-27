# Como Usar o Sistema de Agents - Guia Prático

## 🎯 Resumo Rápido

Para fazer requests e receber respostas dos modelos treinados, você precisa seguir estes passos:

1. **Iniciar o servidor**: `python app.py`
2. **Criar um agent**: POST `/api/v1/agents`
3. **Aguardar treinamento**: GET `/api/v1/agents/{user_id}/status`
4. **Fazer perguntas**: POST `/api/v1/agents/{user_id}/inference`

## 🚀 Início Rápido

### 1. Configurar e Iniciar

```bash
# Navegar para o diretório
cd agents

# Instalar dependências (se necessário)
pip install -r requirements.txt

# Configurar variáveis de ambiente
export GOOGLE_CLOUD_PROJECT="seu-projeto-id"

# Iniciar o servidor
python app.py
```

O servidor estará disponível em `http://localhost:8080`

### 2. Testar com o Script de Exemplo

```bash
# Executar o script de teste
python test_agent_api.py
```

Este script criará automaticamente 3 agents especializados e fará perguntas para cada um.

## 📋 Exemplos de Uso

### Exemplo 1: Agent Especialista em Python

```bash
# 1. Criar agent
curl -X POST http://localhost:8080/api/v1/agents \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "python_expert",
    "training_data": "Sou especialista em Python, Flask, Django, data science e machine learning. Posso ajudar com dúvidas sobre programação, algoritmos e boas práticas.",
    "base_model": "distilbert-base-uncased"
  }'

# 2. Verificar status
curl http://localhost:8080/api/v1/agents/python_expert/status

# 3. Fazer pergunta
curl -X POST http://localhost:8080/api/v1/agents/python_expert/inference \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Como criar uma API REST com Flask?"
  }'
```

### Exemplo 2: Agent Especialista em Machine Learning

```bash
# 1. Criar agent
curl -X POST http://localhost:8080/api/v1/agents \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "ml_expert",
    "training_data": "Sou especialista em machine learning, deep learning, NLP, computer vision e MLOps. Posso ajudar com implementação de algoritmos e análise de dados.",
    "base_model": "distilbert-base-uncased"
  }'

# 2. Aguardar e fazer pergunta
curl -X POST http://localhost:8080/api/v1/agents/ml_expert/inference \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Como escolher entre Random Forest e XGBoost?"
  }'
```

## 🔧 Uso Programático

### Python

```python
import requests

# Criar agent
def create_agent(user_id, expertise):
    training_data = f"Sou especialista em {expertise}. Posso ajudar com dúvidas e explicações sobre este assunto."
    
    response = requests.post(
        "http://localhost:8080/api/v1/agents",
        json={
            "user_id": user_id,
            "training_data": training_data
        }
    )
    return response.json()

# Fazer pergunta
def ask_question(user_id, question):
    response = requests.post(
        f"http://localhost:8080/api/v1/agents/{user_id}/inference",
        json={"prompt": question}
    )
    return response.json()['response']

# Exemplo de uso
agent = create_agent("dev_expert", "desenvolvimento web e programação")
resposta = ask_question("dev_expert", "Como implementar autenticação JWT?")
print(resposta)
```

### JavaScript/Node.js

```javascript
// Criar agent
async function createAgent(userId, expertise) {
  const trainingData = `Sou especialista em ${expertise}. Posso ajudar com dúvidas e explicações sobre este assunto.`;
  
  const response = await fetch('http://localhost:8080/api/v1/agents', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_id: userId,
      training_data: trainingData
    })
  });
  
  return await response.json();
}

// Fazer pergunta
async function askQuestion(userId, question) {
  const response = await fetch(`http://localhost:8080/api/v1/agents/${userId}/inference`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt: question })
  });
  
  const result = await response.json();
  return result.response;
}

// Exemplo de uso
async function exemplo() {
  const agent = await createAgent('js_expert', 'JavaScript e Node.js');
  const resposta = await askQuestion('js_expert', 'Como implementar async/await?');
  console.log(resposta);
}
```

## 📊 Status dos Agents

Os possíveis status são:

- **`initializing`**: Agent sendo criado
- **`processing`**: Dados sendo processados e modelo sendo treinado  
- **`ready`**: Modelo pronto para inferência
- **`error`**: Erro durante o processo

## ⚠️ Pontos Importantes

1. **Aguardar o Treinamento**: O modelo precisa ser treinado antes de responder perguntas
2. **Dados de Treinamento**: Seja específico na descrição da expertise para melhores resultados
3. **Um Agent por Usuário**: Cada `user_id` pode ter apenas um agent ativo
4. **Timeout**: O treinamento pode levar alguns minutos

## 🔍 Endpoints Disponíveis

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/health` | Verificar se o serviço está funcionando |
| POST | `/api/v1/agents` | Criar um novo agent |
| GET | `/api/v1/agents/{user_id}/status` | Verificar status do agent |
| POST | `/api/v1/agents/{user_id}/inference` | **Fazer pergunta ao modelo** |
| GET | `/api/v1/agents` | Listar todos os agents |
| DELETE | `/api/v1/agents/{user_id}` | Deletar um agent |

## 🚨 Troubleshooting

### Problema: "Connection refused"
**Solução**: Verificar se o servidor está rodando
```bash
curl http://localhost:8080/health
```

### Problema: "Model not ready"
**Solução**: Aguardar o treinamento completar
```bash
curl http://localhost:8080/api/v1/agents/user123/status
```

### Problema: "User not found"
**Solução**: Criar o agent primeiro
```bash
curl -X POST http://localhost:8080/api/v1/agents \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user123", "training_data": "..."}'
```

## 🎮 Modo Interativo

Use o script de teste para modo interativo:

```bash
python test_agent_api.py
```

Comandos disponíveis:
- `status user_id`: Verificar status
- `ask user_id pergunta`: Fazer pergunta
- `list`: Listar agents
- `delete user_id`: Deletar agent

## 📚 Exemplos Completos

Veja os arquivos:
- `GUIA_REQUESTS.md`: Guia completo detalhado
- `test_agent_api.py`: Script de teste com exemplos
- `README.md`: Documentação geral do projeto

## 🎯 Próximos Passos

1. Experimente com diferentes tipos de expertise
2. Teste com perguntas específicas do seu domínio
3. Integre com sua aplicação
4. Monitore performance e ajuste conforme necessário

---

**Resumo**: Para receber respostas dos modelos, crie um agent com `POST /api/v1/agents`, aguarde o status `ready`, e então faça perguntas com `POST /api/v1/agents/{user_id}/inference`.
