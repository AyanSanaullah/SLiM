# Guia Completo: Como Fazer Requests para os Modelos Treinados

Este guia explica como fazer requests para receber respostas dos modelos personalizados que estão sendo treinados no sistema de agents.

## 🚀 Visão Geral do Sistema

O sistema funciona da seguinte forma:

1. **Criação do Agent**: Você cria um agent personalizado para um usuário
2. **Treinamento**: O sistema processa os dados e treina o modelo
3. **Deploy**: O modelo é deployado no Vertex AI
4. **Inferência**: Você pode fazer requests para obter respostas do modelo treinado

## 📋 Endpoints Disponíveis

### 1. Health Check
```bash
GET /health
```
Verifica se o serviço está funcionando.

### 2. Criar Agent Personalizado
```bash
POST /api/v1/agents
```
Cria um novo agent personalizado para um usuário.

### 3. Verificar Status do Agent
```bash
GET /api/v1/agents/{user_id}/status
```
Verifica o status do treinamento do modelo.

### 4. **Fazer Inferência (Obter Resposta do Modelo)**
```bash
POST /api/v1/agents/{user_id}/inference
```
Este é o endpoint principal para receber respostas dos modelos treinados.

### 5. Listar Todos os Agents
```bash
GET /api/v1/agents
```
Lista todos os agents ativos.

## 🔧 Configuração Inicial

### 1. Iniciar o Servidor
```bash
cd agents
python app.py
```

O servidor estará disponível em `http://localhost:8080`

### 2. Configurar Variáveis de Ambiente
```bash
export GOOGLE_CLOUD_PROJECT="seu-projeto-id"
export GEMINI_API_KEY="sua-chave-gemini"
export GOOGLE_APPLICATION_CREDENTIALS="credentials/service-account.json"
```

## 📝 Exemplos Práticos

### Exemplo 1: Fluxo Completo

#### Passo 1: Criar um Agent Personalizado
```bash
curl -X POST http://localhost:8080/api/v1/agents \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "usuario123",
    "training_data": "Sou especialista em programação Python e desenvolvimento web. Tenho experiência com Flask, Django, FastAPI e outras tecnologias. Posso ajudar com dúvidas sobre algoritmos, estruturas de dados e boas práticas de programação.",
    "base_model": "distilbert-base-uncased"
  }'
```

**Resposta esperada:**
```json
{
  "message": "Agent pipeline created and started for user usuario123",
  "user_id": "usuario123",
  "status": "created",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

#### Passo 2: Verificar Status do Treinamento
```bash
curl http://localhost:8080/api/v1/agents/usuario123/status
```

**Resposta esperada:**
```json
{
  "user_id": "usuario123",
  "status": "ready",
  "created_at": "2024-01-15T10:30:00.000Z",
  "pipeline_id": "abc123-def456-ghi789",
  "base_model": "distilbert-base-uncased",
  "training_data_size": 245,
  "model_ready_at": "2024-01-15T10:35:00.000Z",
  "endpoint_url": "https://us-central1-aiplatform.googleapis.com/v1/projects/seu-projeto/locations/us-central1/endpoints/endpoint-usuario123-xyz789"
}
```

#### Passo 3: Fazer Inferência (Obter Resposta do Modelo)
```bash
curl -X POST http://localhost:8080/api/v1/agents/usuario123/inference \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Como criar uma API REST com Flask?"
  }'
```

**Resposta esperada:**
```json
{
  "user_id": "usuario123",
  "prompt": "Como criar uma API REST com Flask?",
  "response": "Para criar uma API REST com Flask, você pode seguir estes passos:\n\n1. Instalar o Flask:\n```bash\npip install flask\n```\n\n2. Criar um arquivo app.py:\n```python\nfrom flask import Flask, jsonify, request\n\napp = Flask(__name__)\n\n@app.route('/api/hello', methods=['GET'])\ndef hello():\n    return jsonify({'message': 'Hello World!'})\n\n@app.route('/api/users', methods=['POST'])\ndef create_user():\n    data = request.get_json()\n    return jsonify({'user': data}), 201\n\nif __name__ == '__main__':\n    app.run(debug=True)\n```\n\n3. Executar a aplicação:\n```bash\npython app.py\n```\n\nA API estará disponível em http://localhost:5000",
  "timestamp": "2024-01-15T10:40:00.000Z"
}
```

### Exemplo 2: Usando JavaScript/Fetch

```javascript
// Criar agent
async function createAgent(userId, trainingData) {
  const response = await fetch('http://localhost:8080/api/v1/agents', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      user_id: userId,
      training_data: trainingData,
      base_model: 'distilbert-base-uncased'
    })
  });
  
  return await response.json();
}

// Fazer inferência
async function getModelResponse(userId, prompt) {
  const response = await fetch(`http://localhost:8080/api/v1/agents/${userId}/inference`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      prompt: prompt
    })
  });
  
  return await response.json();
}

// Exemplo de uso
async function exemplo() {
  // Criar agent
  const agent = await createAgent('usuario456', 'Sou especialista em machine learning e deep learning...');
  console.log('Agent criado:', agent);
  
  // Aguardar um pouco para o treinamento
  setTimeout(async () => {
    // Fazer pergunta
    const resposta = await getModelResponse('usuario456', 'Explique o que é deep learning');
    console.log('Resposta do modelo:', resposta.response);
  }, 10000); // 10 segundos
}
```

### Exemplo 3: Usando Python

```python
import requests
import time

class AgentClient:
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url
    
    def create_agent(self, user_id, training_data, base_model="distilbert-base-uncased"):
        """Criar um agent personalizado"""
        url = f"{self.base_url}/api/v1/agents"
        data = {
            "user_id": user_id,
            "training_data": training_data,
            "base_model": base_model
        }
        response = requests.post(url, json=data)
        return response.json()
    
    def get_status(self, user_id):
        """Verificar status do agent"""
        url = f"{self.base_url}/api/v1/agents/{user_id}/status"
        response = requests.get(url)
        return response.json()
    
    def make_inference(self, user_id, prompt):
        """Fazer inferência com o modelo treinado"""
        url = f"{self.base_url}/api/v1/agents/{user_id}/inference"
        data = {"prompt": prompt}
        response = requests.post(url, json=data)
        return response.json()
    
    def wait_for_ready(self, user_id, max_wait=300):
        """Aguardar até o modelo estar pronto"""
        start_time = time.time()
        while time.time() - start_time < max_wait:
            status = self.get_status(user_id)
            if status.get('status') == 'ready':
                return True
            elif status.get('status') == 'error':
                raise Exception(f"Erro no treinamento: {status.get('error')}")
            time.sleep(5)
        return False

# Exemplo de uso
client = AgentClient()

# Criar agent
agent = client.create_agent(
    user_id="python_user",
    training_data="Sou especialista em Python, data science e machine learning..."
)
print("Agent criado:", agent)

# Aguardar modelo ficar pronto
if client.wait_for_ready("python_user"):
    # Fazer pergunta
    resposta = client.make_inference(
        "python_user", 
        "Como implementar uma rede neural em Python?"
    )
    print("Resposta:", resposta['response'])
else:
    print("Timeout aguardando modelo ficar pronto")
```

## 🔍 Status dos Agents

Os possíveis status são:

- **`initializing`**: Agent sendo criado
- **`processing`**: Dados sendo processados e modelo sendo treinado
- **`ready`**: Modelo pronto para inferência
- **`error`**: Erro durante o processo

## ⚠️ Pontos Importantes

### 1. Aguardar o Treinamento
O modelo precisa ser treinado antes de poder responder perguntas. Verifique o status antes de fazer inferência.

### 2. Dados de Treinamento
Os dados de treinamento são usados para personalizar o modelo. Quanto mais específicos e relevantes, melhor será a resposta.

### 3. Limitações
- Cada usuário pode ter apenas um agent ativo
- O treinamento pode levar alguns minutos
- O modelo usa o Gemini como base para inferência

### 4. Monitoramento
Use o endpoint `/api/v1/agents` para listar todos os agents e seus status.

## 🚨 Troubleshooting

### Erro: "Model not ready"
```bash
# Verificar status
curl http://localhost:8080/api/v1/agents/usuario123/status

# Aguardar e tentar novamente
```

### Erro: "User not found"
```bash
# Verificar se o agent foi criado
curl http://localhost:8080/api/v1/agents

# Criar o agent se necessário
```

### Erro de Conexão
```bash
# Verificar se o servidor está rodando
curl http://localhost:8080/health
```

## 📊 Exemplo de Integração Completa

```python
import requests
import json
from typing import Dict, Any

class ShellHacksAgentAPI:
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
    
    def create_personalized_agent(self, user_id: str, expertise: str) -> Dict[str, Any]:
        """Criar agent personalizado baseado na expertise do usuário"""
        training_data = f"""
        Sou um especialista em {expertise}. 
        Tenho ampla experiência e conhecimento profundo nesta área.
        Posso ajudar com dúvidas, explicações e orientações sobre {expertise}.
        """
        
        response = requests.post(
            f"{self.base_url}/api/v1/agents",
            json={
                "user_id": user_id,
                "training_data": training_data,
                "base_model": "distilbert-base-uncased"
            }
        )
        return response.json()
    
    def ask_question(self, user_id: str, question: str) -> str:
        """Fazer pergunta ao modelo personalizado"""
        response = requests.post(
            f"{self.base_url}/api/v1/agents/{user_id}/inference",
            json={"prompt": question}
        )
        
        if response.status_code == 200:
            return response.json()['response']
        else:
            return f"Erro: {response.json().get('error', 'Erro desconhecido')}"
    
    def get_agent_info(self, user_id: str) -> Dict[str, Any]:
        """Obter informações do agent"""
        response = requests.get(f"{self.base_url}/api/v1/agents/{user_id}/status")
        return response.json()

# Exemplo de uso
api = ShellHacksAgentAPI()

# Criar agent para especialista em Python
agent = api.create_personalized_agent("dev_python", "programação Python e desenvolvimento web")
print("Agent criado:", agent)

# Aguardar um pouco para o treinamento
import time
time.sleep(30)

# Fazer perguntas
perguntas = [
    "Como implementar autenticação JWT em Flask?",
    "Qual a diferença entre list comprehension e generator expression?",
    "Como otimizar performance de queries Django?"
]

for pergunta in perguntas:
    resposta = api.ask_question("dev_python", pergunta)
    print(f"Pergunta: {pergunta}")
    print(f"Resposta: {resposta}\n")
```

## 🎯 Próximos Passos

1. **Implementar Cache**: Para respostas mais rápidas
2. **Adicionar Autenticação**: Para segurança
3. **Implementar Rate Limiting**: Para controle de uso
4. **Adicionar Métricas**: Para monitoramento
5. **Implementar Webhooks**: Para notificações quando o modelo estiver pronto

---

Este guia fornece tudo que você precisa para fazer requests e receber respostas dos modelos treinados no sistema de agents!
