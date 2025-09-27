# ShellHacks ADK Agents

Sistema de agents personalizados usando Google Agents Development Kit (ADK) para hospedagem na Google Cloud Platform.

## Visão Geral

Este projeto implementa um sistema de agents de IA personalizados que permite:
- Criação de agents únicos para cada usuário
- Fine-tuning de modelos baseados em dados do usuário
- Deploy automático para Vertex AI
- API REST para gerenciamento e inferência
- Monitoramento e logging integrados

## Arquitetura

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Flask API      │    │   Google ADK    │
│   (Next.js)     │◄──►│   (app.py)       │◄──►│   Agents        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   Vertex AI      │
                       │   - Training     │
                       │   - Deployment   │
                       │   - Inference    │
                       └──────────────────┘
```

## Componentes

### 1. Agents ADK
- **UserAgentManager**: Gerencia agents personalizados por usuário
- **DataProcessorAgent**: Processa dados de treinamento
- **ModelTrainerAgent**: Treina modelos personalizados
- **ModelEvaluatorAgent**: Avalia performance dos modelos
- **ModelDeployerAgent**: Faz deploy para Vertex AI
- **VertexAIClient**: Cliente para integração com Vertex AI

### 2. API REST
- Criação de agents personalizados
- Status e monitoramento
- Inferência em tempo real
- Gerenciamento de modelos

### 3. Infraestrutura
- Docker para containerização
- Google Cloud Run para hospedagem
- Monitoramento com Cloud Monitoring
- Logging com Cloud Logging

## Configuração

### 1. Pré-requisitos
- Python 3.8+
- Docker
- Google Cloud SDK
- Conta Google Cloud com billing habilitado

### 2. Setup Local
```bash
# Clone o repositório
git clone <repository-url>
cd agents

# Execute o script de setup
./setup.sh

# Ative o ambiente virtual
source venv/bin/activate

# Configure as credenciais do Google Cloud
export GOOGLE_APPLICATION_CREDENTIALS="credentials/service-account.json"
export GOOGLE_CLOUD_PROJECT="your-project-id"
```

### 3. Configuração do Google Cloud
```bash
# Autentique com o Google Cloud
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Habilite as APIs necessárias
gcloud services enable aiplatform.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable storage.googleapis.com
```

## Uso

### 1. Desenvolvimento Local
```bash
# Inicie o servidor local
python app.py

# A API estará disponível em http://localhost:8080
```

### 2. Deploy para Google Cloud
```bash
# Deploy para Cloud Run
./deploy.sh cloud-run

# O serviço estará disponível na URL fornecida pelo Cloud Run
```

### 3. API Endpoints

#### Criar Agent Personalizado
```bash
curl -X POST http://localhost:8080/api/v1/agents \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "training_data": "Seus dados de treinamento aqui...",
    "base_model": "distilbert-base-uncased"
  }'
```

#### Verificar Status do Agent
```bash
curl http://localhost:8080/api/v1/agents/user123/status
```

#### Fazer Inferência
```bash
curl -X POST http://localhost:8080/api/v1/agents/user123/inference \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Sua pergunta aqui..."
  }'
```

#### Listar Todos os Agents
```bash
curl http://localhost:8080/api/v1/agents
```

## Estrutura de Arquivos

```
agents/
├── adk_agents/           # Agents do Google ADK
│   ├── user_agent_manager.py
│   ├── data_processor_agent.py
│   ├── model_trainer_agent.py
│   ├── model_evaluator_agent.py
│   ├── model_deployer_agent.py
│   └── vertex_client.py
├── config/               # Configurações
│   ├── adk_config.yaml
│   └── vertex_config.yaml
├── deployment/           # Scripts de deploy
│   ├── cloud_run.yaml
│   └── deploy.sh
├── docker/              # Configuração Docker
│   ├── Dockerfile
│   └── docker-compose.yml
├── monitoring/          # Monitoramento
│   ├── dashboard.json
│   └── alert-policies.yaml
├── tools/               # Ferramentas auxiliares
│   ├── custom_tools.py
│   └── model_tools.py
├── app.py              # Aplicação principal Flask
├── requirements.txt    # Dependências Python
├── setup.sh           # Script de setup
├── deploy.sh          # Script de deploy
└── README.md          # Este arquivo
```

## Monitoramento

### 1. Dashboard
- Acesse o Google Cloud Console
- Vá para Monitoring > Dashboards
- Importe o arquivo `monitoring/dashboard.json`

### 2. Alertas
- Configure as políticas de alerta em `monitoring/alert-policies.yaml`
- Receba notificações para:
  - Alta taxa de erro
  - Tempo de resposta alto
  - Serviço indisponível

### 3. Logs
```bash
# Visualizar logs do Cloud Run
gcloud logging read 'resource.type="cloud_run_revision"' --limit 50

# Filtrar logs por serviço
gcloud logging read 'resource.type="cloud_run_revision" AND resource.labels.service_name="shellhacks-adk-agents"' --limit 50
```

## Desenvolvimento

### 1. Adicionar Novo Agent
1. Crie um novo arquivo em `adk_agents/`
2. Implemente a classe do agent
3. Adicione ao pipeline em `user_agent_manager.py`
4. Atualize a documentação

### 2. Adicionar Novo Endpoint
1. Adicione a rota em `app.py`
2. Implemente a lógica de negócio
3. Adicione tratamento de erros
4. Atualize a documentação da API

### 3. Testes
```bash
# Testes unitários
python -m pytest tests/

# Testes de integração
python -m pytest tests/integration/

# Testes de carga
python -m pytest tests/load/
```

## Troubleshooting

### 1. Problemas de Autenticação
```bash
# Verificar credenciais
gcloud auth list

# Renovar credenciais
gcloud auth login

# Verificar projeto
gcloud config get-value project
```

### 2. Problemas de Deploy
```bash
# Verificar logs do build
gcloud builds list --limit 5

# Verificar status do serviço
gcloud run services describe shellhacks-adk-agents --region us-central1
```

### 3. Problemas de Performance
- Verifique os logs de monitoramento
- Ajuste os recursos no Cloud Run
- Otimize o código dos agents

## Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

## Suporte

Para suporte e dúvidas:
- Abra uma issue no GitHub
- Entre em contato com a equipe de desenvolvimento
- Consulte a documentação do Google ADK
