# Quick Start - ShellHacks ADK Agents

Guia rápido para configurar e hospedar os agents usando Google ADK.

## 1. Configuração Inicial

### Pré-requisitos
- Conta Google Cloud com billing habilitado
- Google Cloud SDK instalado
- Docker instalado
- Python 3.8+

### Configurar Google Cloud
```bash
# 1. Autenticar
gcloud auth login

# 2. Definir projeto
export GOOGLE_CLOUD_PROJECT="seu-projeto-id"
gcloud config set project $GOOGLE_CLOUD_PROJECT

# 3. Habilitar APIs
gcloud services enable aiplatform.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

### Configurar Credenciais
```bash
# 1. Criar service account
gcloud iam service-accounts create shellhacks-adk-agent \
    --display-name="ShellHacks ADK Agent"

# 2. Conceder permissões
gcloud projects add-iam-policy-binding $GOOGLE_CLOUD_PROJECT \
    --member="serviceAccount:shellhacks-adk-agent@$GOOGLE_CLOUD_PROJECT.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding $GOOGLE_CLOUD_PROJECT \
    --member="serviceAccount:shellhacks-adk-agent@$GOOGLE_CLOUD_PROJECT.iam.gserviceaccount.com" \
    --role="roles/storage.admin"

# 3. Baixar chave
gcloud iam service-accounts keys create credentials/service-account.json \
    --iam-account=shellhacks-adk-agent@$GOOGLE_CLOUD_PROJECT.iam.gserviceaccount.com
```

## 2. Setup Local

```bash
# 1. Navegar para o diretório
cd agents

# 2. Executar setup
./setup.sh

# 3. Ativar ambiente virtual
source venv/bin/activate

# 4. Configurar variáveis de ambiente
cp env.example .env
# Editar .env com suas configurações
```

## 3. Teste Local

```bash
# 1. Iniciar servidor local
python app.py

# 2. Testar health check
curl http://localhost:8080/health

# 3. Criar um agent de teste
curl -X POST http://localhost:8080/api/v1/agents \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test-user",
    "training_data": "Este é um exemplo de dados de treinamento para testar o sistema.",
    "base_model": "distilbert-base-uncased"
  }'

# 4. Verificar status
curl http://localhost:8080/api/v1/agents/test-user/status
```

## 4. Deploy para Google Cloud

```bash
# 1. Deploy para Cloud Run
./deploy.sh cloud-run

# 2. Verificar deploy
gcloud run services list

# 3. Testar endpoint
curl https://seu-servico-url/health
```

## 5. Monitoramento

### Dashboard
```bash
# Importar dashboard
gcloud monitoring dashboards create --config-from-file=monitoring/dashboard.json
```

### Logs
```bash
# Ver logs em tempo real
gcloud logging tail 'resource.type="cloud_run_revision"'

# Ver logs específicos
gcloud logging read 'resource.type="cloud_run_revision" AND resource.labels.service_name="shellhacks-adk-agents"' --limit 50
```

## 6. Uso da API

### Criar Agent Personalizado
```bash
curl -X POST https://seu-servico-url/api/v1/agents \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "usuario123",
    "training_data": "Dados específicos do usuário para personalização...",
    "base_model": "distilbert-base-uncased"
  }'
```

### Fazer Inferência
```bash
curl -X POST https://seu-servico-url/api/v1/agents/usuario123/inference \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Qual é a resposta para minha pergunta?"
  }'
```

### Listar Agents
```bash
curl https://seu-servico-url/api/v1/agents
```

## 7. Troubleshooting

### Problemas Comuns

#### Erro de Autenticação
```bash
# Verificar credenciais
gcloud auth list

# Renovar autenticação
gcloud auth login
```

#### Erro de Permissões
```bash
# Verificar permissões do service account
gcloud projects get-iam-policy $GOOGLE_CLOUD_PROJECT
```

#### Erro de Deploy
```bash
# Verificar logs do build
gcloud builds list --limit 5

# Verificar status do serviço
gcloud run services describe shellhacks-adk-agents --region us-central1
```

### Comandos Úteis

```bash
# Ver status do serviço
gcloud run services list

# Ver logs
gcloud logging read 'resource.type="cloud_run_revision"' --limit 50

# Escalar serviço
gcloud run services update shellhacks-adk-agents --min-instances 2

# Deletar serviço
gcloud run services delete shellhacks-adk-agents --region us-central1
```

## 8. Próximos Passos

1. **Integração com Frontend**: Conectar com a aplicação Next.js
2. **Autenticação**: Implementar sistema de autenticação
3. **Cache**: Adicionar Redis para cache
4. **Queue**: Implementar filas para processamento assíncrono
5. **MLOps**: Automatizar pipeline de ML
6. **Monitoramento**: Configurar alertas avançados

## 9. Recursos Adicionais

- [Documentação Google ADK](https://cloud.google.com/vertex-ai/docs/agents)
- [Documentação Vertex AI](https://cloud.google.com/vertex-ai/docs)
- [Documentação Cloud Run](https://cloud.google.com/run/docs)
- [Documentação Cloud Monitoring](https://cloud.google.com/monitoring/docs)

## 10. Suporte

Para suporte:
- Abra uma issue no GitHub
- Consulte a documentação completa
- Entre em contato com a equipe de desenvolvimento
