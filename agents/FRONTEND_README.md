# 🌐 Frontend Dashboard - ShellHacks Agents

Interface web completa para gerenciar e monitorar agentes de treinamento de modelos.

## 🚀 Início Rápido

```bash
# 1. Abrir frontend automaticamente
./open_frontend.sh

# 2. Ou abrir manualmente
open frontend.html
```

## 📊 Funcionalidades Principais

### 🤖 **Gestão de Agentes**
- ✅ **Listar todos os agentes** com atualização automática
- ✅ **Acurácia real** baseada no string comparison (não estimativa)
- ✅ **Interface única** para cada agente
- ✅ **Métricas detalhadas** de performance
- ✅ **Status em tempo real** de cada agente

### ➕ **Criação de Agentes**
- ✅ **Agentes Básicos**: Texto simples de treinamento
- ✅ **Agentes Avançados**: JSON dataset com prompt/answer pairs
- ✅ **Validação automática** de dados
- ✅ **Feedback visual** do processo de criação

### 🧪 **Teste e Avaliação**
- ✅ **Inferência interativa** com qualquer agente
- ✅ **Avaliação por string comparison** com métricas reais
- ✅ **Visualização de similaridade** (Jaccard, caracteres, etc.)
- ✅ **Percentual exato** de acurácia

### 📡 **Monitoramento de API**
- ✅ **Todos os endpoints** (GET, POST, DELETE) visíveis
- ✅ **Teste direto** de cada endpoint
- ✅ **Status dos serviços** em tempo real
- ✅ **Respostas formatadas** em JSON

## 🎯 Como Usar

### 1. **Listar Agentes**
1. Clique na aba "🤖 Agentes"
2. Clique em "🔄 Atualizar Lista de Agentes"
3. Veja todos os agentes com:
   - **Acurácia real** do string comparison
   - **Métricas de performance**
   - **Status atual**
   - **Informações de dataset**

### 2. **Criar Agente Básico**
1. Vá para "➕ Criar Agente" → "Básico"
2. Preencha:
   - **User ID**: Nome único do agente
   - **Dados de Treinamento**: Expertise do agente
   - **Modelo Base**: Escolha o modelo base
3. Clique "🚀 Criar Agente Básico"

### 3. **Criar Agente Avançado**
1. Vá para "➕ Criar Agente" → "Avançado (JSON)"
2. Preencha:
   - **User ID**: Nome único do agente
   - **Dataset JSON**: Array de objetos `{"prompt": "...", "answer": "..."}`
3. Clique "🚀 Criar Agente Avançado"

### 4. **Testar Agente**
1. Vá para "🧪 Testar"
2. Preencha:
   - **User ID**: Nome do agente
   - **Pergunta**: Sua pergunta
3. Clique "💬 Fazer Pergunta"

### 5. **Avaliar com String Comparison**
1. Na aba "🧪 Testar"
2. Preencha:
   - **Pergunta**: Prompt de teste
   - **Resposta Esperada**: Resposta correta
3. Clique "📈 Avaliar Modelo"
4. Veja a **acurácia real** baseada na comparação de strings

## 📊 Métricas Exibidas

### **Por Agente**
- 🎯 **Acurácia Real**: Percentual baseado no string comparison
- 📅 **Data de Criação**: Quando foi criado
- 🤖 **Tipo**: Básico ou avançado (QLoRA)
- 📊 **Tamanho do Dataset**: Número de samples
- ✅ **Testes Bem-sucedidos**: Quantos passaram
- 🟢 **Alta Qualidade**: Quantos tiveram >80% similaridade

### **Por Avaliação**
- 🎯 **Similaridade Jaccard**: Comparação por palavras
- 🔤 **Similaridade de Caracteres**: Comparação por caracteres
- 📏 **Confiança do Modelo**: Confiança interna do modelo
- 🏆 **Similaridade Geral**: Métrica combinada

## 🔧 Recursos Técnicos

### **Interface Responsiva**
- ✅ Design moderno com gradientes
- ✅ Cards interativos para cada agente
- ✅ Barras de progresso visuais
- ✅ Indicadores de status coloridos
- ✅ Atualização automática a cada 30s

### **Integração Completa**
- ✅ Consome API local (localhost:8080)
- ✅ Integra string comparison (localhost:8000)
- ✅ Validação de formulários
- ✅ Tratamento de erros
- ✅ Loading states visuais

### **Recursos Avançados**
- ✅ Verificação automática de status dos serviços
- ✅ Formatação JSON para respostas
- ✅ Tabs organizadas por funcionalidade
- ✅ Botão flutuante de refresh
- ✅ Confirmação para ações destrutivas

## 🎨 Layout e Organização

### **Header**
- Status dos serviços (🟢 Online / 🔴 Offline)
- Título e descrição do sistema

### **Tabs Principais**
1. **🤖 Agentes**: Lista e gerencia agentes existentes
2. **➕ Criar Agente**: Formulários para novos agentes
3. **🧪 Testar**: Interface de teste e avaliação
4. **📡 API Endpoints**: Documentação e teste de endpoints

### **Cartões de Agente**
- **Header**: Nome e status
- **Métricas**: Todas as informações importantes
- **Barra de Acurácia**: Visualização da performance
- **Ações**: Detalhes, Testar, Deletar

## 💡 Exemplos de Uso

### **Dataset JSON para Agente Avançado**

```json
[
  {
    "prompt": "Como criar uma API REST?",
    "answer": "Use Flask com @app.route() para definir endpoints HTTP"
  },
  {
    "prompt": "O que é machine learning?",
    "answer": "ML é uma subárea da IA que permite sistemas aprenderem de dados"
  },
  {
    "prompt": "Como usar Docker?",
    "answer": "Docker cria containers portáteis para aplicações"
  }
]
```

### **Especialistas Sugeridos**

#### Python Expert
```json
{
  "user_id": "python_expert",
  "json_dataset": [
    {"prompt": "Como criar API?", "answer": "Use Flask com @app.route()"},
    {"prompt": "O que é Django?", "answer": "Framework web Python completo"},
    {"prompt": "Como usar pandas?", "answer": "Biblioteca para análise de dados"}
  ]
}
```

#### DevOps Expert
```json
{
  "user_id": "devops_expert",
  "json_dataset": [
    {"prompt": "Como configurar CI/CD?", "answer": "Use GitHub Actions ou GitLab CI"},
    {"prompt": "O que é Kubernetes?", "answer": "Orquestrador de containers"},
    {"prompt": "Como monitorar apps?", "answer": "Use Prometheus + Grafana"}
  ]
}
```

## 🔍 Troubleshooting

### **Serviços Offline**
- Verifique se os serviços estão rodando:
  ```bash
  python3 app.py                              # Agents
  cd ../string-comparison && python3 backend.py  # String Comparison
  ```

### **Erro de CORS**
- O frontend usa `file://` protocol
- API está configurada para aceitar requisições locais
- Verifique se as URLs estão corretas

### **Agentes não aparecem**
- Clique em "🔄 Atualizar Lista de Agentes"
- Verifique se há agentes criados
- Confira se o serviço está online

### **Métricas não aparecem**
- Agentes básicos: Execute um teste primeiro
- Agentes avançados: Aguarde o treinamento completar
- Métricas são baseadas em avaliações reais

## 🎯 Diferencial

### **Acurácia Real, Não Estimada**
- ✅ Usa string comparison real entre respostas
- ✅ Métricas Jaccard, caracteres e similaridade geral
- ✅ Não há "chute" de percentual
- ✅ Cada agente tem modelo único treinado

### **Interface Única por Agente**
- ✅ Cada agente tem seu card individual
- ✅ Métricas específicas por agente
- ✅ Status independente
- ✅ Ações individuais

### **Monitoramento Completo**
- ✅ Todos os endpoints HTTP visíveis
- ✅ Teste direto da interface
- ✅ Status dos serviços em tempo real
- ✅ Logs visuais de performance

---

**🚀 O frontend está pronto para uso completo com todas as funcionalidades solicitadas!**
