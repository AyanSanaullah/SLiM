#!/usr/bin/env python3
"""
Script de teste para demonstrar como fazer requests para os modelos treinados
no sistema de agents do ShellHacks.

Este script mostra:
1. Como criar um agent personalizado
2. Como verificar o status do treinamento
3. Como fazer inferência (obter respostas do modelo)
4. Como gerenciar múltiplos agents
"""

import requests
import time
import json
from typing import Dict, Any, Optional
import sys

class AgentAPITester:
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> bool:
        """Verificar se o serviço está funcionando"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                print("✅ Serviço está funcionando!")
                return True
            else:
                print(f"❌ Serviço com problema: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print("❌ Não foi possível conectar ao serviço. Certifique-se de que está rodando em localhost:8080")
            return False
    
    def create_agent(self, user_id: str, training_data: str, base_model: str = "distilbert-base-uncased") -> Dict[str, Any]:
        """Criar um agent personalizado"""
        print(f"\n🤖 Criando agent para usuário: {user_id}")
        
        data = {
            "user_id": user_id,
            "training_data": training_data,
            "base_model": base_model
        }
        
        try:
            response = self.session.post(f"{self.base_url}/api/v1/agents", json=data)
            result = response.json()
            
            if response.status_code == 201:
                print(f"✅ Agent criado com sucesso!")
                print(f"   Status: {result.get('status')}")
                print(f"   User ID: {result.get('user_id')}")
            else:
                print(f"❌ Erro ao criar agent: {result.get('error')}")
            
            return result
            
        except Exception as e:
            print(f"❌ Erro na requisição: {e}")
            return {"error": str(e)}
    
    def get_agent_status(self, user_id: str) -> Dict[str, Any]:
        """Verificar status do agent"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/agents/{user_id}/status")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def wait_for_ready(self, user_id: str, max_wait: int = 300) -> bool:
        """Aguardar até o modelo estar pronto"""
        print(f"\n⏳ Aguardando modelo ficar pronto para {user_id}...")
        
        start_time = time.time()
        while time.time() - start_time < max_wait:
            status = self.get_agent_status(user_id)
            
            if "error" in status:
                print(f"❌ Erro ao verificar status: {status['error']}")
                return False
            
            current_status = status.get('status', 'unknown')
            print(f"   Status atual: {current_status}")
            
            if current_status == 'ready':
                print("✅ Modelo pronto!")
                return True
            elif current_status == 'error':
                print(f"❌ Erro no treinamento: {status.get('error', 'Erro desconhecido')}")
                return False
            
            time.sleep(5)
        
        print("⏰ Timeout aguardando modelo ficar pronto")
        return False
    
    def make_inference(self, user_id: str, prompt: str) -> Optional[str]:
        """Fazer inferência com o modelo treinado"""
        print(f"\n💬 Fazendo pergunta para {user_id}: {prompt}")
        
        data = {"prompt": prompt}
        
        try:
            response = self.session.post(f"{self.base_url}/api/v1/agents/{user_id}/inference", json=data)
            result = response.json()
            
            if response.status_code == 200:
                response_text = result.get('response', 'Sem resposta')
                print(f"🤖 Resposta do modelo:")
                print(f"   {response_text}")
                return response_text
            else:
                error_msg = result.get('error', 'Erro desconhecido')
                print(f"❌ Erro na inferência: {error_msg}")
                return None
                
        except Exception as e:
            print(f"❌ Erro na requisição: {e}")
            return None
    
    def list_all_agents(self) -> Dict[str, Any]:
        """Listar todos os agents"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/agents")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def delete_agent(self, user_id: str) -> bool:
        """Deletar um agent"""
        try:
            response = self.session.delete(f"{self.base_url}/api/v1/agents/{user_id}")
            if response.status_code == 200:
                print(f"✅ Agent {user_id} deletado com sucesso")
                return True
            else:
                print(f"❌ Erro ao deletar agent: {response.json().get('error')}")
                return False
        except Exception as e:
            print(f"❌ Erro na requisição: {e}")
            return False

def main():
    """Função principal para demonstrar o uso da API"""
    print("🚀 ShellHacks Agent API Tester")
    print("=" * 50)
    
    # Inicializar o tester
    tester = AgentAPITester()
    
    # Verificar se o serviço está funcionando
    if not tester.health_check():
        print("\n❌ Não é possível continuar. Verifique se o servidor está rodando:")
        print("   cd agents && python app.py")
        sys.exit(1)
    
    # Exemplo 1: Agent especialista em Python
    print("\n" + "=" * 50)
    print("📚 EXEMPLO 1: Agent Especialista em Python")
    print("=" * 50)
    
    python_expert_data = """
    Sou um especialista em programação Python com mais de 10 anos de experiência.
    Tenho conhecimento profundo em:
    - Desenvolvimento web com Flask, Django e FastAPI
    - Data Science com pandas, numpy, scikit-learn
    - Machine Learning e Deep Learning
    - Análise de dados e visualização
    - Automação e scripts
    - Boas práticas de programação e clean code
    - Arquitetura de software e design patterns
    
    Posso ajudar com dúvidas técnicas, explicações de conceitos,
    revisão de código e orientações sobre melhores práticas.
    """
    
    # Criar agent
    agent1 = tester.create_agent("python_expert", python_expert_data)
    
    if "error" not in agent1:
        # Aguardar modelo ficar pronto
        if tester.wait_for_ready("python_expert", max_wait=60):
            # Fazer algumas perguntas
            perguntas_python = [
                "Como implementar autenticação JWT em Flask?",
                "Qual a diferença entre list comprehension e generator expression?",
                "Como otimizar performance de uma query Django com muitos JOINs?"
            ]
            
            for pergunta in perguntas_python:
                tester.make_inference("python_expert", pergunta)
                print("-" * 30)
    
    # Exemplo 2: Agent especialista em Machine Learning
    print("\n" + "=" * 50)
    print("🧠 EXEMPLO 2: Agent Especialista em Machine Learning")
    print("=" * 50)
    
    ml_expert_data = """
    Sou um cientista de dados especializado em Machine Learning e Deep Learning.
    Minha expertise inclui:
    - Algoritmos de classificação e regressão
    - Redes neurais e deep learning
    - Processamento de linguagem natural (NLP)
    - Visão computacional
    - Análise de séries temporais
    - Feature engineering e seleção de features
    - Avaliação de modelos e métricas
    - TensorFlow, PyTorch, scikit-learn
    - MLOps e deploy de modelos
    
    Posso ajudar com implementação de algoritmos, análise de dados,
    escolha de modelos e otimização de performance.
    """
    
    # Criar agent
    agent2 = tester.create_agent("ml_expert", ml_expert_data)
    
    if "error" not in agent2:
        # Aguardar modelo ficar pronto
        if tester.wait_for_ready("ml_expert", max_wait=60):
            # Fazer algumas perguntas
            perguntas_ml = [
                "Como escolher entre Random Forest e XGBoost para um problema de classificação?",
                "Quais são as melhores práticas para lidar com overfitting em redes neurais?",
                "Como implementar early stopping em PyTorch?"
            ]
            
            for pergunta in perguntas_ml:
                tester.make_inference("ml_expert", pergunta)
                print("-" * 30)
    
    # Exemplo 3: Agent especialista em DevOps
    print("\n" + "=" * 50)
    print("⚙️ EXEMPLO 3: Agent Especialista em DevOps")
    print("=" * 50)
    
    devops_expert_data = """
    Sou um especialista em DevOps e engenharia de software.
    Tenho experiência em:
    - CI/CD pipelines
    - Docker e containerização
    - Kubernetes e orquestração
    - Cloud platforms (AWS, GCP, Azure)
    - Infraestrutura como código (Terraform, CloudFormation)
    - Monitoramento e observabilidade
    - Git e versionamento
    - Automação de deploy
    - Microserviços e arquitetura distribuída
    
    Posso ajudar com setup de pipelines, troubleshooting,
    otimização de infraestrutura e melhores práticas DevOps.
    """
    
    # Criar agent
    agent3 = tester.create_agent("devops_expert", devops_expert_data)
    
    if "error" not in agent3:
        # Aguardar modelo ficar pronto
        if tester.wait_for_ready("devops_expert", max_wait=60):
            # Fazer algumas perguntas
            perguntas_devops = [
                "Como configurar um pipeline CI/CD para uma aplicação Node.js?",
                "Quais são as melhores práticas para monitoramento de aplicações em produção?",
                "Como otimizar custos de infraestrutura na AWS?"
            ]
            
            for pergunta in perguntas_devops:
                tester.make_inference("devops_expert", pergunta)
                print("-" * 30)
    
    # Listar todos os agents
    print("\n" + "=" * 50)
    print("📋 LISTANDO TODOS OS AGENTS")
    print("=" * 50)
    
    all_agents = tester.list_all_agents()
    if "error" not in all_agents:
        users = all_agents.get('users', {})
        total = all_agents.get('total_users', 0)
        
        print(f"Total de agents ativos: {total}")
        for user_id, info in users.items():
            status = info.get('status', 'unknown')
            created_at = info.get('created_at', 'unknown')
            print(f"  - {user_id}: {status} (criado em {created_at})")
    
    # Menu interativo
    print("\n" + "=" * 50)
    print("🎮 MODO INTERATIVO")
    print("=" * 50)
    print("Digite 'help' para ver comandos disponíveis")
    
    while True:
        try:
            comando = input("\n> ").strip().lower()
            
            if comando == 'help':
                print("""
Comandos disponíveis:
- status <user_id>: Verificar status de um agent
- ask <user_id> <pergunta>: Fazer pergunta a um agent
- list: Listar todos os agents
- delete <user_id>: Deletar um agent
- quit: Sair
                """)
            
            elif comando.startswith('status '):
                user_id = comando.split(' ', 1)[1]
                status = tester.get_agent_status(user_id)
                print(json.dumps(status, indent=2, ensure_ascii=False))
            
            elif comando.startswith('ask '):
                parts = comando.split(' ', 2)
                if len(parts) >= 3:
                    user_id = parts[1]
                    pergunta = parts[2]
                    tester.make_inference(user_id, pergunta)
                else:
                    print("Uso: ask <user_id> <pergunta>")
            
            elif comando == 'list':
                all_agents = tester.list_all_agents()
                if "error" not in all_agents:
                    users = all_agents.get('users', {})
                    for user_id, info in users.items():
                        status = info.get('status', 'unknown')
                        print(f"  - {user_id}: {status}")
            
            elif comando.startswith('delete '):
                user_id = comando.split(' ', 1)[1]
                tester.delete_agent(user_id)
            
            elif comando == 'quit':
                break
            
            else:
                print("Comando não reconhecido. Digite 'help' para ver comandos disponíveis.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Saindo...")
            break
        except Exception as e:
            print(f"Erro: {e}")

if __name__ == "__main__":
    main()
