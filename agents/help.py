#!/usr/bin/env python3
"""
Script de ajuda que mostra todos os comandos disponíveis no projeto
"""

import sys
import requests

def show_help():
    """Mostra menu de ajuda com todos os comandos"""
    
    print("🚀 SHELLHACKS AGENTS - COMANDOS DISPONÍVEIS")
    print("=" * 60)
    
    # Verificar status dos serviços
    agents_status = check_service("http://localhost:8080/health", "Agents Service")
    string_status = check_service("http://localhost:8000/health", "String Comparison")
    
    print(f"\n📊 STATUS DOS SERVIÇOS:")
    print(f"   Agents Service (8080): {'🟢 Online' if agents_status else '🔴 Offline'}")
    print(f"   String Comparison (8000): {'🟢 Online' if string_status else '🔴 Offline'}")
    
    print(f"\n🔧 COMANDOS DE INICIALIZAÇÃO:")
    print(f"   Iniciar Agents:           python3 app.py")
    print(f"   Iniciar String Compare:   cd ../string-comparison && python3 backend.py")
    print(f"   Iniciar Ambos:            ./start_services.sh")
    
    print(f"\n🤖 CRIAR AGENTES:")
    print(f"   Agente Básico:            curl -X POST localhost:8080/api/v1/agents -H 'Content-Type: application/json' -d '{{\"user_id\": \"expert\", \"training_data\": \"Sou especialista em...\"}}'")
    print(f"   Agente Avançado:          curl -X POST localhost:8080/api/v1/agents/advanced -H 'Content-Type: application/json' -d '{{\"user_id\": \"advanced\", \"json_dataset\": [...]}}'")
    
    print(f"\n📋 GERENCIAMENTO:")
    print(f"   Listar Agentes:           curl localhost:8080/api/v1/agents")
    print(f"   Status do Agente:         curl localhost:8080/api/v1/agents/USER_ID/status")
    print(f"   Pipeline do Agente:       curl localhost:8080/api/v1/agents/USER_ID/pipeline")
    print(f"   Deletar Agente:           curl -X DELETE localhost:8080/api/v1/agents/USER_ID")
    
    print(f"\n💬 INFERÊNCIA:")
    print(f"   Fazer Pergunta:           curl -X POST localhost:8080/api/v1/agents/USER_ID/inference -H 'Content-Type: application/json' -d '{{\"prompt\": \"Sua pergunta\"}}'")
    print(f"   Avaliar Resposta:         curl -X POST localhost:8080/api/v1/agents/USER_ID/evaluate -H 'Content-Type: application/json' -d '{{\"test_prompt\": \"...\", \"expected_answer\": \"...\"}}'")
    
    print(f"\n🧪 TESTES AUTOMÁTICOS:")
    print(f"   Teste Sistema Avançado:   python3 test_advanced_system.py")
    print(f"   Teste Sistema Básico:     python3 test_agent_api.py")
    print(f"   Listar Agentes:           python3 listar_agentes.py")
    
    print(f"\n🔍 MONITORAMENTO:")
    print(f"   Health Check Agents:      curl localhost:8080/health")
    print(f"   Health Check String:      curl localhost:8000/health")
    print(f"   Ver Configuração:         curl localhost:8080/api/v1/config")
    
    print(f"\n📚 DOCUMENTAÇÃO:")
    print(f"   Comandos Completos:       cat COMANDOS_COMPLETOS.md")
    print(f"   Comandos Rápidos:         cat COMANDOS_RAPIDOS.md")
    print(f"   Como Usar:                cat COMO_USAR.md")
    print(f"   README Principal:         cat README.md")
    
    print(f"\n🛠️ UTILITÁRIOS:")
    print(f"   Setup Inicial:            ./setup.sh")
    print(f"   Deploy Cloud:             ./deploy.sh cloud-run")
    print(f"   Parar Serviços:           killall Python")
    print(f"   Limpar Dados:             rm -rf data/user_data/* models/user_models/* logs/user_logs/*")
    
    print(f"\n💡 EXEMPLOS RÁPIDOS:")
    show_examples()
    
    print(f"\n❓ Para mais detalhes, veja:")
    print(f"   📖 COMANDOS_COMPLETOS.md - Documentação completa")
    print(f"   ⚡ COMANDOS_RAPIDOS.md - Comandos essenciais")
    print(f"   🎯 COMO_USAR.md - Guia prático")

def check_service(url, name):
    """Verifica se um serviço está online"""
    try:
        response = requests.get(url, timeout=2)
        return response.status_code == 200
    except:
        return False

def show_examples():
    """Mostra exemplos práticos"""
    print(f"\n   🐍 Python Expert:")
    print(f"   curl -X POST localhost:8080/api/v1/agents/advanced \\")
    print(f"     -H 'Content-Type: application/json' \\")
    print(f"     -d '{{")
    print(f"       \"user_id\": \"python_expert\",")
    print(f"       \"json_dataset\": [")
    print(f"         {{\"prompt\": \"Como criar API?\", \"answer\": \"Use Flask com @app.route()\"}},")
    print(f"         {{\"prompt\": \"O que é Django?\", \"answer\": \"Framework web Python\"}}") 
    print(f"       ]")
    print(f"     }}'")
    
    print(f"\n   🔧 DevOps Expert:")
    print(f"   curl -X POST localhost:8080/api/v1/agents/advanced \\")
    print(f"     -H 'Content-Type: application/json' \\")
    print(f"     -d '{{")
    print(f"       \"user_id\": \"devops_expert\",")
    print(f"       \"json_dataset\": [")
    print(f"         {{\"prompt\": \"Como usar Docker?\", \"answer\": \"Docker cria containers\"}},")
    print(f"         {{\"prompt\": \"O que é CI/CD?\", \"answer\": \"Integração e deploy contínuo\"}}") 
    print(f"       ]")
    print(f"     }}'")

def interactive_menu():
    """Menu interativo para comandos"""
    while True:
        print(f"\n🎮 MENU INTERATIVO")
        print(f"1. Ver status dos serviços")
        print(f"2. Listar agentes")
        print(f"3. Criar agente básico")
        print(f"4. Fazer pergunta")
        print(f"5. Executar teste automático")
        print(f"6. Ver documentação")
        print(f"0. Sair")
        
        choice = input(f"\n➤ Escolha uma opção: ").strip()
        
        if choice == '1':
            check_services()
        elif choice == '2':
            list_agents()
        elif choice == '3':
            create_basic_agent()
        elif choice == '4':
            make_inference()
        elif choice == '5':
            run_test()
        elif choice == '6':
            show_docs()
        elif choice == '0':
            print("👋 Saindo...")
            break
        else:
            print("❌ Opção inválida")

def check_services():
    """Verifica status dos serviços"""
    print(f"\n🔍 Verificando serviços...")
    
    agents_status = check_service("http://localhost:8080/health", "Agents")
    string_status = check_service("http://localhost:8000/health", "String Comparison")
    
    print(f"Agents Service: {'🟢 Online' if agents_status else '🔴 Offline'}")
    print(f"String Comparison: {'🟢 Online' if string_status else '🔴 Offline'}")
    
    if not agents_status:
        print(f"💡 Para iniciar: python3 app.py")
    if not string_status:
        print(f"💡 Para iniciar: cd ../string-comparison && python3 backend.py")

def list_agents():
    """Lista agentes existentes"""
    try:
        response = requests.get("http://localhost:8080/api/v1/agents")
        if response.status_code == 200:
            data = response.json()
            users = data.get('users', {})
            print(f"\n📋 Agentes encontrados ({len(users)}):")
            for user_id, info in users.items():
                status = info.get('status', 'unknown')
                training_type = info.get('training_type', 'basic')
                print(f"   - {user_id}: {status} ({training_type})")
        else:
            print(f"❌ Erro ao listar agentes")
    except:
        print(f"❌ Não foi possível conectar ao serviço")

def create_basic_agent():
    """Cria um agente básico interativamente"""
    user_id = input("➤ User ID: ").strip()
    if not user_id:
        print("❌ User ID é obrigatório")
        return
    
    expertise = input("➤ Área de expertise: ").strip()
    if not expertise:
        expertise = "programação"
    
    training_data = f"Sou especialista em {expertise}. Posso ajudar com dúvidas e explicações sobre este assunto."
    
    try:
        response = requests.post(
            "http://localhost:8080/api/v1/agents",
            json={
                "user_id": user_id,
                "training_data": training_data
            }
        )
        if response.status_code == 201:
            print(f"✅ Agente '{user_id}' criado com sucesso!")
        else:
            print(f"❌ Erro ao criar agente: {response.json()}")
    except:
        print(f"❌ Não foi possível conectar ao serviço")

def make_inference():
    """Faz inferência interativamente"""
    user_id = input("➤ User ID do agente: ").strip()
    if not user_id:
        print("❌ User ID é obrigatório")
        return
    
    prompt = input("➤ Sua pergunta: ").strip()
    if not prompt:
        print("❌ Pergunta é obrigatória")
        return
    
    try:
        response = requests.post(
            f"http://localhost:8080/api/v1/agents/{user_id}/inference",
            json={"prompt": prompt}
        )
        if response.status_code == 200:
            result = response.json()
            print(f"\n🤖 Resposta: {result['response']}")
        else:
            print(f"❌ Erro: {response.json()}")
    except:
        print(f"❌ Não foi possível conectar ao serviço")

def run_test():
    """Executa teste automático"""
    import subprocess
    print(f"\n🧪 Executando teste automático...")
    try:
        subprocess.run(["python3", "test_advanced_system.py"])
    except:
        print(f"❌ Erro ao executar teste")

def show_docs():
    """Mostra documentação disponível"""
    print(f"\n📚 Documentação disponível:")
    print(f"   - COMANDOS_COMPLETOS.md: Guia completo com todos os comandos")
    print(f"   - COMANDOS_RAPIDOS.md: Comandos essenciais para uso diário")
    print(f"   - COMO_USAR.md: Guia prático de uso")
    print(f"   - README.md: Visão geral do projeto")
    print(f"   - GUIA_REQUESTS.md: Como fazer requests para os modelos")

def main():
    """Função principal"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--interactive" or sys.argv[1] == "-i":
            interactive_menu()
        elif sys.argv[1] == "--services" or sys.argv[1] == "-s":
            check_services()
        elif sys.argv[1] == "--agents" or sys.argv[1] == "-a":
            list_agents()
        else:
            print(f"Uso: python3 help.py [--interactive|-i] [--services|-s] [--agents|-a]")
    else:
        show_help()

if __name__ == "__main__":
    main()
