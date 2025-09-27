#!/usr/bin/env python3
"""
Script simples para listar os agentes existentes no projeto
"""

import requests
import json

def listar_agentes():
    """Lista todos os agentes ativos"""
    base_url = "http://localhost:8080"
    
    try:
        # Verificar se o servidor está rodando
        health_response = requests.get(f"{base_url}/health", timeout=5)
        if health_response.status_code != 200:
            print("❌ Servidor não está rodando. Execute: python3 app.py")
            return
        
        print("✅ Servidor está funcionando!")
        
        # Listar agentes
        response = requests.get(f"{base_url}/api/v1/agents")
        
        if response.status_code == 200:
            data = response.json()
            total = data.get('total_users', 0)
            
            print(f"\n📋 Total de agentes ativos: {total}")
            
            if total == 0:
                print("   Nenhum agent encontrado.")
                print("   Para criar um agent, use:")
                print("   python3 exemplo_rapido.py")
            else:
                print("\n🤖 Agentes encontrados:")
                print("-" * 50)
                
                users = data.get('users', {})
                for user_id, info in users.items():
                    status = info.get('status', 'unknown')
                    created_at = info.get('created_at', 'unknown')
                    base_model = info.get('base_model', 'unknown')
                    
                    # Formatar data
                    if created_at != 'unknown':
                        try:
                            from datetime import datetime
                            dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                            created_at = dt.strftime('%d/%m/%Y %H:%M')
                        except:
                            pass
                    
                    print(f"👤 User ID: {user_id}")
                    print(f"   Status: {status}")
                    print(f"   Base Model: {base_model}")
                    print(f"   Criado em: {created_at}")
                    
                    if 'model_ready_at' in info:
                        print(f"   Modelo pronto em: {info['model_ready_at']}")
                    
                    if 'training_data_size' in info:
                        print(f"   Tamanho dos dados: {info['training_data_size']} caracteres")
                    
                    print("-" * 50)
                
                print(f"\n💡 Para fazer perguntas a um agent:")
                print(f"   curl -X POST {base_url}/api/v1/agents/USER_ID/inference \\")
                print(f"     -H 'Content-Type: application/json' \\")
                print(f"     -d '{{\"prompt\": \"Sua pergunta aqui\"}}'")
        
        else:
            error_data = response.json()
            print(f"❌ Erro ao listar agentes: {error_data.get('error', 'Erro desconhecido')}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Não foi possível conectar ao servidor.")
        print("   Certifique-se de que o servidor está rodando:")
        print("   python3 app.py")
    except Exception as e:
        print(f"❌ Erro: {e}")

if __name__ == "__main__":
    print("🔍 Listando agentes existentes...")
    print("=" * 40)
    listar_agentes()
