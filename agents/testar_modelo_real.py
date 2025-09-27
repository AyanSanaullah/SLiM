#!/usr/bin/env python3
"""
Script para testar o sistema REAL de treinamento de modelos
Demonstra o treinamento de modelos personalizados usando dados de prompt/answer
e avaliação usando string comparison
"""

import requests
import time
import json
from typing import Dict, Any

class RealModelTester:
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
    
    def test_real_model_training(self):
        """Testa o sistema completo de treinamento real"""
        print("🚀 TESTE DO SISTEMA REAL DE TREINAMENTO DE MODELOS")
        print("=" * 60)
        
        # 1. Verificar se o servidor está rodando
        if not self._check_server():
            return
        
        # 2. Criar agent com dados de treinamento ricos
        training_data = """
        Sou especialista em Python e desenvolvimento web. Tenho experiência com:
        
        - Flask: Framework web minimalista para Python
        - Django: Framework web robusto para aplicações complexas
        - FastAPI: Framework moderno para APIs rápidas
        - SQLAlchemy: ORM para bancos de dados
        - Pandas: Biblioteca para análise de dados
        - NumPy: Computação científica com arrays
        - Scikit-learn: Machine learning em Python
        - TensorFlow: Deep learning e redes neurais
        
        Posso ajudar com:
        - Criação de APIs REST
        - Desenvolvimento de aplicações web
        - Análise de dados
        - Machine learning
        - Debugging e otimização de código
        - Boas práticas de programação
        """
        
        user_id = "real_python_expert"
        
        print("🤖 Criando agent com treinamento REAL...")
        agent_result = self._create_agent(user_id, training_data)
        
        if "error" in agent_result:
            print(f"❌ Erro: {agent_result['error']}")
            return
        
        print("✅ Agent criado! Aguardando treinamento...")
        
        # 3. Aguardar treinamento completar
        if not self._wait_for_training(user_id):
            return
        
        # 4. Testar inferência
        print("\n💬 TESTANDO INFERÊNCIA COM MODELO TREINADO")
        print("-" * 50)
        
        test_prompts = [
            "Como criar uma API REST com Flask?",
            "O que é Django?",
            "Como usar pandas para análise de dados?",
            "Explique sobre machine learning",
            "Como debugar código Python?",
            "O que são boas práticas de programação?"
        ]
        
        for prompt in test_prompts:
            print(f"\n🤔 Pergunta: {prompt}")
            response = self._make_inference(user_id, prompt)
            print(f"🤖 Resposta: {response}")
            print("-" * 30)
        
        # 5. Testar avaliação com string comparison
        print("\n📊 TESTANDO AVALIAÇÃO COM STRING COMPARISON")
        print("-" * 50)
        
        evaluation_tests = [
            {
                "prompt": "O que é Flask?",
                "expected": "Flask é um framework web minimalista para Python"
            },
            {
                "prompt": "Como usar pandas?",
                "expected": "Pandas é uma biblioteca para análise de dados em Python"
            },
            {
                "prompt": "O que é machine learning?",
                "expected": "Machine learning é uma técnica de inteligência artificial"
            }
        ]
        
        for test in evaluation_tests:
            print(f"\n🧪 Teste: {test['prompt']}")
            print(f"📝 Resposta esperada: {test['expected']}")
            
            eval_result = self._evaluate_model(user_id, test['prompt'], test['expected'])
            
            if eval_result:
                print(f"🤖 Resposta do modelo: {eval_result.get('predicted_answer', 'N/A')}")
                print(f"📊 Similaridade geral: {eval_result.get('overall_similarity', 0):.2%}")
                print(f"📈 Similaridade Jaccard: {eval_result.get('jaccard_similarity', 0):.2%}")
                print(f"🔤 Similaridade de caracteres: {eval_result.get('character_similarity', 0):.2%}")
                print(f"📏 Confiança do modelo: {eval_result.get('model_confidence', 0):.2%}")
            
            print("-" * 30)
        
        # 6. Mostrar status final
        print("\n📋 STATUS FINAL DO MODELO")
        print("-" * 30)
        status = self._get_status(user_id)
        if status:
            print(json.dumps(status, indent=2, ensure_ascii=False))
        
        print("\n🎉 TESTE COMPLETO!")
        print("✅ O sistema agora treina modelos REAIS usando:")
        print("   - Dados de prompt/answer extraídos do treinamento")
        print("   - Modelos TF-IDF para similaridade semântica")
        print("   - Avaliação usando string comparison")
        print("   - Métricas de confiança e similaridade")
    
    def _check_server(self) -> bool:
        """Verifica se o servidor está rodando"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                print("✅ Servidor está funcionando!")
                return True
            else:
                print("❌ Servidor com problemas")
                return False
        except:
            print("❌ Servidor não está rodando. Execute: python3 app.py")
            return False
    
    def _create_agent(self, user_id: str, training_data: str) -> Dict[str, Any]:
        """Cria um agent"""
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/agents",
                json={
                    "user_id": user_id,
                    "training_data": training_data,
                    "base_model": "distilbert-base-uncased"
                }
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def _wait_for_training(self, user_id: str, max_wait: int = 60) -> bool:
        """Aguarda o treinamento completar"""
        print("⏳ Aguardando treinamento do modelo...")
        
        for i in range(max_wait // 5):
            status = self._get_status(user_id)
            if status:
                current_status = status.get('status', 'unknown')
                print(f"   Status: {current_status}")
                
                if current_status == 'ready':
                    print("✅ Modelo treinado e pronto!")
                    return True
                elif current_status == 'error':
                    print(f"❌ Erro no treinamento: {status.get('error')}")
                    return False
            
            time.sleep(5)
        
        print("⏰ Timeout no treinamento")
        return False
    
    def _get_status(self, user_id: str) -> Dict[str, Any]:
        """Obtém status do agent"""
        try:
            response = requests.get(f"{self.base_url}/api/v1/agents/{user_id}/status")
            return response.json()
        except:
            return {}
    
    def _make_inference(self, user_id: str, prompt: str) -> str:
        """Faz inferência"""
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/agents/{user_id}/inference",
                json={"prompt": prompt}
            )
            result = response.json()
            return result.get('response', 'Erro na resposta')
        except Exception as e:
            return f"Erro: {e}"
    
    def _evaluate_model(self, user_id: str, prompt: str, expected: str) -> Dict[str, Any]:
        """Avalia modelo usando string comparison"""
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/agents/{user_id}/evaluate",
                json={
                    "test_prompt": prompt,
                    "expected_answer": expected
                }
            )
            result = response.json()
            return result.get('evaluation', {})
        except Exception as e:
            print(f"Erro na avaliação: {e}")
            return {}

def main():
    """Função principal"""
    tester = RealModelTester()
    tester.test_real_model_training()

if __name__ == "__main__":
    main()
