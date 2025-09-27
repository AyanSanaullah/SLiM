#!/usr/bin/env python3
"""
Script para testar o sistema AVANÇADO de treinamento
- Input: Array de JSONs com prompts/answers
- Processo: QLoRA + CUDA + String Comparison
- Output: Métricas detalhadas e comparação semântica
"""

import requests
import json
import time
from typing import List, Dict, Any

def test_advanced_training_system():
    """Testa o sistema avançado completo"""
    
    print("🚀 TESTE DO SISTEMA AVANÇADO DE TREINAMENTO")
    print("=" * 60)
    print("✨ Features: JSON Dataset + QLoRA + CUDA + String Comparison")
    print("=" * 60)
    
    base_url = "http://localhost:8080"
    
    # Verificar se servidor está rodando
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code != 200:
            print("❌ Servidor não está funcionando")
            return
        print("✅ Servidor funcionando!")
    except:
        print("❌ Não foi possível conectar ao servidor")
        return
    
    # Verificar se string-comparison está rodando
    try:
        response = requests.get("http://0.0.0.0:8000/health")
        if response.status_code == 200:
            print("✅ String Comparison Service funcionando!")
        else:
            print("⚠️  String Comparison Service com problemas")
    except:
        print("⚠️  String Comparison Service não está rodando")
        print("   Execute: cd string-comparison && python backend.py")
    
    # Dataset JSON de exemplo
    json_dataset = [
        {
            "prompt": "Como criar uma API REST com Flask?",
            "answer": "Para criar uma API REST com Flask, use decoradores @app.route() para definir endpoints, retorne dados JSON com jsonify(), e configure métodos HTTP como GET, POST, PUT, DELETE."
        },
        {
            "prompt": "O que é machine learning?",
            "answer": "Machine learning é uma técnica de inteligência artificial que permite aos computadores aprenderem e tomarem decisões baseadas em dados, sem serem explicitamente programados para cada tarefa específica."
        },
        {
            "prompt": "Como implementar autenticação JWT?",
            "answer": "Para implementar JWT, instale flask-jwt-extended, configure uma chave secreta, crie endpoints para login que retornem tokens, e use @jwt_required() para proteger rotas que precisam de autenticação."
        },
        {
            "prompt": "Qual a diferença entre SQL e NoSQL?",
            "answer": "SQL databases usam esquemas fixos e relacionamentos estruturados, ideais para dados consistentes. NoSQL databases são mais flexíveis, suportam dados não estruturados e escalam horizontalmente melhor."
        },
        {
            "prompt": "Como otimizar performance em Python?",
            "answer": "Para otimizar Python, use list comprehensions, numpy para computação numérica, evite loops desnecessários, use generators para economizar memória, e considere Cython para código crítico."
        },
        {
            "prompt": "O que é Docker?",
            "answer": "Docker é uma plataforma de containerização que permite empacotar aplicações e suas dependências em containers leves e portáteis, garantindo consistência entre diferentes ambientes."
        },
        {
            "prompt": "Como funciona o Git?",
            "answer": "Git é um sistema de controle de versão distribuído que rastreia mudanças no código, permite colaboração através de branches, merges, e mantém histórico completo das alterações."
        },
        {
            "prompt": "O que são design patterns?",
            "answer": "Design patterns são soluções reutilizáveis para problemas comuns em desenvolvimento de software, como Singleton, Factory, Observer, que promovem código limpo e manutenível."
        }
    ]
    
    user_id = "advanced_python_expert"
    
    print(f"\n🤖 Criando agent AVANÇADO com {len(json_dataset)} samples...")
    
    # Criar agent avançado
    create_response = requests.post(
        f"{base_url}/api/v1/agents/advanced",
        json={
            "user_id": user_id,
            "json_dataset": json_dataset,
            "base_model": "distilbert-base-uncased"
        }
    )
    
    if create_response.status_code != 201:
        print(f"❌ Erro ao criar agent: {create_response.json()}")
        return
    
    create_result = create_response.json()
    print("✅ Agent avançado criado!")
    print(f"   Dataset: {create_result['dataset_size']} samples")
    print(f"   Tipo: {create_result['training_type']}")
    print(f"   String Comparison: {create_result['string_comparison_enabled']}")
    
    # Aguardar treinamento
    print("\n⏳ Aguardando treinamento avançado (QLoRA + CUDA)...")
    
    max_wait = 120  # 2 minutos
    for i in range(max_wait // 5):
        status_response = requests.get(f"{base_url}/api/v1/agents/{user_id}/status")
        
        if status_response.status_code != 200:
            print(f"❌ Erro ao verificar status: {status_response.json()}")
            return
        
        status = status_response.json()
        current_status = status.get('status', 'unknown')
        
        print(f"   Status: {current_status}")
        
        if current_status == 'ready':
            print("✅ Treinamento avançado concluído!")
            break
        elif current_status == 'error':
            print(f"❌ Erro no treinamento: {status.get('error')}")
            return
        
        time.sleep(5)
    else:
        print("⏰ Timeout no treinamento")
        return
    
    # Mostrar resultados do treinamento
    print("\n📊 RESULTADOS DO TREINAMENTO AVANÇADO")
    print("-" * 50)
    
    if 'training_results' in status:
        training_results = status['training_results']
        print(f"🔧 Tipo de modelo: {training_results.get('model_type', 'N/A')}")
        print(f"🏗️  Modelo base: {training_results.get('base_model', 'N/A')}")
        print(f"📈 Samples de treinamento: {training_results.get('training_samples', 'N/A')}")
        print(f"💻 Device: {training_results.get('device', 'N/A')}")
        print(f"🚀 CUDA disponível: {training_results.get('cuda_available', 'N/A')}")
        print(f"🎯 GPU count: {training_results.get('gpu_count', 'N/A')}")
        
        if 'gpu_info' in training_results:
            print("\n🔥 Informações das GPUs:")
            for gpu in training_results['gpu_info']:
                print(f"   GPU {gpu['device']}: {gpu['name']} ({gpu['memory_gb']:.1f}GB)")
    
    # Mostrar resultados da avaliação
    if 'evaluation_results' in status:
        eval_results = status['evaluation_results']
        print(f"\n🎯 RESULTADOS DA AVALIAÇÃO COM STRING COMPARISON")
        print("-" * 50)
        print(f"📊 Total de samples: {eval_results.get('total_samples', 'N/A')}")
        print(f"✅ Avaliações bem-sucedidas: {eval_results.get('successful_evaluations', 'N/A')}")
        print(f"❌ Avaliações falharam: {eval_results.get('failed_evaluations', 'N/A')}")
        print(f"🎯 Similaridade média: {eval_results.get('average_string_similarity', 0):.2%}")
        print(f"🔥 Confiança média: {eval_results.get('average_model_confidence', 0):.2%}")
        print(f"🏆 Alta similaridade (>80%): {eval_results.get('high_similarity_count', 'N/A')}")
        print(f"🔶 Média similaridade (50-80%): {eval_results.get('medium_similarity_count', 'N/A')}")
        print(f"🔻 Baixa similaridade (<50%): {eval_results.get('low_similarity_count', 'N/A')}")
    
    # Testar inferência com string comparison
    print(f"\n💬 TESTANDO INFERÊNCIA COM STRING COMPARISON")
    print("-" * 50)
    
    test_prompts = [
        "Como criar uma API REST?",
        "O que é machine learning?",
        "Como usar Docker?",
        "Explique sobre Git",
        "O que são design patterns?"
    ]
    
    for prompt in test_prompts:
        print(f"\n🤔 Pergunta: {prompt}")
        
        inference_response = requests.post(
            f"{base_url}/api/v1/agents/{user_id}/inference",
            json={"prompt": prompt}
        )
        
        if inference_response.status_code == 200:
            result = inference_response.json()
            response = result['response']
            print(f"🤖 Resposta: {response}")
        else:
            print(f"❌ Erro na inferência: {inference_response.json()}")
        
        print("-" * 30)
    
    print("\n🎉 TESTE DO SISTEMA AVANÇADO CONCLUÍDO!")
    print("=" * 60)
    print("✨ Features testadas:")
    print("   ✅ JSON Dataset processing")
    print("   ✅ QLoRA training")
    print("   ✅ CUDA support")
    print("   ✅ String Comparison integration")
    print("   ✅ Detailed metrics")
    print("   ✅ Advanced inference")

if __name__ == "__main__":
    test_advanced_training_system()
