#!/bin/bash

echo "🌐 Abrindo Frontend do ShellHacks Agents"
echo "======================================="

# Verificar se o arquivo frontend.html existe
if [ ! -f "frontend.html" ]; then
    echo "❌ Arquivo frontend.html não encontrado!"
    echo "   Execute este script no diretório agents/"
    exit 1
fi

# Verificar se os serviços estão rodando
echo "🔍 Verificando serviços..."

# Verificar Agents Service
if curl -s http://localhost:8080/health > /dev/null; then
    echo "✅ Agents Service (localhost:8080) está online"
else
    echo "⚠️  Agents Service (localhost:8080) está offline"
    echo "   Para iniciar: python3 app.py"
fi

# Verificar String Comparison Service
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ String Comparison Service (localhost:8000) está online"
else
    echo "⚠️  String Comparison Service (localhost:8000) está offline"
    echo "   Para iniciar: cd ../string-comparison && python3 backend.py"
fi

echo ""
echo "🚀 Iniciando serviços se necessário..."

# Iniciar serviços se não estiverem rodando
if ! curl -s http://localhost:8080/health > /dev/null; then
    echo "🤖 Iniciando Agents Service..."
    python3 app.py &
    sleep 3
fi

if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "🔤 Iniciando String Comparison Service..."
    cd ../string-comparison
    python3 backend.py &
    cd ../agents
    sleep 3
fi

echo ""
echo "🌐 Abrindo frontend no navegador..."
echo "   Frontend local: file://$(pwd)/frontend.html"
echo ""

# Abrir no navegador padrão (multiplataforma)
if command -v open > /dev/null; then
    # macOS
    open frontend.html
elif command -v xdg-open > /dev/null; then
    # Linux
    xdg-open frontend.html
elif command -v start > /dev/null; then
    # Windows
    start frontend.html
else
    echo "📁 Abra manualmente o arquivo: frontend.html"
fi

echo "✅ Frontend aberto!"
echo ""
echo "📋 URLs importantes:"
echo "   Frontend:        file://$(pwd)/frontend.html"
echo "   Agents API:      http://localhost:8080"
echo "   String Compare:  http://localhost:8000"
echo ""
echo "🎯 Funcionalidades do Frontend:"
echo "   ✅ Listar agentes com acurácia real"
echo "   ✅ Criar agentes básicos e avançados"
echo "   ✅ Testar inferência e avaliação"
echo "   ✅ Monitorar todos os endpoints"
echo "   ✅ Interface única para cada agente"
echo "   ✅ Métricas de string comparison"
