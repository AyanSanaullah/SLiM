#!/bin/bash

echo "🚀 Iniciando ShellHacks Agents Services"
echo "======================================"

# Verificar se estamos no diretório correto
if [ ! -f "app.py" ]; then
    echo "❌ Execute este script no diretório agents/"
    exit 1
fi

# Parar serviços existentes
echo "🛑 Parando serviços existentes..."
killall Python 2>/dev/null || true
sleep 2

# Iniciar string-comparison service
echo "🔤 Iniciando String Comparison Service..."
if [ -d "../string-comparison" ]; then
    cd ../string-comparison
    if [ -f "backend.py" ]; then
        echo "   Iniciando em http://localhost:8000"
        python3 backend.py &
        STRING_PID=$!
        cd ../agents
        sleep 3
        
        # Verificar se iniciou
        if curl -s http://localhost:8000/health > /dev/null; then
            echo "   ✅ String Comparison Service online"
        else
            echo "   ⚠️  String Comparison Service com problemas"
        fi
    else
        echo "   ❌ backend.py não encontrado em string-comparison/"
    fi
else
    echo "   ⚠️  Diretório string-comparison não encontrado"
fi

# Iniciar agents service
echo "🤖 Iniciando Agents Service..."
echo "   Iniciando em http://localhost:8080"
python3 app.py &
AGENTS_PID=$!
sleep 5

# Verificar se iniciou
if curl -s http://localhost:8080/health > /dev/null; then
    echo "   ✅ Agents Service online"
else
    echo "   ❌ Agents Service falhou ao iniciar"
    exit 1
fi

echo ""
echo "🎉 Serviços iniciados com sucesso!"
echo "================================="
echo "🤖 Agents Service:        http://localhost:8080"
echo "🔤 String Comparison:     http://localhost:8000"
echo ""
echo "📋 Comandos úteis:"
echo "   Ver status:             python3 help.py --services"
echo "   Menu interativo:        python3 help.py --interactive"
echo "   Teste automático:       python3 test_advanced_system.py"
echo "   Parar serviços:         killall Python"
echo ""
echo "📚 Documentação:"
echo "   Comandos completos:     cat COMANDOS_COMPLETOS.md"
echo "   Comandos rápidos:       cat COMANDOS_RAPIDOS.md"
echo "   Ajuda:                  python3 help.py"
echo ""

# Manter o script rodando para mostrar logs
echo "📊 Logs dos serviços (Ctrl+C para parar):"
echo "=========================================="

# Função para cleanup quando o script for interrompido
cleanup() {
    echo ""
    echo "🛑 Parando serviços..."
    kill $AGENTS_PID 2>/dev/null || true
    kill $STRING_PID 2>/dev/null || true
    killall Python 2>/dev/null || true
    echo "✅ Serviços parados"
    exit 0
}

# Configurar trap para cleanup
trap cleanup SIGINT SIGTERM

# Aguardar indefinidamente (os serviços rodando em background)
wait
