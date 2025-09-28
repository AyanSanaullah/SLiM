#!/bin/bash

echo "🚀 Iniciando ShellHacks Agents Services (Versão Corrigida)"
echo "=========================================================="

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Verificar se estamos no diretório correto
if [ ! -f "app.py" ]; then
    log_error "Execute este script no diretório agents/"
    exit 1
fi

# Parar serviços existentes
log_info "Parando serviços existentes..."
killall Python 2>/dev/null || true
sleep 3

# Verificar se as portas estão livres
check_port() {
    local port=$1
    if lsof -i :$port >/dev/null 2>&1; then
        log_warning "Porta $port ainda está em uso. Tentando liberar..."
        kill -9 $(lsof -t -i:$port) 2>/dev/null || true
        sleep 2
    fi
}

check_port 8000
check_port 8080

# Corrigir problema do NLTK
log_info "Configurando NLTK..."
python3 -c "
import ssl
import os
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import nltk
try:
    nltk.data.find('corpora/wordnet')
    print('✅ NLTK wordnet já está disponível')
except LookupError:
    print('📦 Baixando NLTK wordnet...')
    nltk.download('wordnet', quiet=True)
    print('✅ NLTK wordnet baixado com sucesso')
except Exception as e:
    print(f'⚠️  Aviso: {e}')
"

# Iniciar string-comparison service
log_info "Iniciando String Comparison Service..."
if [ -d "../string-comparison" ]; then
    cd ../string-comparison
    if [ -f "backend.py" ]; then
        log_info "Iniciando em http://localhost:8000"
        python3 backend.py &
        STRING_PID=$!
        cd ../agents
        sleep 5
        
        # Verificar se iniciou
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            log_success "String Comparison Service online"
        else
            log_warning "String Comparison Service com problemas, mas continuando..."
        fi
    else
        log_error "backend.py não encontrado em string-comparison/"
    fi
else
    log_warning "Diretório string-comparison não encontrado"
fi

# Ativar ambiente virtual se existir
if [ -d "venv" ]; then
    log_info "Ativando ambiente virtual..."
    source venv/bin/activate
fi

# Iniciar agents service
log_info "Iniciando Agents Service..."
log_info "Iniciando em http://localhost:8080"
python3 app.py &
AGENTS_PID=$!
sleep 5

# Verificar se iniciou
if curl -s http://localhost:8080/health > /dev/null 2>&1; then
    log_success "Agents Service online"
else
    log_error "Agents Service falhou ao iniciar"
    exit 1
fi

echo ""
log_success "Serviços iniciados com sucesso!"
echo "================================="
echo "🤖 Agents Service:        http://localhost:8080"
echo "🔤 String Comparison:     http://localhost:8000"
echo "🌐 Frontend:              file://$(pwd)/frontend.html"
echo ""
echo "📋 Comandos úteis:"
echo "   Abrir frontend:         ./open_frontend.sh"
echo "   Teste automático:       python3 test_advanced_system.py"
echo "   Parar serviços:         killall Python"
echo ""

# Abrir frontend automaticamente
log_info "Abrindo frontend automaticamente..."
if command -v open > /dev/null; then
    open frontend.html
elif command -v xdg-open > /dev/null; then
    xdg-open frontend.html
else
    log_info "Abra manualmente: frontend.html"
fi

echo "📊 Logs dos serviços (Ctrl+C para parar):"
echo "=========================================="

# Função para cleanup quando o script for interrompido
cleanup() {
    echo ""
    log_info "Parando serviços..."
    kill $AGENTS_PID 2>/dev/null || true
    kill $STRING_PID 2>/dev/null || true
    killall Python 2>/dev/null || true
    log_success "Serviços parados"
    exit 0
}

# Configurar trap para cleanup
trap cleanup SIGINT SIGTERM

# Aguardar indefinidamente (os serviços rodando em background)
wait
