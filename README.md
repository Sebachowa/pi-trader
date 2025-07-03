# 🤖 Trading Systems Portfolio

Este repositorio contiene un portfolio completo de sistemas de trading automatizado, diseñado para operar 24/7 con mínima intervención humana.

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────┐
│                   SISTEMA PRINCIPAL                      │
│              autonomous_trading (Nautilus)               │
│  • ML/AI Strategy Selection                              │
│  • Advanced Risk Management                              │
│  • Self-healing capabilities                             │
│  • Target: 10% annual returns                            │
└─────────────────────────────────────────────────────────┘
                          ↕️
┌─────────────────────────────────────────────────────────┐
│                    SISTEMA EDGE                          │
│                trader-pi (Raspberry Pi)                  │
│  • Lightweight CCXT-based                                │
│  • Basic strategies                                      │
│  • Remote monitoring                                     │
│  • Backup trading capability                             │
└─────────────────────────────────────────────────────────┘
```

## 📁 Estructura del Proyecto

### 🎯 Sistema Principal: `autonomous_trading/`
Sistema profesional completo basado en Nautilus Trader para servidores/workstations.

**Características:**
- ✅ ML-powered strategy selection
- ✅ Multi-asset support (crypto, stocks, forex)
- ✅ Advanced risk management with Kelly Criterion
- ✅ Self-healing and auto-recovery
- ✅ Paper trading capabilities
- ✅ Comprehensive monitoring

**Uso:**
```bash
./start_autonomous_trading.sh
```

### 📱 Sistema Edge: `trader-pi/`
Sistema ligero optimizado para dispositivos de recursos limitados.

**Características:**
- ✅ Optimizado para ARM (Raspberry Pi)
- ✅ Consumo mínimo de recursos (<1GB RAM)
- ✅ Estrategias básicas pero efectivas
- ✅ Deployment automático con GitHub Actions
- ✅ Monitoreo remoto

**Deployment:**
```bash
cd trader-pi
make deploy
```

### 📊 Sistemas Adicionales

- **`paper_trading/`** - Motor de simulación realista
- **`multi_asset_system/`** - Sistema unificado multi-activos
- **`deployment/`** - Scripts y configuraciones de deployment

## 🚀 Quick Start

### Opción 1: Sistema Principal (Producción)
```bash
# Configurar
cp autonomous_trading/config/trading_config.json.example autonomous_trading/config/trading_config.json
# Editar configuración...

# Ejecutar
./start_autonomous_trading.sh
```

### Opción 2: Sistema Raspberry Pi
```bash
# En tu Raspberry Pi
cd trader-pi
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-pi.txt
python3 run.py --paper
```

## 🔧 Configuración

### Variables de Entorno
```bash
# Binance
export BINANCE_API_KEY="your-key"
export BINANCE_API_SECRET="your-secret"

# Notificaciones (opcional)
export TELEGRAM_BOT_TOKEN="your-token"
export TELEGRAM_CHAT_ID="your-chat-id"
```

## 📈 Estrategias Disponibles

| Estrategia | Sistema Principal | Sistema Pi | Descripción |
|------------|------------------|-------------|-------------|
| Trend Following | ✅ Advanced | ✅ Simple | Sigue tendencias del mercado |
| Mean Reversion | ✅ ML-enhanced | ✅ Bollinger | Aprovecha reversiones |
| Momentum | ✅ Multi-timeframe | ✅ RSI-based | Trading de momentum |
| Market Making | ✅ Advanced | ❌ | Provee liquidez |
| ML Strategy | ✅ Ensemble | ❌ | Estrategias con IA |

## 🛡️ Risk Management

Ambos sistemas implementan:
- Stop-loss dinámico
- Position sizing basado en volatilidad
- Límites de drawdown diario
- Circuit breakers automáticos

## 📊 Monitoreo

### Sistema Principal
- Logs detallados en `logs/`
- Métricas en tiempo real
- Dashboard web (opcional)

### Sistema Pi
```bash
# Monitoreo remoto
make logs
make status
```

## 🔄 CI/CD

### GitHub Actions
El repositorio incluye workflows para:
- ✅ Tests automáticos
- ✅ Deployment a Raspberry Pi
- ✅ Notificaciones de deployment
- ✅ Rollback automático

## 📚 Documentación Adicional

- [Deployment Guide](trader-pi/DEPLOYMENT.md) - Guía completa de deployment
- [Architecture](architecture/system_design.py) - Diseño del sistema
- [Trading Strategies](autonomous_trading/strategies/) - Implementación de estrategias

## 🤝 Contribuir

1. Fork el repositorio
2. Crea tu feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add AmazingFeature'`)
4. Push al branch (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver [LICENSE](LICENSE) para detalles.

## ⚠️ Disclaimer

Este software es para fines educativos. El trading conlleva riesgos significativos. Úsalo bajo tu propia responsabilidad.

---

**Desarrollado para operar 24/7 con mínima intervención humana** 🚀