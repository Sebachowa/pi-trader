# 🚀 Inicio Rápido - Prueba Local

## 1️⃣ Configuración Inicial (5 minutos)

### Paso 1: Instalar dependencias
```bash
# Crear entorno virtual
python3 -m venv venv

# Activar entorno (Mac/Linux)
source venv/bin/activate

# Activar entorno (Windows)
# venv\Scripts\activate

# Instalar paquetes
pip install -r requirements-pi.txt
```

### Paso 2: Configurar API Keys

#### Opción A: Binance Testnet (RECOMENDADO para pruebas)
1. Ve a: https://testnet.binance.vision/
2. Crea una cuenta de prueba
3. Genera API keys de testnet

#### Opción B: Binance Real (con dinero real)
1. Ve a: https://www.binance.com/en/my/settings/api-management
2. Crea API key con permisos de:
   - ✅ Read
   - ✅ Spot Trading
   - ❌ Withdrawals (NO necesario)

### Paso 3: Editar configuración
```bash
# Editar config
nano config/config.json
```

Cambia estas líneas:
```json
{
  "exchange": {
    "api_key": "TU_API_KEY_AQUI",
    "api_secret": "TU_API_SECRET_AQUI",
    "testnet": true  // true = testnet, false = real
  }
}
```

## 2️⃣ Probar Componentes

### Test 1: Probar Scanner de Mercado
```bash
python scripts/test_scanner.py
```

Deberías ver:
```
🚀 Initializing Market Scanner...
📊 Test 1: Top Volume Symbols
Found 87 symbols with >$1M volume
Top 5: ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', ...]

🔍 Test 2: Conservative Scan
Found 3 opportunities
  BTC/USDT - trend_following (score: 82.5)
    Entry: $45234.5678
    SL: $44329.1234 | TP: $46140.0123
```

### Test 2: Probar Dashboard de Impuestos
```bash
python scripts/tax_dashboard.py
```

Verás un dashboard interactivo mostrando posiciones simuladas.

### Test 3: Ejecutar Bot en Modo Paper
```bash
python run.py --paper
```

Verás:
```
🤖 Crypto Trading Bot v1.0
📊 Mode: PAPER TRADING
🔌 Exchange: binance (testnet)
💰 Initial Balance: $10,000

[2024-01-07 15:23:45] INFO - Trading engine initialized successfully
[2024-01-07 15:23:46] INFO - Market scanner initialized
[2024-01-07 15:23:46] INFO - Scanner stats - Avg time: 3.45s, Opportunities: 12
[2024-01-07 15:23:47] INFO - Processed opportunity: BTC/USDT - trend_following (score: 78.3)
[2024-01-07 15:23:48] INFO - Opened position: BTC/USDT @ 45234.56
```

## 3️⃣ Monitoreo en Tiempo Real

### Terminal 1: Bot Principal
```bash
python run.py --paper
```

### Terminal 2: Dashboard Web (si está implementado)
```bash
python -m http.server 8080 --directory frontend
# Abre: http://localhost:8080
```

### Terminal 3: Logs en Tiempo Real
```bash
tail -f logs/trader.log
```

### Terminal 4: Dashboard de Impuestos
```bash
python scripts/tax_dashboard.py
```

## 4️⃣ Comandos Útiles para Testing

### Ver todas las opciones
```bash
python run.py --help
```

### Modo simulación rápida
```bash
# Solo 5 símbolos, scan cada 10s
python run.py --paper --symbols "BTC/USDT,ETH/USDT,BNB/USDT,SOL/USDT,ADA/USDT" --scan-interval 10
```

### Modo agresivo (más trades)
```bash
python run.py --paper --risk-level aggressive --min-score 60
```

### Modo conservador (menos trades)
```bash
python run.py --paper --risk-level conservative --min-score 85
```

## 5️⃣ Verificar que Todo Funciona

### ✅ Checklist:
- [ ] Scanner encuentra oportunidades
- [ ] Bot abre posiciones (paper)
- [ ] Stop loss y take profit funcionan
- [ ] Se generan logs
- [ ] Dashboard de impuestos muestra datos

### 📊 Métricas a Observar:
```
Después de 1 hora deberías ver:
- Símbolos escaneados: 500-1000
- Oportunidades encontradas: 20-50
- Posiciones abiertas: 1-3
- Win rate: Variable (normal 40-60%)
```

## 6️⃣ Errores Comunes

### Error: "No module named ccxt"
```bash
pip install ccxt
```

### Error: "API key invalid"
- Verifica que copiaste correctamente
- Si usas testnet, asegúrate que `testnet: true`

### Error: "Insufficient balance"
- En testnet, recarga balance en https://testnet.binance.vision/
- En real, asegúrate de tener USDT

### Scanner no encuentra oportunidades
```bash
# Reducir el score mínimo
nano config/config.json
# Cambiar "min_opportunity_score": 50
```

## 🎯 Próximos Pasos

Una vez que veas que funciona bien:

1. **Optimizar configuración** para tu estilo
2. **Probar 24 horas** en paper trading
3. **Revisar logs** y métricas
4. **Ajustar estrategias** según resultados
5. **Deploy a Raspberry Pi** cuando estés listo

## 💡 Tips para Testing

1. **Empieza con testnet** - No arriesgues dinero real
2. **Observa patrones** - ¿Cuándo abre más trades?
3. **Revisa tax dashboard** - ¿Está calculando bien?
4. **Monitorea recursos** - ¿Cuánta CPU/RAM usa?
5. **Lee los logs** - Ahí está toda la info

¡Listo para probar! 🚀