# 💰 Sistema de Cálculo de Impuestos

## 🎯 Características Principales

### 1. **Cálculo Automático de Impuestos**
- Rastrea TODAS las transacciones automáticamente
- Calcula ganancias/pérdidas realizadas
- Diferencia entre corto plazo (<1 año) y largo plazo (>1 año)
- Soporta FIFO, LIFO, HIFO

### 2. **Multi-Jurisdicción**
Configuraciones pre-definidas para 15+ países:
- 🇺🇸 USA (Form 8949)
- 🇪🇸 España (Modelo 720)
- 🇩🇪 Alemania (Tax-free después de 1 año!)
- 🇬🇧 UK (£12,300 allowance)
- 🇸🇬 Singapur (0% tax!)
- 🇦🇪 Dubai (0% tax!)
- Y muchos más...

### 3. **Monitor en Tiempo Real**
```bash
python scripts/tax_dashboard.py
```

Te muestra:
- 📊 Impacto fiscal de cerrar cada posición
- ⏳ Días para calificar como largo plazo
- 🎯 Oportunidades de tax loss harvesting
- 💸 Estimación de pagos trimestrales

### 4. **Reportes Automáticos**
```bash
python scripts/generate_tax_report.py --year 2024
```

Genera:
- 📄 Form 8949 (USA)
- 📊 Resumen completo JSON
- 💾 CSV para TurboTax
- 📈 Exportación para CoinTracker

## 🔧 Configuración

### Configuración Básica (USA)
```json
{
  "tax": {
    "enabled": true,
    "jurisdiction": "USA",
    "method": "FIFO",
    "short_term_rate": 0.35,
    "long_term_rate": 0.15
  }
}
```

### Configuración España
```json
{
  "tax": {
    "enabled": true,
    "jurisdiction": "SPAIN",
    "method": "FIFO",
    "short_term_rate": 0.26,
    "long_term_rate": 0.26,
    "reporting_threshold": 1000
  }
}
```

## 📊 Dashboard en Tiempo Real

```
┌─────────────────────────────────────────────────────────┐
│ 🧮 Real-Time Tax Monitor                                │
│ Jurisdiction: USA | Method: FIFO | Updated: 14:23:45    │
├─────────────────────────────────────────────────────────┤
│ Symbol  │   P&L    │ Term  │ Tax Rate │ After Tax │     │
├─────────┼──────────┼───────┼──────────┼───────────┼─────┤
│ BTC/USDT│ +$3,000  │ short │   35%    │  $1,950   │ WAIT│
│ ETH/USDT│ -$2,000  │ short │   35%    │  -$2,000  │ SELL│
│ SOL/USDT│ +$2,000  │ long  │   15%    │  $1,700   │ OK  │
└─────────────────────────────────────────────────────────┘

💰 Tax Summary
  Unrealized Gains: $3,000
  Estimated Tax: $1,350
  YTD Realized: $5,000
  Quarterly Payment: $437.50

🎯 Tax Loss Harvesting        ⏳ Wait for Long-Term
  ETH: Sell to save $700        BTC: Wait 25 days save $600
```

## 🚀 Casos de Uso

### 1. **Antes de Cerrar una Posición**
El bot automáticamente considera:
- ¿Cuánto impuesto pagaré?
- ¿Vale la pena esperar para largo plazo?
- ¿Debería hacer tax loss harvesting?

### 2. **Fin de Año**
```bash
# Ver oportunidades de optimización
python scripts/generate_tax_report.py --year 2024

# Te muestra:
- Posiciones con pérdidas para vender
- Ganancias que puedes diferir
- Estimación total de impuestos
```

### 3. **Pagos Trimestrales (USA)**
```bash
python scripts/generate_tax_report.py --estimate

# Output:
Next Quarter Payment: $1,250
Due Date: 2024-06-15
```

## 💡 Estrategias Inteligentes

### Tax Loss Harvesting Automático
- Detecta pérdidas que pueden compensar ganancias
- Sugiere ventas óptimas para reducir impuestos
- Respeta wash sale rules (USA)

### Optimización de Holding Period
- Te avisa cuando faltan pocos días para largo plazo
- Calcula el ahorro exacto de esperar
- Balancea con condiciones de mercado

### Gestión de Fin de Año
- Rebalanceo tax-efficient
- Diferir ganancias al próximo año
- Realizar pérdidas estratégicamente

## 📈 Ejemplo Real

```
Tienes BTC comprado hace 350 días con $5,000 de ganancia:

Si vendes HOY:
- Impuesto: $1,750 (35% short-term)
- Ganancia neta: $3,250

Si esperas 15 días:
- Impuesto: $750 (15% long-term)
- Ganancia neta: $4,250
- AHORRAS: $1,000! 🎉
```

## 🛡️ Compliance

- ✅ Cumple con regulaciones IRS (USA)
- ✅ Soporta múltiples métodos contables
- ✅ Registra todo para auditorías
- ✅ Exporta en formatos oficiales

## 🔍 Comandos Útiles

```bash
# Generar reporte anual
./scripts/generate_tax_report.py --year 2024

# Ver dashboard en tiempo real
./scripts/tax_dashboard.py

# Estimar próximo pago trimestral
./scripts/generate_tax_report.py --estimate

# Exportar para TurboTax
./scripts/generate_tax_report.py --format turbotax
```

¡El bot no solo tradea, también optimiza tus impuestos automáticamente! 🚀