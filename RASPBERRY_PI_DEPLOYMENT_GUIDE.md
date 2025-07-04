# 🥧 Guía Completa de Deployment en Raspberry Pi

## 📋 Pre-requisitos

### En tu Raspberry Pi:
- Raspberry Pi 5 (8GB RAM recomendado)
- Raspberry Pi OS 64-bit instalado
- Conexión a internet estable
- Python 3.11+ instalado

### En tu computadora:
- Repositorio clonado
- GitHub account (para deployment automático)

## 🚀 Opción 1: Deployment Automático con GitHub Actions (RECOMENDADO)

### Paso 1: Preparar la Raspberry Pi

```bash
# En tu Raspberry Pi, abre terminal:

# 1. Actualizar sistema
sudo apt update && sudo apt upgrade -y

# 2. Instalar dependencias
sudo apt install python3-pip python3-venv git -y

# 3. Crear directorio para el bot
mkdir -p ~/trading-bot
cd ~/trading-bot

# 4. Generar SSH key para GitHub
ssh-keygen -t ed25519 -C "tu-email@ejemplo.com"
# Presiona Enter en todo (no pongas contraseña)

# 5. Mostrar tu clave pública
cat ~/.ssh/id_ed25519.pub
# COPIA ESTE TEXTO - lo necesitarás después
```

### Paso 2: Configurar GitHub

1. Ve a tu repositorio en GitHub
2. Settings → Secrets and variables → Actions
3. Añade estos secrets:

```
PI_HOST: 192.168.1.100  # IP de tu Raspberry Pi
PI_USER: pi             # Usuario de tu Pi (normalmente 'pi')
PI_SSH_KEY: -----BEGIN OPENSSH PRIVATE KEY-----
            (pega tu clave privada de tu computadora)
            -----END OPENSSH PRIVATE KEY-----
```

Para obtener tu clave SSH privada (en tu computadora):
```bash
cat ~/.ssh/id_rsa  # o id_ed25519
```

### Paso 3: Configurar el Bot

En tu computadora, edita `config/config.json`:

```json
{
  "exchange": {
    "name": "binance",
    "api_key": "TU_API_KEY_AQUI",
    "api_secret": "TU_API_SECRET_AQUI",
    "testnet": true  // Cambiar a false para trading real
  },
  "scanner": {
    "interval_seconds": 60,  // Más lento para Pi
    "max_concurrent_scans": 20  // Menos workers
  }
}
```

### Paso 4: Deploy Automático

```bash
# En tu computadora
git add .
git commit -m "Configure for Raspberry Pi deployment"
git push origin main

# ¡GitHub Actions se encarga del resto!
```

## 🔧 Opción 2: Deployment Manual

### Paso 1: Copiar archivos a la Pi

```bash
# Desde tu computadora
rsync -avz --exclude 'venv' --exclude '__pycache__' \
  ./ pi@192.168.1.100:~/trading-bot/
```

### Paso 2: Configurar en la Pi

```bash
# SSH a tu Pi
ssh pi@192.168.1.100

cd ~/trading-bot

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements-pi.txt

# Configurar API keys
nano config/config.json
# Edita con tus keys
```

### Paso 3: Primera prueba

```bash
# Probar que funciona
python run.py --paper

# Si todo está bien, verás:
# 🚀 Trading engine started...
# 📊 Scanner initialized with 200 markets
# 🔍 Scanning 98 symbols...
```

## 🏃 Ejecutar como Servicio (24/7)

### Paso 1: Instalar servicio systemd

```bash
# Copiar archivo de servicio
sudo cp trader.service /etc/systemd/system/

# Editar paths si es necesario
sudo nano /etc/systemd/system/trader.service
```

El archivo debe verse así:
```ini
[Unit]
Description=Crypto Trading Bot
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/trading-bot
Environment="PATH=/home/pi/trading-bot/venv/bin"
ExecStart=/home/pi/trading-bot/venv/bin/python run.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Paso 2: Activar servicio

```bash
# Recargar systemd
sudo systemctl daemon-reload

# Habilitar inicio automático
sudo systemctl enable trader.service

# Iniciar el bot
sudo systemctl start trader.service

# Verificar estado
sudo systemctl status trader.service
```

## 📊 Monitoreo

### Ver logs en tiempo real:
```bash
# Logs del sistema
sudo journalctl -u trader -f

# Logs del bot
tail -f ~/trading-bot/logs/trader.log
```

### Dashboard web:
```
http://IP-DE-TU-PI:8080
```

### Comandos útiles:
```bash
# Parar el bot
sudo systemctl stop trader

# Reiniciar el bot
sudo systemctl restart trader

# Ver métricas
curl http://localhost:9090/metrics
```

## 🔐 Seguridad

### 1. Configurar Firewall
```bash
# Solo permitir SSH y dashboard
sudo ufw allow 22/tcp
sudo ufw allow 8080/tcp
sudo ufw enable
```

### 2. Cambiar contraseña por defecto
```bash
passwd
# Pon una contraseña fuerte
```

### 3. Backups automáticos
```bash
# Crear script de backup
cat > ~/backup-trader.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
tar -czf ~/backups/trader_$DATE.tar.gz \
  --exclude='venv' \
  --exclude='__pycache__' \
  ~/trading-bot/
  
# Mantener solo últimos 7 backups
cd ~/backups && ls -t trader_*.tar.gz | tail -n +8 | xargs rm -f
EOF

chmod +x ~/backup-trader.sh

# Añadir a crontab (backup diario a las 2am)
(crontab -l 2>/dev/null; echo "0 2 * * * /home/pi/backup-trader.sh") | crontab -
```

## 🎯 Optimización para Raspberry Pi

### 1. Reducir carga del scanner
```json
{
  "scanner": {
    "interval_seconds": 60,
    "max_concurrent_scans": 20,
    "top_volume_count": 50,
    "whitelist": ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
  }
}
```

### 2. Swap para más memoria
```bash
# Aumentar swap a 2GB
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Cambiar: CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### 3. Overclocking (opcional)
```bash
sudo nano /boot/config.txt
# Añadir:
over_voltage=4
arm_freq=2000
```

## 🚨 Troubleshooting

### Error: "Module not found"
```bash
source ~/trading-bot/venv/bin/activate
pip install -r requirements-pi.txt
```

### Error: "Connection refused"
```bash
# Verificar que el exchange esté accesible
ping api.binance.com

# Verificar configuración
cat config/config.json
```

### Bot se detiene constantemente
```bash
# Ver logs detallados
sudo journalctl -u trader -n 100

# Aumentar memoria swap
# Reducir concurrent scans
```

## 📱 Notificaciones Telegram

```bash
# Configurar notificaciones
cp telegram_config.example.json telegram_config.json
nano telegram_config.json

# Añadir tu bot token y chat ID
{
  "bot_token": "123456:ABC-DEF...",
  "chat_id": "123456789"
}
```

## ✅ Checklist Final

- [ ] Raspberry Pi actualizada
- [ ] Python 3.11+ instalado
- [ ] API keys configuradas
- [ ] Servicio systemd activo
- [ ] Firewall configurado
- [ ] Backups automáticos
- [ ] Notificaciones funcionando
- [ ] Dashboard accesible

## 🎉 ¡Listo!

Tu bot está corriendo 24/7 en tu Raspberry Pi. Puedes:

1. Ver el dashboard: `http://IP-DE-TU-PI:8080`
2. Ver logs: `ssh pi@IP-DE-TU-PI && tail -f trading-bot/logs/trader.log`
3. Ver estado: `ssh pi@IP-DE-TU-PI && sudo systemctl status trader`

¡Disfruta las ganancias pasivas! 🚀💰