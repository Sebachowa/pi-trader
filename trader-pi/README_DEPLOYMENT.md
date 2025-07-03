# 🚀 Sistema de Deployment Automático para Trading Bot

## 🎯 ¿Qué hemos creado?

Un sistema **SUPER CÓMODO** de deployment automático que:

1. **Deploy con un simple `git push`** - Push a GitHub = Deploy automático a tu Raspberry Pi
2. **Rollback automático** - Si algo falla, vuelve a la versión anterior
3. **Notificaciones Telegram** - Te avisa cuando se deploya o si hay errores
4. **Health checks** - Verifica que todo funcione después del deploy
5. **Docker opcional** - Puedes usar Docker o instalación directa

## 🛠️ Configuración Rápida (10 minutos)

### 1️⃣ Prepara tu GitHub

```bash
# Genera las SSH keys
chmod +x scripts/setup_github_secrets.sh
./scripts/setup_github_secrets.sh
```

### 2️⃣ Configura tu Raspberry Pi

```bash
# Copia y ejecuta en tu Pi
ssh pi@tu-raspberry.local
curl -O https://raw.githubusercontent.com/tu-usuario/tu-repo/main/scripts/setup_pi.sh
bash setup_pi.sh
```

### 3️⃣ Añade los Secrets a GitHub

Ve a tu repositorio → Settings → Secrets → Actions y añade:

- `PI_HOST`: IP de tu Raspberry (ej: `192.168.1.100`)
- `PI_USER`: Usuario SSH (normalmente `pi`)
- `PI_SSH_KEY`: La clave privada que generó el script
- `TELEGRAM_BOT_TOKEN`: (Opcional) Para notificaciones
- `TELEGRAM_CHAT_ID`: (Opcional) Tu ID de Telegram

### 4️⃣ ¡Listo! Ahora solo haz push

```bash
git add .
git commit -m "Mi trading bot"
git push origin main
```

**¡Y se desplegará automáticamente!** 🎉

## 📱 Comandos Súper Útiles

### Con Makefile (lo más cómodo)

```bash
make deploy       # Despliega a tu Pi
make logs        # Ve los logs en tiempo real
make status      # Checa el status
make monitor     # Monitorea el bot
make docker-run  # Corre con Docker
```

### Deployment Manual

```bash
# Opción 1: Con el script
./deploy_to_pi.sh raspberrypi.local pi

# Opción 2: Trigger desde GitHub
# Ve a Actions → Run workflow
```

## 🐳 Opción Docker (Más Pro)

### Deploy con Docker

```bash
# En tu Pi
cd /home/pi/trader-bot
docker-compose up -d

# Con monitoring incluido
docker-compose --profile monitoring up -d
```

### Ventajas de Docker
- ✅ Aislamiento completo
- ✅ Fácil rollback
- ✅ Límites de recursos automáticos
- ✅ Logs centralizados

## 📊 Monitoreo

### Logs en Tiempo Real

```bash
# Desde tu compu
make logs

# O directo
ssh pi@raspberrypi.local "journalctl -u trader -f"
```

### Dashboard (si usas Docker + Monitoring)

- Grafana: `http://tu-raspberry:3000`
- Prometheus: `http://tu-raspberry:9090`

## 🔧 Personalización

### Cambiar el Trigger de Deploy

Edita `.github/workflows/deploy.yml`:

```yaml
on:
  push:
    branches:
      - main        # Cambia a tu branch
      - produccion  # O añade más branches
```

### Añadir más Notificaciones

```yaml
# Discord
- uses: sarisia/actions-status-discord@v1
  with:
    webhook: ${{ secrets.DISCORD_WEBHOOK }}

# Slack
- uses: 8398a7/action-slack@v3
  with:
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

## 🚨 Troubleshooting

### El deploy falla

1. Checa los logs en GitHub Actions
2. Verifica la conexión SSH:
   ```bash
   ssh -i ~/.ssh/github_actions_deploy pi@tu-raspberry
   ```

### El bot no arranca

```bash
# En tu Pi
cd /home/pi/trader-bot
source venv/bin/activate
python3 run.py --dry-run
```

### Rollback manual

```bash
# En tu Pi
cd /home/pi
ls -la trader-bot_backup_*  # Lista backups
cp -r trader-bot_backup_XXXXXX trader-bot  # Restaura
sudo systemctl restart trader
```

## 🎯 Workflow Típico

1. **Desarrollas** en tu compu
2. **Pruebas** localmente: `make run`
3. **Commit y Push**: `git push`
4. **GitHub Actions** se encarga de todo
5. **Recibes notificación** en Telegram
6. **Verificas logs** si quieres: `make logs`

## 🔐 Seguridad

- ✅ Las claves SSH están en GitHub Secrets (encriptadas)
- ✅ La config con API keys está en Secrets
- ✅ Conexión SSH segura
- ✅ Firewall configurado en Pi

## 💡 Tips Pro

1. **Crea branches para diferentes ambientes**
   - `main` → Producción
   - `test` → Para probar

2. **Usa los health checks**
   ```bash
   make status  # Checa que todo esté bien
   ```

3. **Monitorea el performance**
   - El Pi tiene recursos limitados
   - Usa `htop` para ver el uso

4. **Backups automáticos**
   - Se crean automáticamente en cada deploy
   - Se mantienen los últimos 3

## 🎉 ¡Ya está!

Ahora tienes un sistema de deployment **profesional y automático**. 

Cada vez que hagas cambios, solo necesitas:

```bash
git add .
git commit -m "Nueva estrategia"
git push
```

¡Y tu bot se actualizará solo en la Raspberry Pi! 🤖🚀