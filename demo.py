#!/usr/bin/env python3
"""
Demo rápido del Trading Bot - No requiere API keys
"""
import asyncio
import random
from datetime import datetime
import time

class TradingBotDemo:
    def __init__(self):
        self.balance = 10000
        self.positions = {}
        self.trades = 0
        
    async def run_demo(self):
        print("🤖 Trading Bot Demo - Modo Simulación")
        print("="*50)
        print(f"💰 Balance Inicial: ${self.balance:,.2f}")
        print(f"📊 Escaneando mercados cada 5 segundos...")
        print("="*50)
        
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT']
        strategies = ['Trend Following', 'Mean Reversion', 'Momentum', 'Volume Breakout']
        
        try:
            while True:
                # Simular escaneo
                print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')} - Escaneando {len(symbols)} pares...")
                await asyncio.sleep(1)
                
                # Probabilidad de encontrar oportunidad
                if random.random() > 0.6:
                    symbol = random.choice(symbols)
                    strategy = random.choice(strategies)
                    score = random.randint(70, 95)
                    
                    print(f"🎯 Oportunidad encontrada!")
                    print(f"   Símbolo: {symbol}")
                    print(f"   Estrategia: {strategy}")
                    print(f"   Score: {score}/100")
                    
                    # Simular entrada
                    if symbol not in self.positions and len(self.positions) < 3:
                        price = {
                            'BTC/USDT': 45000 + random.randint(-1000, 1000),
                            'ETH/USDT': 2500 + random.randint(-100, 100),
                            'BNB/USDT': 300 + random.randint(-20, 20),
                            'SOL/USDT': 100 + random.randint(-10, 10),
                            'ADA/USDT': 0.5 + random.random() * 0.1
                        }[symbol]
                        
                        size = self.balance * 0.1  # 10% por trade
                        self.positions[symbol] = {
                            'entry': price,
                            'size': size,
                            'strategy': strategy
                        }
                        self.trades += 1
                        
                        print(f"✅ Posición abierta: {symbol} @ ${price:,.2f}")
                        print(f"   Tamaño: ${size:,.2f}")
                        print(f"   Stop Loss: ${price * 0.98:,.2f} (-2%)")
                        print(f"   Take Profit: ${price * 1.05:,.2f} (+5%)")
                
                # Actualizar posiciones existentes
                for symbol in list(self.positions.keys()):
                    pos = self.positions[symbol]
                    # Simular movimiento de precio
                    change = random.uniform(-0.03, 0.03)
                    current = pos['entry'] * (1 + change)
                    pnl = (current - pos['entry']) / pos['entry'] * 100
                    
                    # Cerrar si hit SL o TP
                    if pnl < -2 or pnl > 5 or random.random() > 0.9:
                        profit = pos['size'] * (pnl / 100)
                        self.balance += profit
                        
                        if pnl > 0:
                            print(f"💚 Cerrada {symbol}: +${profit:.2f} ({pnl:.1f}%)")
                        else:
                            print(f"🔴 Cerrada {symbol}: -${abs(profit):.2f} ({pnl:.1f}%)")
                        
                        del self.positions[symbol]
                
                # Mostrar estado
                print(f"\n📊 Estado Actual:")
                print(f"   Balance: ${self.balance:,.2f}")
                print(f"   Posiciones abiertas: {len(self.positions)}")
                print(f"   Total trades: {self.trades}")
                
                # Tax info
                if self.trades > 0:
                    gains = self.balance - 10000
                    if gains > 0:
                        tax = gains * 0.35  # Short term
                        print(f"   💰 Ganancias: ${gains:,.2f}")
                        print(f"   🧮 Impuestos estimados: ${tax:,.2f}")
                
                await asyncio.sleep(5)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Demo detenido")
            print(f"Balance Final: ${self.balance:,.2f}")
            print(f"Ganancia/Pérdida: ${self.balance - 10000:,.2f}")
            roi = ((self.balance - 10000) / 10000) * 100
            print(f"ROI: {roi:.2f}%")

if __name__ == "__main__":
    print("\n🚀 Iniciando Demo del Trading Bot")
    print("Presiona Ctrl+C para detener\n")
    
    demo = TradingBotDemo()
    asyncio.run(demo.run_demo())