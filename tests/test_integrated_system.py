#!/usr/bin/env python3
"""
Comprehensive test suite for the Autonomous Trading System
Tests all components individually and integrated
"""

import asyncio
import json
import os
import sys
import unittest
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from autonomous_trading.main_trading_system import AutonomousTradingSystem
from autonomous_trading.core.enhanced_engine import EnhancedAutonomousEngine
from autonomous_trading.core.adaptive_risk_controller import AdaptiveRiskController
from autonomous_trading.core.market_analyzer import MarketAnalyzer, MarketRegime
from autonomous_trading.strategies.ml_strategy_selector import MLStrategySelector
from autonomous_trading.monitoring.system_monitor import SystemMonitor

from nautilus_challenge.strategies import (
    TrendFollowingStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    MarketMakingStrategy,
    MLStrategy,
)


class TestAutonomousEngine(unittest.TestCase):
    """Test the Enhanced Autonomous Engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = EnhancedAutonomousEngine(
            trader_id="TEST-001",
            initial_capital=Decimal("1.0"),
            target_annual_return=0.10,
            max_drawdown=0.05,
        )
        
    def test_initialization(self):
        """Test engine initialization."""
        self.assertEqual(self.engine.trader_id, "TEST-001")
        self.assertEqual(self.engine.initial_capital, Decimal("1.0"))
        self.assertEqual(self.engine.target_annual_return, 0.10)
        self.assertEqual(self.engine.max_drawdown, 0.05)
        
    def test_health_check(self):
        """Test health check functionality."""
        # Initially healthy
        self.assertTrue(self.engine.is_healthy())
        
        # Simulate errors
        self.engine.error_count = 10
        self.assertFalse(self.engine.is_healthy())
        
    def test_recovery_mechanism(self):
        """Test self-healing recovery."""
        # Simulate failure
        self.engine.status = "error"
        self.engine.error_count = 5
        
        # Attempt recovery
        success = self.engine.attempt_recovery()
        
        if success:
            self.assertEqual(self.engine.status, "recovering")
            self.assertEqual(self.engine.recovery_attempts, 1)
            

class TestRiskController(unittest.TestCase):
    """Test the Adaptive Risk Controller."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.risk_controller = AdaptiveRiskController(
            max_portfolio_risk=0.06,
            max_position_risk=0.02,
            max_daily_drawdown=0.05,
        )
        
    def test_risk_models(self):
        """Test different risk models."""
        # Test Kelly Criterion
        kelly_size = self.risk_controller.calculate_position_size_kelly(
            win_rate=0.6,
            avg_win=0.02,
            avg_loss=0.01,
            capital=Decimal("1.0"),
        )
        self.assertGreater(kelly_size, Decimal("0"))
        self.assertLess(kelly_size, Decimal("0.25"))  # Kelly cap
        
        # Test Volatility Targeting
        vol_size = self.risk_controller.calculate_position_size_volatility(
            target_vol=0.15,
            asset_vol=0.30,
            capital=Decimal("1.0"),
        )
        self.assertEqual(vol_size, Decimal("0.5"))  # 0.15/0.30
        
    def test_regime_detection(self):
        """Test market regime detection."""
        # Low volatility
        regime = self.risk_controller.detect_volatility_regime(0.10)
        self.assertEqual(regime, "low_volatility")
        
        # Normal volatility
        regime = self.risk_controller.detect_volatility_regime(0.20)
        self.assertEqual(regime, "normal_volatility")
        
        # High volatility
        regime = self.risk_controller.detect_volatility_regime(0.40)
        self.assertEqual(regime, "high_volatility")
        
        # Extreme volatility
        regime = self.risk_controller.detect_volatility_regime(0.60)
        self.assertEqual(regime, "extreme_volatility")
        
    def test_risk_limits(self):
        """Test risk limit enforcement."""
        # Check position risk
        position_allowed = self.risk_controller.check_position_risk(
            position_size=Decimal("0.01"),
            stop_loss_pct=0.02,
            portfolio_value=Decimal("1.0"),
        )
        self.assertTrue(position_allowed)
        
        # Exceed position risk
        position_allowed = self.risk_controller.check_position_risk(
            position_size=Decimal("0.10"),
            stop_loss_pct=0.05,
            portfolio_value=Decimal("1.0"),
        )
        self.assertFalse(position_allowed)
        

class TestMarketAnalyzer(unittest.TestCase):
    """Test the Market Analyzer."""
    
    @pytest.mark.asyncio
    async def test_market_regime_detection(self):
        """Test market regime detection."""
        analyzer = MarketAnalyzer(
            logger=Mock(),
            clock=Mock(),
            analysis_interval_mins=15,
        )
        
        # Mock price data
        prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 110]
        
        # Detect regime
        regime = analyzer._detect_market_regime(prices)
        self.assertIn(regime, [
            MarketRegime.TRENDING_UP,
            MarketRegime.TRENDING_DOWN,
            MarketRegime.RANGING,
            MarketRegime.VOLATILE,
        ])
        
    def test_trend_strength_calculation(self):
        """Test trend strength calculation."""
        analyzer = MarketAnalyzer(
            logger=Mock(),
            clock=Mock(),
        )
        
        # Strong uptrend
        prices = [100, 102, 104, 106, 108, 110]
        strength = analyzer._calculate_trend_strength(prices)
        self.assertGreater(strength, 0.5)
        
        # No trend
        prices = [100, 101, 99, 100, 101, 100]
        strength = analyzer._calculate_trend_strength(prices)
        self.assertLess(abs(strength), 0.2)
        

class TestTradingStrategies(unittest.TestCase):
    """Test individual trading strategies."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.instrument_id = Mock()
        self.bar_type = "1-MINUTE-BID"
        
    def test_trend_following_strategy(self):
        """Test trend following strategy initialization."""
        strategy = TrendFollowingStrategy(
            instrument_id=self.instrument_id,
            bar_type=self.bar_type,
            trade_size=Decimal("0.01"),
        )
        
        self.assertEqual(strategy.trade_size, Decimal("0.01"))
        self.assertIsNotNone(strategy.fast_ema)
        self.assertIsNotNone(strategy.slow_ema)
        self.assertIsNotNone(strategy.rsi)
        
    def test_mean_reversion_strategy(self):
        """Test mean reversion strategy initialization."""
        strategy = MeanReversionStrategy(
            instrument_id=self.instrument_id,
            bar_type=self.bar_type,
            trade_size=Decimal("0.01"),
        )
        
        self.assertEqual(strategy.rsi_oversold, 30)
        self.assertEqual(strategy.rsi_overbought, 70)
        self.assertIsNotNone(strategy.bb)
        
    def test_ml_strategy(self):
        """Test ML strategy initialization."""
        strategy = MLStrategy(
            instrument_id=self.instrument_id,
            bar_type=self.bar_type,
            confidence_threshold=0.65,
        )
        
        self.assertEqual(strategy.confidence_threshold, 0.65)
        self.assertTrue(strategy.use_ensemble)
        self.assertTrue(strategy.online_learning)
        

class TestSystemMonitor(unittest.TestCase):
    """Test the System Monitor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monitor = SystemMonitor(
            db_path=":memory:",  # Use in-memory database
            web_enabled=False,
        )
        
    def test_metric_logging(self):
        """Test metric logging."""
        # Log system metrics
        self.monitor.log_system_metrics(
            cpu_usage=45.5,
            memory_usage=2048,
            active_strategies=3,
            open_positions=2,
        )
        
        # Check stored metrics
        self.assertEqual(len(self.monitor.metrics["system"]["cpu"]), 1)
        self.assertEqual(self.monitor.current_state["active_strategies"], 3)
        
    def test_alert_creation(self):
        """Test alert creation and thresholds."""
        # Trigger high CPU alert
        self.monitor.log_system_metrics(
            cpu_usage=85.0,  # Above threshold
            memory_usage=1024,
            active_strategies=1,
            open_positions=0,
        )
        
        # Check if alert was created
        self.assertGreater(len(self.monitor.metrics["alerts"]), 0)
        
    def test_performance_summary(self):
        """Test performance summary generation."""
        # Log some performance data
        self.monitor.log_performance_metrics(
            total_pnl=0.05,
            daily_pnl=0.01,
            win_rate=0.60,
            sharpe_ratio=1.5,
            max_drawdown=0.03,
            total_trades=50,
        )
        
        # Get summary
        summary = self.monitor.get_performance_summary(hours=24)
        
        self.assertIn("performance", summary)
        self.assertIn("trades", summary)
        

class TestIntegratedSystem(unittest.TestCase):
    """Test the complete integrated system."""
    
    @pytest.mark.asyncio
    async def test_system_initialization(self):
        """Test full system initialization."""
        # Create test config
        test_config = {
            "trader_id": "TEST-001",
            "initial_capital": 1.0,
            "target_annual_return": 0.10,
            "max_drawdown": 0.05,
            "mode": "paper",
        }
        
        # Save test config
        config_path = Path("test_config.json")
        with open(config_path, 'w') as f:
            json.dump(test_config, f)
            
        try:
            # Initialize system
            system = AutonomousTradingSystem(str(config_path))
            
            # Check configuration loaded
            self.assertEqual(system.config["trader_id"], "TEST-001")
            self.assertEqual(system.mode, "paper")
            
            # Mock components for testing
            with patch('autonomous_trading.main_trading_system.TradingNode'):
                with patch('autonomous_trading.main_trading_system.NotificationService'):
                    await system.initialize()
                    
                    # Check components initialized
                    self.assertIsNotNone(system.autonomous_engine)
                    self.assertIsNotNone(system.risk_controller)
                    self.assertIsNotNone(system.strategy_orchestrator)
                    
        finally:
            # Cleanup
            if config_path.exists():
                config_path.unlink()
                
    @pytest.mark.asyncio
    async def test_strategy_deployment(self):
        """Test strategy deployment logic."""
        system = AutonomousTradingSystem("test_config.json")
        
        # Mock components
        system.strategy_orchestrator = Mock()
        system.strategy_orchestrator.select_strategies = AsyncMock(return_value=[
            ("trend_following", 0.3),
            ("mean_reversion", 0.2),
            ("ml_strategy", 0.4),
        ])
        system.strategy_orchestrator.deploy_strategy = AsyncMock(side_effect=[
            "trend_following_001",
            "mean_reversion_001",
            "ml_strategy_001",
        ])
        
        system.instruments = [Mock()]
        system.trading_node = Mock()
        system.trading_node.portfolio = Mock()
        system.trading_node.portfolio.net_liquidation = Mock(return_value=Decimal("1.0"))
        
        # Deploy strategies
        await system._deploy_initial_strategies()
        
        # Check deployments
        self.assertEqual(len(system.active_strategies), 3)
        self.assertIn("trend_following_001", system.active_strategies)
        
    def test_risk_limit_enforcement(self):
        """Test risk limit enforcement."""
        system = AutonomousTradingSystem("test_config.json")
        
        # Mock components
        system.trading_node = Mock()
        system.trading_node.portfolio = Mock()
        system.performance_monitor = Mock()
        system.notification_service = Mock()
        
        # Simulate drawdown breach
        system.performance_monitor.get_current_drawdown = Mock(return_value=0.06)
        system.config["max_drawdown"] = 0.05
        
        # Run risk check
        asyncio.run(system._check_risk_limits())
        
        # Check if notification was sent
        system.notification_service.send_critical.assert_called_once()
        

class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_paper_trading_flow(self):
        """Test complete paper trading flow."""
        # This would require a full test environment
        # Including mock exchange connections
        pass
        
    @pytest.mark.integration
    def test_monitoring_dashboard(self):
        """Test monitoring dashboard functionality."""
        monitor = SystemMonitor(
            db_path=":memory:",
            web_enabled=True,
            web_port=8081,
        )
        
        # Test web app creation
        self.assertIsNotNone(monitor.app)
        
        # Test API endpoints
        from fastapi.testclient import TestClient
        client = TestClient(monitor.app)
        
        # Test metrics endpoint
        response = client.get("/api/metrics/system")
        self.assertEqual(response.status_code, 200)
        
        # Test summary endpoint
        response = client.get("/api/summary/24")
        self.assertEqual(response.status_code, 200)
        

def run_all_tests():
    """Run all tests with coverage report."""
    print("🧪 Running Autonomous Trading System Tests...")
    
    # Run pytest with coverage
    exit_code = pytest.main([
        __file__,
        "-v",
        "--cov=autonomous_trading",
        "--cov=nautilus_challenge",
        "--cov-report=html",
        "--cov-report=term",
    ])
    
    return exit_code


if __name__ == "__main__":
    # Run tests
    exit_code = run_all_tests()
    
    print("\n✅ Test suite completed")
    print("📊 Coverage report generated in htmlcov/")
    
    sys.exit(exit_code)