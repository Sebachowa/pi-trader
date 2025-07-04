# 📚 Documentation Index

Complete guide to the Raspberry Pi Trading Bot documentation.

## 🚀 Getting Started

New to the bot? Start here:

1. **[Quick Start Guide](getting-started/quick-start.md)** ⭐
   - 5-minute setup for demo, testnet, or live trading
   - Installation and basic configuration
   - First run and troubleshooting

2. **[Configuration Guide](getting-started/configuration.md)**
   - Complete configuration reference
   - Environment variables and settings
   - Security best practices

3. **[Deployment Guide](getting-started/deployment.md)**
   - Manual deployment to Raspberry Pi
   - Systemd service setup
   - GitHub Actions automated deployment

## 🔧 Technical Documentation

Deep dive into how the bot works:

1. **[Architecture Overview](technical/architecture.md)**
   - System design and components
   - Data flow and decision making
   - Performance characteristics

2. **[Scanner Flow](technical/scanner-flow.md)**
   - How market opportunities are detected
   - Technical indicator calculations
   - Scoring and filtering logic

3. **[Logging Guide](technical/logging-guide.md)** 🌈
   - Understanding the beautiful colored logs
   - Emoji reference and meanings
   - Monitoring and debugging

4. **[Troubleshooting Guide](troubleshooting.md)** 🔧
   - Diagnose why bot isn't finding opportunities
   - View historical logs and data
   - Common issues and solutions

## 🎯 Features

Understanding what the bot can do:

1. **[Trading Strategies](features/strategies.md)**
   - 4 built-in strategies explained
   - Configuration and optimization
   - Performance characteristics

2. **[Tax Features](features/tax-features.md)** 💰
   - Built-in capital gains calculation
   - Multi-jurisdiction support
   - Export formats and compliance

## 📊 Analysis & Comparisons

Why choose this bot:

1. **[Alternative Bots](analysis/alternatives.md)**
   - Honest comparison with Freqtrade, Jesse, etc.
   - Performance and feature comparison
   - When to choose each option

2. **[Why This Bot?](analysis/why-this-bot.md)** ⭐
   - What makes us different from tutorials
   - Professional features in simple package
   - Raspberry Pi optimization

## 📋 Additional Resources

- **[CLAUDE.md](../CLAUDE.md)** - Claude Code AI assistant configuration
- **[AUTONOMOUS_BOT_ARCHITECTURE.md](../AUTONOMOUS_BOT_ARCHITECTURE.md)** - High-level design philosophy
- **[data/README.md](../data/README.md)** - Data directory documentation

## 🗺️ Documentation Map

```
docs/
├── README.md (this file)
├── getting-started/     ← Start here for setup
│   ├── quick-start.md      ⭐ 5-minute setup
│   ├── configuration.md    🔧 All settings
│   └── deployment.md       🚀 Production deploy
├── technical/           ← How it works
│   ├── architecture.md     🏗️ System design
│   ├── scanner-flow.md     🔍 Opportunity detection
│   └── logging-guide.md    🌈 Beautiful logs
├── features/            ← What it does
│   ├── strategies.md       🎯 Trading strategies
│   └── tax-features.md     💰 Tax tracking
└── analysis/            ← Why choose this
    ├── alternatives.md     📊 Vs other bots
    └── why-this-bot.md     ⭐ Our advantages
```

## 🎯 Quick Links by Use Case

### "I'm completely new to trading bots"
1. [Why This Bot?](analysis/why-this-bot.md) - Understand what makes us special
2. [Quick Start Guide](getting-started/quick-start.md) - Get running in 5 minutes
3. [Logging Guide](technical/logging-guide.md) - Understand what you're seeing

### "I'm evaluating different bots"
1. [Alternative Bots](analysis/alternatives.md) - Compare with Freqtrade, Jesse, etc.
2. [Architecture Overview](technical/architecture.md) - Technical capabilities
3. [Trading Strategies](features/strategies.md) - What strategies are included

### "I want to deploy to production"
1. [Configuration Guide](getting-started/configuration.md) - Secure configuration
2. [Deployment Guide](getting-started/deployment.md) - Production deployment
3. [Tax Features](features/tax-features.md) - Compliance requirements

### "I'm having issues"
1. [Quick Start Guide](getting-started/quick-start.md) - Troubleshooting section
2. [Logging Guide](technical/logging-guide.md) - Understanding error messages
3. [Configuration Guide](getting-started/configuration.md) - Common configuration issues

### "I want to understand the code"
1. [Architecture Overview](technical/architecture.md) - System design
2. [Scanner Flow](technical/scanner-flow.md) - Opportunity detection logic
3. [Trading Strategies](features/strategies.md) - Strategy implementation

## 📝 Contributing to Documentation

Found an error or want to improve the docs?

1. **Small fixes**: Edit the file directly and create a pull request
2. **Major changes**: Open an issue first to discuss
3. **New sections**: Follow the existing structure and emoji conventions

### Documentation Standards
- Use emojis consistently (🚀 for getting started, 🔧 for technical, etc.)
- Include code examples where helpful
- Keep explanations clear and beginner-friendly
- Test all instructions on a real Raspberry Pi

---

**Need help?** Check the [Quick Start Guide](getting-started/quick-start.md) or create an issue on GitHub!