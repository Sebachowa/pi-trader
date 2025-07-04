#!/bin/bash
# Clean up redundant markdown files

echo "🧹 Cleaning up redundant documentation files..."

# Files to delete (redundant or outdated)
FILES_TO_DELETE=(
    "README_DEPLOYMENT.md"
    "UPGRADE_PLAN.md" 
    "CLEANUP_SUMMARY.md"
    "DEEP_BLOG_ANALYSIS.md"
    "docs/dependencies_cleaned.md"
    "DEPLOYMENT.md"
    "QUICK_START_LOCAL.md"
    "GETTING_STARTED.md"
    "CONFIG_GUIDE.md"
    "RASPBERRY_PI_DEPLOYMENT_GUIDE.md"
    "BLOG_COMPARISON_ANALYSIS.md"
    "PROFESSIONAL_ALTERNATIVES.md"
    "TAX_INTEGRATION_SUMMARY.md"
    "SCANNER_COMPARISON.md"
    "DOCKER_VS_NATIVE.md"
    "README_OLD.md"
    "docs/SCANNER_FLOW.md"
)

# Delete redundant files
for file in "${FILES_TO_DELETE[@]}"; do
    if [ -f "$file" ]; then
        echo "❌ Deleting: $file"
        rm "$file"
    else
        echo "⚠️  File not found: $file"
    fi
done

# Keep important files
KEEP_FILES=(
    "README.md"
    "CLAUDE.md"
    "AUTONOMOUS_BOT_ARCHITECTURE.md"
    "data/README.md"
)

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "📁 New documentation structure:"
echo "├── README.md (main)"
echo "├── CLAUDE.md (Claude Code config)"
echo "├── AUTONOMOUS_BOT_ARCHITECTURE.md (high-level design)"
echo "└── docs/"
echo "    ├── getting-started/"
echo "    │   ├── quick-start.md"
echo "    │   ├── configuration.md"
echo "    │   └── deployment.md"
echo "    ├── technical/"
echo "    │   ├── architecture.md"
echo "    │   ├── scanner-flow.md"
echo "    │   └── logging-guide.md"
echo "    ├── features/"
echo "    │   ├── strategies.md"
echo "    │   └── tax-features.md"
echo "    └── analysis/"
echo "        ├── alternatives.md"
echo "        └── why-this-bot.md"
echo ""
echo "🎯 From 22+ files down to 12 organized files!"