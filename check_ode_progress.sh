#!/bin/bash
#
# Script to check ODE generation progress
#

OUTPUT_FOLDER="/data/karthik_data/vidprom_ode_pairs"
TOTAL_FILES=16000

echo "==========================================="
echo "ODE Generation Progress Check"
echo "==========================================="
echo ""

# Count completed files
COMPLETED=$(ls -1 $OUTPUT_FOLDER/*.pt 2>/dev/null | wc -l)
PERCENTAGE=$(printf "%.2f" $(echo "scale=4; $COMPLETED / $TOTAL_FILES * 100" | bc))

echo "📊 Progress:"
echo "  Completed: $COMPLETED / $TOTAL_FILES files"
echo "  Progress: $PERCENTAGE%"
echo ""

# Calculate estimated time remaining
FILES_PER_HOUR=160
REMAINING=$((TOTAL_FILES - COMPLETED))
HOURS_REMAINING=$(echo "scale=1; $REMAINING / $FILES_PER_HOUR" | bc)
DAYS_REMAINING=$(echo "scale=1; $HOURS_REMAINING / 24" | bc)

echo "⏱️  Estimated Time Remaining:"
echo "  Files remaining: $REMAINING"
echo "  Hours remaining: ~$HOURS_REMAINING hours"
echo "  Days remaining: ~$DAYS_REMAINING days"
echo ""

# Disk usage
DISK_USAGE=$(du -sh $OUTPUT_FOLDER 2>/dev/null | cut -f1)
echo "💾 Disk Usage: $DISK_USAGE"
echo ""

# Show latest files
echo "📁 Latest files:"
ls -lht $OUTPUT_FOLDER/*.pt 2>/dev/null | head -5
echo ""

# Check if process is running
if ps aux | grep -v grep | grep generate_ode_pairs > /dev/null; then
    echo "✅ Generation process is RUNNING"
else
    echo "⚠️  Generation process is NOT running"
    echo ""
    echo "To resume, run: ./generate_ode_pairs.sh"
fi

echo "==========================================="

